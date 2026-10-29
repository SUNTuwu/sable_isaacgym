"""Independent physical stance/yaw teacher. No locomotion bank is modified."""
import argparse,json,math,os,signal,time
from pathlib import Path
from isaacgym import gymapi,gymtorch
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from migration_env import MigrationRobot,create_migration_env
from migration_config import student_config
from legged_gym.envs.base.legged_robot import LeggedRobot
from model_mapping import quat_apply,quat_inverse
from teacher_bank_core import update,atomic,state_equal

AMPLITUDE=math.pi/6
OMEGA=1/3
PERIOD=2*math.pi/OMEGA
SETTLE=1.
EPISODE=SETTLE+2*PERIOD
BUDGET=1
STOP=False
# Survival-first tuning authorized for the independent yaw teacher. The angular
# waveform, actual event counting, observation schema and PPO/resume stay fixed.
NONFALL_SCALE=.25
FALL_BASE_COST=100.
REWARD_PROFILE='yaw_survival_anchor_v1'

def night_control_path():
    override=os.environ.get('SABLE_NIGHT_CONTROL_FILE')
    return Path(override) if override else Path(os.environ['SABLE_GYM_ROOT'])/'deployment/night_20261029/control.json'

def verify_loaded_checkpoint(policy,optimizer,checkpoint):
    model_equal=state_equal(policy.state_dict(),checkpoint['model_state_dict'])
    optimizer_equal=state_equal(optimizer.state_dict(),checkpoint['optimizer_state_dict'])
    if not model_equal or not optimizer_equal:raise RuntimeError('Yaw resume failed exact model/Adam verification')
    return dict(model_state_equal_before_exploration_control=True,optimizer_state_equal_before_exploration_control=True,
                verification='recursive torch.equal for model parameters and Adam state, before any exploration ceiling is applied')

def exploration_schedule(tuning,previous,iteration,observed_max_deg,floor_deg):
    """Monotone ceiling only. Pure metadata calculation, no policy/state reset."""
    request=tuning.get('exploration',{})
    wanted=request.get('max_std_deg')
    if wanted is None and previous is None:return None
    if not math.isfinite(observed_max_deg) or observed_max_deg<floor_deg-1e-6:
        raise ValueError('Invalid effective exploration standard deviation')
    state=dict(previous or {})
    if wanted is not None and (isinstance(wanted,bool) or not isinstance(wanted,(float,int)) or not math.isfinite(wanted) or wanted<=0):
        raise ValueError('max_std_deg must be a positive finite number')
    cap=max(floor_deg,min(observed_max_deg,state.get('cap_deg',observed_max_deg)))
    target=max(floor_deg,min(float(wanted) if wanted is not None else state['target_deg'],state.get('target_deg',cap)))
    if not state or target<state['target_deg']-1e-9:
        iterations=request.get('anneal_iterations',40)
        if isinstance(iterations,bool) or not isinstance(iterations,int) or not 1<=iterations<=1000:
            raise ValueError('anneal_iterations must be an integer in [1,1000]')
        state=dict(start_iteration=int(iteration),start_cap_deg=cap,target_deg=target,
                   anneal_iterations=iterations,complete=False,completed_iteration=None)
    progress=min(1.,max(0.,(iteration-state['start_iteration'])/state['anneal_iterations']))
    scheduled=state['start_cap_deg']+(state['target_deg']-state['start_cap_deg'])*progress
    state['cap_deg']=max(floor_deg,min(cap,scheduled))
    state['complete']=state['cap_deg']<=state['target_deg']+1e-9
    # The rollout starts at iteration; its resulting checkpoint is iteration+1.
    if state['complete'] and state.get('completed_iteration') is None:state['completed_iteration']=int(iteration)+1
    state.update(floor_deg=floor_deg,updated_iteration=int(iteration),active_joints=14,
                 rule='monotone upper bound; never increase sampled noise; mean/critic/Adam retained')
    return state

def apply_exploration_schedule(policy,tuning,previous,iteration):
    # Apply once BEFORE a rollout; keep the same ceiling for all PPO minibatches.
    factor=.35*180/math.pi
    effective=policy.log_std.exp().clamp(policy.minimum_std,policy.maximum_std)[:14]
    state=exploration_schedule(tuning,previous,iteration,float(effective.max())*factor,policy.minimum_std*factor)
    if state is not None:
        policy.maximum_std=min(policy.maximum_std,state['cap_deg']/factor)
        with torch.no_grad():policy.log_std[:14].clamp_(max=math.log(policy.maximum_std))
    return state

def contact_transition(stance,armed,count,contact,flight,budget):
    """Startup never relocates anchors; one actual flight consumes one budget."""
    lift=stance&flight&armed
    next_count=count+lift
    extra=lift&(next_count>budget)
    # Previous stance is false only after a confirmed flight or at startup.
    # count>0 and armed exclude startup; over-budget landings cannot reanchor.
    reanchor=(~stance)&contact&armed&(next_count>0)&(next_count<=budget)
    next_stance=contact|(stance&~flight)
    return next_stance,armed|contact,next_count,lift,extra,reanchor

def tuning_multipliers(tuning):
    return {name:max(.25,min(4.,float(mult))) for name,mult in tuning.get('multipliers',{}).items()}
def scaled_reward_terms(terms,multipliers,fall,dt):
    """Rates are integrated once; excess lift and fall are single-event costs."""
    scaled={name:NONFALL_SCALE*value*multipliers.get(name,1.) for name,value in terms.items()}
    costs=sum((-v).clamp(min=0) for v in scaled.values())*dt
    scaled['fall']=-(FALL_BASE_COST+costs)*fall/dt
    return scaled

def set_task(degrees,budget):
    global AMPLITUDE,OMEGA,PERIOD,EPISODE,BUDGET
    AMPLITUDE=math.radians(degrees);OMEGA=math.radians(10)/AMPLITUDE;PERIOD=2*math.pi/OMEGA;EPISODE=SETTLE+2*PERIOD;BUDGET=budget
def stop(*_):
    global STOP
    STOP=True
def heading(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wave(age):
    t=(age-SETTLE).clamp(min=0)
    return AMPLITUDE*torch.sin(OMEGA*t),AMPLITUDE*OMEGA*torch.cos(OMEGA*t)*(age>=SETTLE)

class Robot(MigrationRobot):
    def _init_buffers(self):
        super()._init_buffers()
        n=self.num_envs;dev=self.device
        self.count=torch.zeros(n,4,device=dev)
        self.stance=torch.zeros(n,4,device=dev,dtype=torch.bool)
        self.armed=self.stance.clone()
        self.anchors=torch.zeros(n,4,3,device=dev)
        self.start=torch.zeros(n,3,device=dev)
        self.start_yaw=torch.zeros(n,device=dev)
        self.reset_q=self.default_dof_pos[:,self.active].repeat(n,1)
        self.pose_choice=torch.zeros(n,device=dev,dtype=torch.long)
        self.pose_bank=None
        self.tips=torch.stack([pts[(pts.square().sum(-1)).argmax()]+radius*(pts[(pts.square().sum(-1)).argmax()]-pts[1-(pts.square().sum(-1)).argmax()])/(pts[1]-pts[0]).norm() for _,pts,radius in self.mapping.capsules])
        self.last_target=self.executed_target.clone()
        self.initial_gravity=self.reference_gravity.clone()
    def fk_feet(self,q,height=None):
        n=len(q);p=torch.tensor(self.mapping.manifest['init_position'],device=self.device).repeat(n,1)
        if height is not None:p=p+torch.stack((torch.zeros_like(height),torch.zeros_like(height),height),1)
        r=torch.tensor(self.mapping.manifest['init_quat_xyzw'],device=self.device).repeat(n,1)
        full=self.default_dof_pos.repeat(n,1).clone();full[:,self.active]=q
        full[:,self.dependent]=-full[:,self.hip]
        pos,rot=self.mapping.forward(p,r,full,self.dof_names)
        indices=[i for i,_,_ in self.mapping.capsules]
        tips=pos[:,indices]+quat_apply(rot[:,indices],self.tips[None].expand(n,-1,-1))
        surface=self.mapping.task_measurements(p,r,full,self.dof_names)[3]
        return tips,surface
    def build_pose_bank(self,size=256):
        # Genuine Cartesian foothold randomization, not joint noise mislabeled as XY.
        base=self.default_dof_pos[:,self.active].repeat(size,1)
        feet,surface=self.fk_feet(base)
        torch.manual_seed(5039)
        radius=.02*torch.rand(size,4,device=self.device).sqrt()
        angle=2*math.pi*torch.rand(size,4,device=self.device)
        target=feet[:,:,:2]+torch.stack((radius*angle.cos(),radius*angle.sin()),-1)
        q=base.clone().requires_grad_(True);height=torch.full((size,),-.01,device=self.device,requires_grad=True)
        opt=torch.optim.Adam([{'params':[q],'lr':.02},{'params':[height],'lr':.001}])
        for iteration in range(600):
            if iteration==300:
                opt.param_groups[0]['lr']=.005;opt.param_groups[1]['lr']=.0003
            toe,z=self.fk_feet(q,height)
            loss=(toe[:,:,:2]-target).square().sum()+z.square().sum()+1e-7*(q-base).square().sum()
            opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                q.clamp_(self.hard_limits[self.active,0],self.hard_limits[self.active,1]);q[:,-1]=base[:,-1];height.clamp_(-.025,.005)
        q=q.detach();height=height.detach();toe,z=self.fk_feet(q,height)
        error=(toe[:,:,:2]-target).norm(dim=2).max(1).values
        radius_actual=(toe[:,:,:2]-feet[:,:,:2]).norm(dim=2).max(1).values
        good=(error<.001)&(z.abs().max(1).values<.001)&(radius_actual<=.0201)
        print('IK_DIAGNOSTICS',dict(error_quantiles=torch.quantile(error,torch.tensor([0.,.5,1.],device=self.device)).tolist(),surface_best=float(z.abs().max(1).values.min()),radius_best=float(radius_actual.min()),initial_surface=surface[0].tolist()),flush=True)
        assert int(good.sum())>=16,('Too few reachable reset poses',int(good.sum()))
        self.pose_bank=q[good];self.pose_toes=toe[good];self.pose_height=height[good]
        return dict(requested=size,valid=int(good.sum()),max_target_error_m=float(error[good].max()),max_radius_m=float(radius_actual[good].max()),max_surface_error_m=float(z[good].abs().max()),source='physical_manifest frame5 standard stand; waist pitch locked to its standard angle')
    def _resample_commands(self,ids):
        if hasattr(self,'commands16'):self.commands16[ids,:3]=0
        self.commands[ids,:3]=0
    def reset_idx(self,ids):
        if hasattr(self,'pose_bank') and self.pose_bank is not None and len(ids):
            self.pose_choice[ids]=torch.randint(len(self.pose_bank),(len(ids),),device=self.device)
        super().reset_idx(ids)
        if not hasattr(self,'count') or not len(ids):return
        self.count[ids]=0;self.stance[ids]=False;self.armed[ids]=False
        self.start[ids]=self.root_states[ids,:3];self.start_yaw[ids]=heading(self.root_states[ids,3:7])
        if self.pose_bank is not None:
            self.reset_q[ids]=self.pose_bank[self.pose_choice[ids]]
            self.anchors[ids]=self.pose_toes[self.pose_choice[ids]]+self.env_origins[ids,None,:]
        self.executed_target[ids]=self.reset_q[ids];self.student_target[ids]=self.reset_q[ids]
        self.full_student_target[ids[:,None],self.active]=self.reset_q[ids]
        self.actions[ids]=0;self.last_actions[ids]=0;self.last_target[ids]=self.reset_q[ids]
    def _reset_dofs(self,ids):
        if not hasattr(self,'pose_bank') or self.pose_bank is None:return super()._reset_dofs(ids)
        self.dof_pos[ids]=self.default_dof_pos;self.dof_pos[ids[:,None],self.active]=self.pose_bank[self.pose_choice[ids]]
        self.dof_pos[ids,self.dependent]=-self.dof_pos[ids,self.hip];self.dof_vel[ids]=0
        actors=self.robot_actor_indices[ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(actors),len(ids))
    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state
        self.root_states[ids,:3]=torch.tensor(self.mapping.manifest['init_position'],device=self.device)+self.env_origins[ids]
        if self.pose_bank is not None:self.root_states[ids,2]+=self.pose_height[self.pose_choice[ids]]
        self.root_states[ids,3:7]=torch.tensor(self.mapping.manifest['init_quat_xyzw'],device=self.device)
        self.root_states[ids,7:13]=0
        actors=self.robot_actor_indices[ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_all),gymtorch.unwrap_tensor(actors),len(ids))
    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._pending_reset_ids=torch.zeros(0,device=self.device,dtype=torch.long)
        self._refresh_proprio_state();self.gait_signals.update(self);self.knee_contact_detector.update()
    def compute_observations(self):
        self._refresh_proprio_state()
        self.obs_buf=torch.zeros(self.num_envs,326,device=self.device)
        self.privileged_obs_buf=torch.zeros(self.num_envs,92,device=self.device)
    def observation(self):
        target,rate=wave(self.episode_length_buf*self.dt)
        yaw=heading(self.root_states[:,3:7])-self.start_yaw
        anchor=quat_apply(quat_inverse(self.base_quat)[:,None].expand(-1,4,-1),self.anchors-self.root_states[:,:3,None].transpose(1,2))
        return torch.cat((self._proprio(),self.base_lin_vel*2,torch.sin(yaw)[:,None],torch.cos(yaw)[:,None],rate[:,None]/math.radians(10),target[:,None]/AMPLITUDE,
                          (self.root_states[:,:2]-self.start[:,:2])/.1,anchor.flatten(1)*10,self.stance.float(),((BUDGET-self.count)/BUDGET).clamp(-3,1),self.reset_q-self.default_dof_pos[:,self.active]),1)
    def step(self,actions):
        actions=actions.clone();actions[:,-1]=0
        target=self.reset_q+.35*actions
        self.last_target.copy_(self.executed_target)
        return super().step((target-self.default_dof_pos[:,self.active])/.35)
    def check_termination(self):
        age=self.episode_length_buf*self.dt;target,rate=wave(age)
        yaw=heading(self.root_states[:,3:7])-self.start_yaw
        error=torch.atan2(torch.sin(yaw-target),torch.cos(yaw-target))
        feet=self.rigid_states[:,self.feet_indices]
        toe=feet[:,:,:3]+quat_apply(feet[:,:,3:7],self.tips[None].expand(self.num_envs,-1,-1))
        force=self.contact_forces[:,self.feet_indices,2]
        contact=(force>1.)|(self.stance&(force>.3))
        flight=(~contact)&(self.foot_clearance>.001)
        self.stance,self.armed,self.count,lift,extra,reanchor=contact_transition(
            self.stance,self.armed,self.count,contact,flight,BUDGET)
        # Accept the permitted foothold relocation exactly once at touchdown.
        # XY only: changing an anchor never replenishes the whole-episode budget.
        self.anchors[:,:,:2]=torch.where(reanchor[:,:,None],toe[:,:,:2],self.anchors[:,:,:2])
        tilt=(self.projected_gravity*self.reference_gravity).sum(1)
        low=(self.root_states[:,2]-self.env_origins[:,2])<.055
        body=(self.contact_forces[:,self.termination_contact_indices].norm(dim=-1)>2).any(1)
        knee=self.knee_ground_contact.any(1)&bool(self.cfg.termination.knee_contact_enabled)
        self.true_fall=(tilt<.5)|low|body|knee|((self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs()>.08)
        self.time_out_buf=(age>=EPISODE)&~self.true_fall;self.reset_buf=self.true_fall|self.time_out_buf
        self.extras['time_outs']=self.time_out_buf.clone()
        if not torch.isfinite(self.root_states).all():raise FloatingPointError('nonfinite physics')
        anchor_error=(toe[:,:,:2]-self.anchors[:,:,:2]).norm(dim=2)
        tilt_error=(self.projected_gravity-self.reference_gravity).square().sum(1)
        terms=dict(angle=5*torch.exp(-(error/math.radians(10)).square()),rate=1.5*torch.exp(-((self.heading_yaw_rate-rate)/math.radians(5)).square()),
            angle_error=-.5*(error/AMPLITUDE).square().clamp(max=4),body_position=-2*((self.root_states[:,:2]-self.start[:,:2])/.03).square().sum(1).clamp(max=10),
            feet=-2*(anchor_error/.015).square().mean(1).clamp(max=10),gravity=-2*tilt_error,
            height=-((self.root_states[:,2]-self.start[:,2])/.03).square().clamp(max=5),air=-.5*(~contact).float().sum(1),
            smooth=-.1*((self.executed_target-self.last_target)/self.dt/3).square().mean(1),torque=-.05*(self.torques/self.torque_limits.clamp(min=1e-3)).square().mean(1),
            extra_liftoff=-5*extra.float().sum(1)/self.dt,alive=torch.ones_like(error))
        terms=scaled_reward_terms(terms,tuning_multipliers(getattr(self,'yaw_tuning',{})),self.true_fall,self.dt)
        self.task_reward=sum(terms.values())*self.dt
        self.terms=terms;self.terminal_task_obs=self.observation().clone()
        self.metrics=dict(contact_force=force.clone(),error=error.clone(),yaw=yaw.clone(),target=target.clone(),rate_error=(self.heading_yaw_rate-rate).clone(),foot_error=anchor_error.clone(),count=self.count.clone(),
            root_drift=(self.root_states[:,:2]-self.start[:,:2]).norm(dim=1),fall=self.true_fall.clone(),age=age.clone(),q=self.dof_pos[:,self.active].clone(),root=self.root_states.clone(),toe=toe.clone())
    def compute_reward(self):
        self.rew_buf.copy_(self.task_reward)

class Policy(nn.Module):
    def __init__(self,obs):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(obs,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,out))
        self.actor=net(15);self.critic=net(1);self.log_std=nn.Parameter(torch.full((15,),math.log(.15)))
        self.minimum_std=.01;self.maximum_std=.5
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def distribution(self,obs):return Normal(self.actor(obs),self.log_std.exp().clamp(self.minimum_std,self.maximum_std))
    def value(self,obs):return self.critic(obs).squeeze(-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--duration',type=float,default=1200);p.add_argument('--envs',type=int,default=512);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--deadline',type=float);a=p.parse_args()
    import hashlib
    a.run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(357)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if a.resume:
        previous_meta=json.loads((a.resume/'config.json').read_text());set_task(previous_meta['amplitude_deg'],previous_meta['liftoff_budget_per_foot_per_episode'])
    cfg=student_config(Path(__file__).parent/'bootstrap',stage='S3',num_envs=a.envs+16)
    cfg.env.episode_length_s=EPISODE;cfg.domain_rand.push_robots=False;cfg.migration.freeze_waist=True
    for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
    cfg.rewards.scales.alive=1.
    import migration_env
    migration_env.MigrationRobot=Robot;e=create_migration_env(cfg,noise=False)
    if a.resume:
        saved=torch.load(a.resume/'reset_poses.pt',map_location=e.device);e.pose_bank=saved['q'];e.pose_toes=saved['toe'];e.pose_height=saved['height'];pose_report=json.loads((a.resume/'initial_poses.json').read_text())
    else:pose_report=e.build_pose_bank(64 if a.smoke else 256)
    atomic(a.run/'initial_poses.json',pose_report)
    torch.save(dict(q=e.pose_bank.cpu(),toe=e.pose_toes.cpu(),height=e.pose_height.cpu()),a.run/'reset_poses.pt')
    e.reset();obs=e.observation();policy=Policy(obs.shape[1]).to(e.device);opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
    resume_iteration=0
    if a.resume:
        receipt=json.loads((a.resume/'checkpoint.json').read_text());cp=torch.load(receipt['path'],map_location=e.device);policy.load_state_dict(cp['model_state_dict']);opt.load_state_dict(cp['optimizer_state_dict']);resume_iteration=cp['iteration']
        verified=verify_loaded_checkpoint(policy,opt,cp)
        atomic(a.run/'resume.json',dict(source=str(a.resume),checkpoint=receipt['path'],source_iteration=resume_iteration,
            source_checkpoint_sha256=hashlib.sha256(Path(receipt['path']).read_bytes()).hexdigest(),**verified))
    exploration_state=cp.get('metadata',{}).get('exploration_schedule') if a.resume else None
    evaluator=Policy(obs.shape[1]).to(e.device);evaluator.load_state_dict(policy.state_dict())
    n=a.envs;sl=slice(n,None);ev_ids=torch.arange(n,n+16,device=e.device)
    iteration=resume_iteration;start=time.time();end=a.deadline or start+a.duration;last_eval=start;eval_active=True;eval_data=[];eval_alive=torch.ones(16,device=e.device,dtype=torch.bool);eval_count=0;eval_iteration=resume_iteration;pass_streak=0;pending_stage=False;last_checkpoint=start
    metadata=dict(task='YAW_SINE_STANCE',axis='yaw',amplitude_deg=math.degrees(AMPLITUDE),peak_rate_deg_s=10,period_s=PERIOD,episode_s=EPISODE,liftoff_budget_per_foot_per_episode=BUDGET,extra_liftoff_cost=5*NONFALL_SCALE,
        reward_profile=REWARD_PROFILE,nonfall_reward_scale=NONFALL_SCALE,fall_base_cost=FALL_BASE_COST,
        fall_cost_rule='100 + sum of scaled negative nonfall costs on the same transition',
        anchor_rule='First actual permitted touchdown latches that toe XY; startup and excess landings do not relocate anchors',
        reward_multipliers={},exploration_control_supported=True,exploration_schedule=exploration_state,
        exploration_floor_deg=policy.minimum_std*.35*180/math.pi,exploration_active_joints=14,
        action='initial_stance_q + 0.35*action, physical limits, waist pitch locked at standard stand',observation_size=obs.shape[1],command='angular velocity of sine; target angle is its integrated state',train_envs=n,eval_envs=16,deadline_unix=end,pid=os.getpid(),status='training')
    atomic(a.run/'config.json',metadata)
    def save(reason):
        cp=a.run/('model_%06d.pt'%iteration);torch.save(dict(model_state_dict=policy.state_dict(),optimizer_state_dict=opt.state_dict(),iteration=iteration,metadata=metadata),cp)
        atomic(a.run/'checkpoint.json',dict(path=str(cp),iteration=iteration,reason=reason))
    try:
        while time.time()<end and not STOP:
            control_path=night_control_path()
            try:e.yaw_tuning=json.loads(control_path.read_text()).get('yaw',{})
            except (OSError,ValueError):e.yaw_tuning={}
            multipliers=tuning_multipliers(e.yaw_tuning)
            if metadata['reward_multipliers']!=multipliers:
                metadata.update(reward_multipliers=multipliers,extra_liftoff_cost=5*NONFALL_SCALE*multipliers.get('extra_liftoff',1.))
                atomic(a.run/'config.json',metadata)
            reload=e.yaw_tuning.get('reload')
            if reload and getattr(e,'reload_token',None)!=reload['token']:
                source=Path(reload['path']).resolve();assert Path(os.environ['SABLE_GYM_ROOT'])/'runs' in source.parents
                import hashlib
                assert hashlib.sha256(source.read_bytes()).hexdigest()==reload['sha256']
                checkpoint=torch.load(str(source),map_location=e.device);policy.load_state_dict(checkpoint['model_state_dict']);opt.load_state_dict(checkpoint['optimizer_state_dict'])
                baseline_meta=checkpoint['metadata'];set_task(baseline_meta['amplitude_deg'],baseline_meta['liftoff_budget_per_foot_per_episode']);e.max_episode_length=math.ceil(EPISODE/e.dt);e.reset();e.reload_token=reload['token'];eval_active=False;eval_data=[];eval_count=0;last_eval=0;pass_streak=0
                metadata.update(amplitude_deg=math.degrees(AMPLITUDE),period_s=PERIOD,episode_s=EPISODE,liftoff_budget_per_foot_per_episode=BUDGET);atomic(a.run/'config.json',metadata);atomic(a.run/'reload_receipt.json',reload)
            exploration_state=apply_exploration_schedule(policy,e.yaw_tuning,exploration_state,iteration)
            if metadata.get('exploration_schedule')!=exploration_state:
                metadata['exploration_schedule']=exploration_state;atomic(a.run/'config.json',metadata)
            storage=[[] for _ in range(7)];rs=0;term_sums={}
            for _ in range(48):
                obs=e.observation()
                with torch.no_grad():
                    dist=policy.distribution(obs[:n]);act=dist.sample();act[:,-1]=0
                    action=torch.cat((act,evaluator.actor(obs[sl])),0);action[:,-1]=0
                    storage[0].append(obs[:n].clone());storage[1].append(act);storage[2].append(dist.log_prob(act)[:,:14].sum(-1));storage[5].append(policy.value(obs[:n]))
                _,_,reward,done,info=e.step(action)
                with torch.no_grad():tv=policy.value(e.terminal_task_obs[:n])*e.time_out_buf[:n]
                storage[3].append(reward[:n].clone());storage[4].append(done[:n].clone());storage[6].append(tv);rs+=float(reward[:n].mean())
                for name,value in e.terms.items():term_sums[name]=term_sums.get(name,0.)+value[:n].mean().detach()*e.dt
                if eval_active:
                    m=e.metrics;eval_alive&=~m['fall'][sl]
                    eval_data.append({k:v[sl].detach().clone() for k,v in m.items()})
                    eval_count+=1
                    if eval_count>=math.ceil(EPISODE/e.dt):
                        arrays={k:torch.stack([v[k] for v in eval_data]).cpu().numpy() for k in eval_data[0]}
                        errors=arrays['error'];ages=arrays['age'];valid=ages>=SETTLE
                        valid_samples=int(valid.sum())
                        # A policy that falls during settling fails evaluation, not the trainer.
                        rmse=float(np.rad2deg(np.sqrt((errors[valid]**2).mean()))) if valid_samples else None
                        peak=float(np.rad2deg(np.abs(errors[valid]).max())) if valid_samples else None
                        counts=arrays['count'].max(0)
                        report=dict(amplitude_deg=math.degrees(AMPLITUDE),liftoff_budget=BUDGET,iteration=iteration,checkpoint_iteration=eval_iteration if 'eval_iteration' in locals() else 0,angle_rmse_deg=rmse,peak_error_deg=peak,valid_tracking_samples=valid_samples,survival_fraction=float(eval_alive.float().mean()),max_liftoffs_per_foot=counts.max(0).tolist(),
                            accepted=bool(valid_samples and bool(eval_alive.all()) and bool((counts<=BUDGET).all()) and rmse<=5 and peak<=10),criteria='16 environments, 2 full sine cycles, all survive, RMSE<=5deg, peak<=10deg, each foot<=%d liftoff in entire episode'%BUDGET)
                        atomic(a.run/'evaluation.json',report)
                        np.savez_compressed(a.run/'latest_replay.npz',**arrays)
                        print('EVAL',json.dumps(report),flush=True);eval_active=False;eval_data=[]
                        pass_streak=pass_streak+1 if report['accepted'] else 0
                        if pass_streak>=2 and BUDGET==1:
                            atomic(a.run/'stage30_pass.json',report);pending_stage=True
            if pending_stage:storage[4][-1].fill_(True);storage[6][-1].zero_()
            with torch.no_grad():last=policy.value(e.observation()[:n])
            losses=update(policy,opt,storage,last,4);iteration+=1
            assert np.isfinite(losses).all()
            if pending_stage:
                set_task(45,2);e.max_episode_length=math.ceil(EPISODE/e.dt);e.reset();pass_streak=0;pending_stage=False;last_checkpoint=start;last_eval=0
                metadata.update(amplitude_deg=45,period_s=PERIOD,episode_s=EPISODE,liftoff_budget_per_foot_per_episode=2);atomic(a.run/'config.json',metadata)
            active_std_deg=(policy.log_std.exp().clamp(policy.minimum_std,policy.maximum_std)[:14]*.35*180/math.pi).detach().cpu().tolist()
            atomic(a.run/'status.json',dict(metadata,iteration=iteration,mean_reward_per_step=rs/48,reward_terms_per_step={name:float(value/48) for name,value in term_sums.items()},losses=losses,exploration_std_deg=active_std_deg+[0.],exploration_std_active_deg=active_std_deg,updated_unix=time.time()))
            with (a.run/'history.jsonl').open('a') as history:history.write(json.dumps({'time':time.time(),'iteration':iteration,'reward':rs/48,'amplitude_deg':math.degrees(AMPLITUDE)})+'\n')
            if iteration==resume_iteration+1:
                save('smoke' if a.smoke else 'first_update')
                if not a.smoke:
                    baseline_path=Path(os.environ['SABLE_GYM_ROOT'])/'deployment/night_20261029/yaw_baseline.json'
                    if not baseline_path.exists():
                        import hashlib
                        baseline_checkpoint=a.run/('model_%06d.pt'%iteration);atomic(baseline_path,dict(path=str(baseline_checkpoint),sha256=hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest()))
            if a.smoke:break
            if time.time()-last_checkpoint>=180:save('periodic_independent_of_eval');last_checkpoint=time.time()
            if not eval_active and time.time()-last_eval>=180:
                save('periodic');evaluator.load_state_dict(policy.state_dict());eval_iteration=iteration;e.reset_idx(ev_ids);eval_alive.fill_(True);eval_count=0;eval_active=True;last_eval=time.time()
        save('final');atomic(a.run/'status.json',dict(metadata,status='smoke_complete' if a.smoke else 'completed',iteration=iteration,updated_unix=time.time()))
    except BaseException as exc:
        save('failure_recovery')
        atomic(a.run/'status.json',dict(metadata,status='failed',iteration=iteration,error=repr(exc)));raise
if __name__=='__main__':main()
