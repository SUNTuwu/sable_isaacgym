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
from teacher_bank_core import update,atomic

AMPLITUDE=math.pi/6
OMEGA=1/3
PERIOD=2*math.pi/OMEGA
SETTLE=1.
EPISODE=SETTLE+2*PERIOD
STOP=False
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
                          (self.root_states[:,:2]-self.start[:,:2])/.1,anchor.flatten(1)*10,self.stance.float(),(1-self.count).clamp(-3,1),self.reset_q-self.default_dof_pos[:,self.active]),1)
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
        lift=self.stance&flight&self.armed
        self.count+=lift;self.armed|=contact
        # Once lifted, do not count again until actual support has reappeared.
        self.stance=torch.where(contact,torch.ones_like(contact),torch.where(flight,torch.zeros_like(contact),self.stance))
        extra=lift&(self.count>1)
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
        costs=sum((-v).clamp(min=0) for v in terms.values())*self.dt
        terms['fall']=-(20+costs)*self.true_fall/self.dt
        self.task_reward=sum(terms.values())*self.dt
        self.terms=terms;self.terminal_task_obs=self.observation().clone()
        self.metrics=dict(error=error.clone(),yaw=yaw.clone(),target=target.clone(),rate_error=(self.heading_yaw_rate-rate).clone(),foot_error=anchor_error.clone(),count=self.count.clone(),
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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--duration',type=float,default=1200);p.add_argument('--envs',type=int,default=512);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(357)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    cfg=student_config(Path(__file__).parent/'bootstrap',stage='S3',num_envs=a.envs+16)
    cfg.env.episode_length_s=EPISODE;cfg.domain_rand.push_robots=False;cfg.migration.freeze_waist=True
    for name in vars(cfg.rewards.scales):setattr(cfg.rewards.scales,name,0.)
    cfg.rewards.scales.alive=1.
    import migration_env
    migration_env.MigrationRobot=Robot;e=create_migration_env(cfg,noise=False)
    pose_report=e.build_pose_bank(64 if a.smoke else 256);atomic(a.run/'initial_poses.json',pose_report)
    torch.save(dict(q=e.pose_bank.cpu(),toe=e.pose_toes.cpu(),height=e.pose_height.cpu()),a.run/'reset_poses.pt')
    e.reset();obs=e.observation();policy=Policy(obs.shape[1]).to(e.device);opt=torch.optim.Adam(policy.parameters(),lr=3e-4)
    evaluator=Policy(obs.shape[1]).to(e.device);evaluator.load_state_dict(policy.state_dict())
    n=a.envs;sl=slice(n,None);ev_ids=torch.arange(n,n+16,device=e.device)
    iteration=0;start=time.time();end=start+a.duration;last_eval=start;eval_active=True;eval_data=[];eval_alive=torch.ones(16,device=e.device,dtype=torch.bool);eval_count=0
    metadata=dict(task='YAW_SINE_STANCE',axis='yaw',amplitude_deg=30,peak_rate_deg_s=10,period_s=PERIOD,episode_s=EPISODE,liftoff_budget_per_foot_per_episode=1,extra_liftoff_cost=5,
        action='initial_stance_q + 0.35*action, physical limits, waist pitch locked at standard stand',observation_size=obs.shape[1],command='angular velocity of sine; target angle is its integrated state',train_envs=n,eval_envs=16,deadline_unix=end,pid=os.getpid(),status='training')
    atomic(a.run/'config.json',metadata)
    def save(reason):
        cp=a.run/('model_%06d.pt'%iteration);torch.save(dict(model_state_dict=policy.state_dict(),optimizer_state_dict=opt.state_dict(),iteration=iteration,metadata=metadata),cp)
        atomic(a.run/'checkpoint.json',dict(path=str(cp),iteration=iteration,reason=reason))
    try:
        while time.time()<end and not STOP:
            storage=[[] for _ in range(7)];rs=0
            for _ in range(48):
                obs=e.observation()
                with torch.no_grad():
                    dist=policy.distribution(obs[:n]);act=dist.sample();act[:,-1]=0
                    action=torch.cat((act,evaluator.actor(obs[sl])),0);action[:,-1]=0
                    storage[0].append(obs[:n].clone());storage[1].append(act);storage[2].append(dist.log_prob(act)[:,:14].sum(-1));storage[5].append(policy.value(obs[:n]))
                _,_,reward,done,info=e.step(action)
                with torch.no_grad():tv=policy.value(e.terminal_task_obs[:n])*e.time_out_buf[:n]
                storage[3].append(reward[:n].clone());storage[4].append(done[:n].clone());storage[6].append(tv);rs+=float(reward[:n].mean())
                if eval_active:
                    m=e.metrics;eval_alive&=~m['fall'][sl]
                    eval_data.append({k:v[sl].detach().cpu().numpy() for k,v in m.items()})
                    eval_count+=1
                    if eval_count>=math.ceil(EPISODE/e.dt):
                        errors=np.stack([v['error'] for v in eval_data]);ages=np.stack([v['age'] for v in eval_data]);valid=ages>=SETTLE
                        rmse=np.rad2deg(np.sqrt((errors[valid]**2).mean()));peak=np.rad2deg(np.abs(errors[valid]).max())
                        counts=np.stack([v['count'] for v in eval_data]).max(0)
                        report=dict(iteration=iteration,checkpoint_iteration=eval_iteration if 'eval_iteration' in locals() else 0,angle_rmse_deg=float(rmse),peak_error_deg=float(peak),survival_fraction=float(eval_alive.float().mean()),max_liftoffs_per_foot=counts.max(0).tolist(),
                            accepted=bool(eval_alive.all()) and bool((counts<=1).all()) and rmse<=5 and peak<=10,criteria='16 environments, 2 full sine cycles, all survive, RMSE<=5deg, peak<=10deg, each foot<=1 liftoff in entire episode')
                        atomic(a.run/'evaluation.json',report)
                        np.savez_compressed(a.run/'latest_replay.npz',**{k:np.stack([v[k] for v in eval_data]) for k in eval_data[0]})
                        print('EVAL',json.dumps(report),flush=True);eval_active=False;eval_data=[]
            with torch.no_grad():last=policy.value(e.observation()[:n])
            losses=update(policy,opt,storage,last,4);iteration+=1
            assert np.isfinite(losses).all()
            atomic(a.run/'status.json',dict(metadata,iteration=iteration,mean_reward_per_step=rs/48,losses=losses,exploration_std_deg=(policy.log_std.exp()*.35*180/math.pi).detach().cpu().tolist(),updated_unix=time.time()))
            if iteration==1:save('smoke' if a.smoke else 'first_update')
            if a.smoke:break
            if not eval_active and time.time()-last_eval>=180:
                save('periodic');evaluator.load_state_dict(policy.state_dict());eval_iteration=iteration;e.reset_idx(ev_ids);eval_alive.fill_(True);eval_count=0;eval_active=True;last_eval=time.time()
        save('final');atomic(a.run/'status.json',dict(metadata,status='smoke_complete' if a.smoke else 'completed',iteration=iteration,updated_unix=time.time()))
    except BaseException as exc:
        atomic(a.run/'status.json',dict(metadata,status='failed',iteration=iteration,error=repr(exc)));raise
if __name__=='__main__':main()
