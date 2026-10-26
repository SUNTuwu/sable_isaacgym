"""Official rsl_rl PPO with audited saves, fixed deadline and independent replay."""
import argparse,datetime,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from preflight import create_env
from sable_config import SableCfg,SablePPO,MODEL
from legged_gym.utils.helpers import class_to_dict,set_seed
from rsl_rl.runners import OnPolicyRunner
from run_io import atomic_json,append_json,read_json,sha256

class DeadlineReached(Exception):pass
STOP=False
def request_stop(*_):
    global STOP
    STOP=True

class AuditedRunner(OnPolicyRunner):
    def log(self,locs,width=80,pad=35):
        self.completed=int(locs['it'])+1
        std_min_before=float(self.alg.actor_critic.std.min())
        if self.min_policy_std:
            with torch.no_grad():self.alg.actor_critic.std.clamp_(min=self.min_policy_std)
        self.tot_timesteps+=self.num_steps_per_env*self.env.num_envs
        elapsed=locs['collection_time']+locs['learn_time'];self.tot_time+=elapsed
        rows=np.asarray(self.env.last_episode_records[-1024:])
        metrics={'mean_reward':float(statistics.mean(locs['rewbuffer'])) if len(locs['rewbuffer']) else None,
            'mean_episode_length_s':float(statistics.mean(locs['lenbuffer'])*self.env.dt) if len(locs['lenbuffer']) else None,
            'fps':self.num_steps_per_env*self.env.num_envs/max(elapsed,.001),
            'value_loss':float(locs['mean_value_loss']),'surrogate_loss':float(locs['mean_surrogate_loss']),
            'noise_std':float(self.alg.actor_critic.std.mean()),'episodes_recent':len(rows),
            'noise_std_min_before_floor':std_min_before,'noise_std_floor':self.min_policy_std}
        metrics.update(learning_rate=float(self.alg.learning_rate),
            value_estimate_mean=float(self.alg.storage.values.mean()),
            value_estimate_abs_max=float(self.alg.storage.values.abs().max()),
            return_mean=float(self.alg.storage.returns.mean()),
            return_abs_max=float(self.alg.storage.returns.abs().max()))
        if len(rows):metrics.update(mean_duration_s=float(rows[:,0].mean()),fall_fraction=float(rows[:,1].mean()),
            displacement_m=float(rows[:,2].mean()),velocity_rmse=float(rows[:,3].mean()),
            max_coupling_error_rad=float(rows[:,4].max()),min_foot_touchdowns=float(rows[:,5].mean()))
        if len(rows) and rows.shape[1]>6:metrics["knee_ground_failure_fraction"]=float(rows[:,6].mean())
        self.status.update(iteration=self.completed,total_steps=self.tot_timesteps,metrics=metrics,
            updated_unix=time.time(),status='training',command_range=list(self.env.command_ranges['lin_vel_x']),
            updates_this_run=self.completed-self.start_iteration)
        atomic_json(self.out/'status.json',self.status)
        append_json(self.out/'metrics.jsonl',dict(iteration=self.completed,time=time.time(),**metrics))
        for k,v in metrics.items():
            if v is not None:self.writer.add_scalar('Audit/'+k,v,self.completed)
        if self.completed%10==0 or self.completed==1:print('PROGRESS',json.dumps(self.status),flush=True)
        if not all(torch.isfinite(p).all() for p in self.alg.actor_critic.parameters()):raise RuntimeError('Nonfinite policy')
        if STOP or time.time()>=self.deadline or (self.out/'STOP').exists():raise DeadlineReached()
        # Keep early progress too, but require a useful sample before claiming a reward milestone.
        reward=metrics['mean_reward']
        if reward is not None and len(rows)>=100 and self.completed>=20 and reward>self.best_reward+.5:
            self.best_reward=reward
            cp=self.save_checkpoint('reward_progress')
            append_json(self.out/'milestones.jsonl',{'kind':'training_reward','iteration':self.completed,'mean_reward':reward,'checkpoint':cp.name,'time':time.time(),'accepted_locomotion':False})
        if self.completed%50==0:self.save_checkpoint('periodic')
        review_due=bool(self.posture_review_at and time.time()>=self.posture_review_at and not self.posture_review_done)
        if (self.completed-self.start_iteration==50 or self.completed%200==0 or review_due) and time.time()<self.deadline-600:
            cp=self.save_checkpoint('evaluation')
            self.status['status']='evaluating';atomic_json(self.out/'status.json',self.status)
            report=self.out/'evaluations'/('eval_%06d.json'%self.completed)
            with (self.out/'evaluation.log').open('a') as log:
                try:
                    result=subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'--checkpoint',str(cp),'--output',str(report)],stdout=log,stderr=subprocess.STDOUT,timeout=min(480,self.deadline-time.time()-20))
                    eval_code=result.returncode
                except subprocess.TimeoutExpired:eval_code=124
            ev=read_json(report)
            if eval_code==0 and ev:
                atomic_json(self.out/'latest_evaluation.json',ev)
                if review_due:
                    self.posture_review_done=True
                    atomic_json(self.out/'posture_review.json',{'created_unix':time.time(),'training_started_unix':self.status['started_unix'],'scheduled_after_s':self.env.cfg.posture.review_after_seconds,'elapsed_s':time.time()-self.status['started_unix'],'tolerance_deg':self.env.cfg.posture.hip_stand_tolerance_deg,'iteration':self.completed,'report':str(report),'decision_pending':True})
                score=ev['summary']['score'];best=read_json(self.out/'best_evaluation.json',{})
                if score>best.get('summary',{}).get('score',-1e9):
                    atomic_json(self.out/'best_evaluation.json',ev)
                    append_json(self.out/'milestones.jsonl',{'kind':'independent_evaluation','iteration':self.completed,'score':score,'checkpoint':cp.name,'time':time.time(),'accepted_locomotion':False})
                # Do not promote a sliding/noise-driven displacement to a gait curriculum.
                if self.status.get('reward_phase') not in ['lift','guided'] and ev['summary']['survival_16s']>=.9 and ev['summary']['moving_all_four_fraction']>=.8:
                    self.env.command_ranges['lin_vel_x']=[.05,.20]
                    self.status['stage']='forward_0.05_to_0.20'
            else:append_json(self.out/'events.jsonl',{'event':'evaluation_failed','code':eval_code,'iteration':self.completed,'time':time.time()})
            # Evaluation is a separate simulator; current training rollout is intact.

    def save(self,path,infos=None):
        # Upstream writes iteration 0 before advancing its global counter. Use actual completed updates.
        # A timed independent evaluation can coincide with upstream's save interval.
        # Keep its already-recorded checkpoint bytes instead of rewriting only the reason.
        existing=self.out/'checkpoints'/('model_%06d.pt'%self.completed)
        if existing.exists():return existing
        return self.save_checkpoint('upstream')

    def save_checkpoint(self,reason):
        cp=self.out/'checkpoints'/('model_%06d.pt'%self.completed);cp.parent.mkdir(exist_ok=True)
        tmp=cp.with_suffix('.partial')
        torch.save({'model_state_dict':self.alg.actor_critic.state_dict(),'optimizer_state_dict':self.alg.optimizer.state_dict(),
            'iter':self.completed,'infos':{'reason':reason,'runtime_sha256':self.runtime_hash,'policy_joint_names':MODEL['policy_joint_names']}},str(tmp))
        os.replace(str(tmp),str(cp))
        atomic_json(self.out/'checkpoint.json',{'path':str(cp),'iteration':self.completed,'sha256':sha256(cp),'runtime_sha256':self.runtime_hash,'reason':reason,'updated_unix':time.time()})
        return cp

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--deadline',required=True);p.add_argument('--resume');p.add_argument('--critic-checkpoint');p.add_argument('--phase',choices=['balance','walk','gait','lift','guided'],default='walk');p.add_argument('--iterations',type=int,default=1000000)
    p.add_argument('--reference-strength',type=float,default=.25);p.add_argument('--mirror-policy',action='store_true');a=p.parse_args()
    if a.mirror_policy and a.phase!='guided':p.error('Projected policy requires the verified 52D guided observations')
    if not 0<=a.reference_strength<=1:p.error('Reference strength must be in [0,1]')
    if a.critic_checkpoint and not a.resume:p.error('--critic-checkpoint requires --resume')
    deadline=datetime.datetime.fromisoformat(a.deadline).timestamp()
    if time.time()>=deadline:raise SystemExit('Deadline already passed; refusing training')
    out=a.out.resolve();assert str(out).startswith('/root/gpufree-data/')
    out.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    cfg=SableCfg();cfg.commands.ranges.lin_vel_x=[.04,.12]
    ppo=SablePPO();ppo.policy.init_noise_std=.12;ppo.policy.actor_hidden_dims=[256,128,64];ppo.policy.critic_hidden_dims=[256,128,64]
    ppo.algorithm.entropy_coef=.001
    gait_phase=a.phase in ['gait','lift','guided'];ppo.runner.min_policy_std=0.
    if a.critic_checkpoint or gait_phase:ppo.algorithm.learning_rate=.0001
    cfg.rewards.tracking_sigma=.005
    if a.phase in ['walk','gait','lift','guided']:
        # v3's progress depended strongly on sensor jitter. Learn the mean policy
        # with clean observations first; keep friction randomization and PPO exploration.
        cfg.noise.noise_level=0.
        cfg.commands.ranges.lin_vel_x=[.08,.15]
        cfg.rewards.scales.tracking_lin_vel=3.
        cfg.rewards.scales.tracking_ang_vel=.2
        cfg.rewards.scales.feet_air_time=.5
        cfg.rewards.foot_air_time_target=.08
        cfg.rewards.tracking_sigma=.0025
    if gait_phase:
        cfg.gait.enabled=True;cfg.env.num_observations=50;cfg.commands.heading_command=True
        cfg.commands.ranges.lin_vel_x=[.04,.12];cfg.rewards.tracking_sigma=.005
        cfg.rewards.scales.tracking_ang_vel=1.;cfg.rewards.scales.gait_contact=2.
        cfg.rewards.scales.gait_clearance=-2.;cfg.rewards.scales.feet_air_time=1.
        ppo.policy.init_noise_std=.08;ppo.algorithm.entropy_coef=.003
    if a.phase in ['lift','guided']:
        # Give each foot 200 ms to swing and make true clearance more valuable.
        # Keep modest exploration instead of collapsing into a dragging policy.
        cfg.gait.period_s=.8;cfg.commands.ranges.lin_vel_x=[.025,.06]
        cfg.rewards.scales.tracking_lin_vel=2.;cfg.rewards.scales.gait_contact=3.
        cfg.rewards.scales.gait_clearance=-8.
        ppo.policy.init_noise_std=.06;ppo.runner.min_policy_std=.04
        ppo.algorithm.learning_rate=.00005;ppo.algorithm.desired_kl=.005
    if a.phase=='guided':
        cfg.reference.enabled=True;cfg.reference.strength=a.reference_strength;cfg.env.num_observations=52;cfg.gait.height_profile='sine_squared'
        cfg.gait.contact_phase_independent_of_strength=True;cfg.gait.clearance_relative_to_target=True
        cfg.rewards.scales.gait_clearance=-32.  # Four-foot mean: strengthen the active foot's smooth height error.
        cfg.commands.ranges.lin_vel_x=[.03,.06];cfg.rewards.tracking_sigma=.000625
        cfg.rewards.scales.tracking_lin_vel=3.;ppo.policy.init_noise_std=.04
    set_seed(42)
    if a.mirror_policy:
        from equivariant_policy import register_runner
        register_runner();ppo.runner.policy_class_name='SymmetricActorCritic'
    env=create_env(a.num_envs,noise=True,cfg=cfg)
    runner=AuditedRunner(env,class_to_dict(ppo),str(out/'tensorboard'),device='cuda:0')
    env.last_episode_records=[]
    runner.out=out;runner.deadline=deadline;runner.completed=0;runner.best_reward=-1e9
    runner.min_policy_std=ppo.runner.min_policy_std
    runtime=Path(__file__).parent
    hashes={str(f.relative_to(runtime)):sha256(f) for f in sorted(runtime.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.suffix in ['.py','.json','.xml','.obj']}
    atomic_json(out/'runtime_manifest.json',{'files':hashes,'source':str(runtime),'created_unix':time.time()})
    runner.runtime_hash=sha256(out/'runtime_manifest.json')
    atomic_json(out/'config.json',{'env':class_to_dict(cfg),'train':class_to_dict(ppo),'deadline':a.deadline,'phase':a.phase,'mirror_policy':a.mirror_policy})
    runner.status={'status':'starting','stage':'forward_0.04_to_0.12','pid':os.getpid(),'run_dir':str(out),'deadline':a.deadline,
        'runtime_sha256':runner.runtime_hash,'physics_device':'cuda:0','gpu_name':torch.cuda.get_device_name(),
        'num_envs':a.num_envs,'model_frame':5,'waist_pitch_deg':-40,'total_mass_kg':MODEL['total_mass_kg'],
        'actuators':13,'policy_actions':12,'accepted_locomotion':False,'started_unix':time.time()}
    if a.resume:
        if gait_phase:
            previous=torch.load(a.resume,map_location='cuda:0');state=runner.alg.actor_critic.state_dict()
            added_inputs=state['actor.0.weight'].shape[1]-previous['model_state_dict']['actor.0.weight'].shape[1]
            for key,value in previous['model_state_dict'].items():
                if state[key].shape==value.shape:state[key]=value
                elif key in ['actor.0.weight','critic.0.weight'] and (value.shape[1],state[key].shape[1]) in [(48,50),(50,52)]:
                    state[key].zero_();state[key][:,:value.shape[1]]=value
                else:raise ValueError('Unsupported policy transfer shape: '+key)
            runner.alg.actor_critic.load_state_dict(state);runner.alg.actor_critic.std.data.fill_(ppo.policy.init_noise_std)
            runner.current_learning_iteration=int(previous['iter'])
            runner.status.update(optimizer_restored=False,clock_observations_added=added_inputs,exploration_reset_std=ppo.policy.init_noise_std)
        else:runner.load(a.resume,load_optimizer=not bool(a.critic_checkpoint))
        runner.completed=runner.current_learning_iteration
        runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
        runner.status['parent_checkpoint']={'path':a.resume,'sha256':sha256(a.resume),'iteration':runner.completed}
        if a.critic_checkpoint:
            donor=torch.load(a.critic_checkpoint,map_location='cuda:0')
            state=runner.alg.actor_critic.state_dict()
            for key in state:
                if key.startswith('critic.'):state[key]=donor['model_state_dict'][key]
            runner.alg.actor_critic.load_state_dict(state)
            runner.status['critic_checkpoint']={'path':a.critic_checkpoint,'sha256':sha256(a.critic_checkpoint),'iteration':int(donor['iter'])}
            runner.status['optimizer_restored']=False
    runner.start_iteration=runner.completed
    runner.status['stage']='crawl_clock_0.04_to_0.12' if a.phase=='gait' else 'walk_0.08_to_0.15' if a.phase=='walk' else 'forward_0.04_to_0.12'
    if a.phase=='lift':runner.status['stage']='slow_crawl_lift_0.025_to_0.06'
    if a.phase=='guided':runner.status['stage']='reference_plus_residual_0.03_to_0.06'
    runner.status['kinematic_reference_enabled']=bool(cfg.reference.enabled)
    runner.status['reference_strength']=cfg.reference.strength if cfg.reference.enabled else 0.
    runner.status['swing_height_target_m']=cfg.gait.swing_height_m*(cfg.reference.strength if cfg.reference.enabled else 1.)
    runner.status['swing_peak_balance_weight']=cfg.rewards.scales.swing_peak_balance
    runner.status['reward_phase']=a.phase
    runner.status['mirror_policy']=a.mirror_policy
    runner.status['hip_stand_tolerance_deg']=cfg.posture.hip_stand_tolerance_deg
    runner.status['hip_stand_reference_deg']=[float(np.degrees(cfg.init_state.default_joint_angles[n])) for n in ['L_hip_pitch','R_hip_pitch']]
    runner.posture_review_at=runner.status['started_unix']+cfg.posture.review_after_seconds
    runner.posture_review_done=False
    runner.status['posture_review_after_unix']=runner.posture_review_at
    atomic_json(out/'status.json',runner.status);runner.save_checkpoint('initial')
    try:
        runner.learn(a.iterations,init_at_random_ep_len=False)
        runner.status['status']='iterations_complete'
    except DeadlineReached:runner.status['status']='stopped_at_deadline' if time.time()>=deadline else 'stopped_requested'
    except BaseException as error:
        runner.status.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        runner.save_checkpoint('final');runner.status.update(iteration=runner.completed,updated_unix=time.time())
        atomic_json(out/'status.json',runner.status)
        if runner.writer:runner.writer.flush();runner.writer.close()
        env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
