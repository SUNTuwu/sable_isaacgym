"""Fresh-process deterministic replay. No optimization, no training state reuse."""
import argparse,json,time,math
from pathlib import Path
from isaacgym import gymapi
import numpy as np
import torch
from rsl_rl.modules import ActorCritic
from preflight import create_env
from sable_config import SableCfg
from run_io import atomic_json,sha256,read_json
from priority_metrics import PriorityMetrics
from velocity_metrics import VelocityMetrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=16.);p.add_argument('--num-envs',type=int,default=24);p.add_argument('--zero',action='store_true')
    p.add_argument('--front-friction-scale',type=float);p.add_argument('--command-profile',choices=['constant','steps'],default='constant');a=p.parse_args()
    torch.manual_seed(947);np.random.seed(947)
    run=a.checkpoint.parent.parent;config=read_json(run/'config.json')
    cfg=SableCfg();cfg.env.episode_length_s=a.seconds;cfg.commands.resampling_time=1000.
    for key,value in config['env'].get('material_friction',{}).items():setattr(cfg.material_friction,key,value)
    if a.front_friction_scale is not None:cfg.material_friction.front_scale=a.front_friction_scale
    cfg.commands.heading_command=config['env']['commands']['heading_command']
    if config['env'].get('gait',{}).get('enabled'):
        for key,value in config['env']['gait'].items():setattr(cfg.gait,key,value)
        cfg.env.num_observations=config['env']['env']['num_observations']
    if config['env'].get('reference',{}).get('enabled'):
        for key,value in config['env']['reference'].items():
            if key!='file':setattr(cfg.reference,key,value)
    for key,value in config['env'].get('posture',{}).items():setattr(cfg.posture,key,value)
    env=create_env(a.num_envs,noise=False,cfg=cfg);env.reset_noise=.015
    actor_class=ActorCritic
    if config.get('mirror_policy'):
        from equivariant_policy import SymmetricActorCritic
        actor_class=SymmetricActorCritic
    actor=actor_class(env.num_obs,env.num_obs,env.num_actions,**config['train']['policy']).to('cuda:0')
    cp=torch.load(str(a.checkpoint),map_location='cuda:0');actor.load_state_dict(cp['model_state_dict']);actor.eval()
    env.reset();env.episode_length_buf.zero_();env.capture_eval=True
    if env.motion_reference is not None:env.motion_reference.reset(torch.arange(env.num_envs,device=env.device))
    phase=config.get('phase',read_json(run/'status.json',{}).get('reward_phase'))
    command_set=[.03,.06,.09] if phase=='guided' else [.025,.04,.06] if phase=='lift' else [.05,.10,.15]
    commands=np.array(command_set)[np.arange(a.num_envs)%3]
    env.commands[:,0]=torch.tensor(commands,device=env.device);env.commands[:,1:]=0
    env.compute_observations();obs=env.get_observations()
    start=env.root_states[:,:3].cpu().numpy().copy();end=start.copy()
    active=np.ones(a.num_envs,dtype=bool);duration=np.zeros(a.num_envs);falls=np.zeros(a.num_envs,dtype=bool);knee_falls=np.zeros(a.num_envs,dtype=bool)
    airtime=np.zeros((a.num_envs,4));lift=np.zeros((a.num_envs,4,3));steps=np.zeros((a.num_envs,4),dtype=int)
    takeoff_time=np.zeros((a.num_envs,4))
    prior=np.ones((a.num_envs,4),dtype=bool);coupling=np.zeros(a.num_envs);trace=[];t0=time.time()
    clear_airtime=np.zeros_like(airtime);peak_clear_airtime=np.zeros_like(airtime);yaw_total=np.zeros(a.num_envs);last_yaw=np.zeros(a.num_envs)
    swing_counts=np.zeros((a.num_envs,4),dtype=int);swing_above10=np.zeros_like(swing_counts);swing_peaks=[[] for _ in range(4)];paired_errors=[[],[]]
    swing_above20=np.zeros_like(swing_counts);air_peak=np.zeros_like(airtime);all20_steps=np.zeros((a.num_envs,4),dtype=int)
    velocity_metrics=VelocityMetrics(a.num_envs,env.dt)
    ankle_ids=[env.dof_names.index(n) for n in ['L_ankle_pitch','R_ankle_pitch']];ankle_history=[]
    stride_sum=np.zeros(4);stride_counts=np.zeros(4,dtype=int)
    path=np.zeros(a.num_envs);last_xyz=start.copy();peak_clearance=np.zeros_like(airtime)
    priorities=PriorityMetrics(env.dof_names,env.body_names,a.num_envs,env.dt,
        [math.degrees(cfg.init_state.default_joint_angles[n]) for n in ['L_hip_pitch','R_hip_pitch']],cfg.posture.hip_stand_tolerance_deg)
    with torch.inference_mode():
        for k in range(math.ceil(a.seconds/env.dt)):
            if a.command_profile=='steps':
                sequence=[0.,.03,.06,.09,.03,0.];index=min(int(k*env.dt/4),len(sequence)-1)
                env.commands[:,0]=sequence[index];env.commands[:,1]=0.;env.commands[:,3]=0.;env.compute_observations();obs=env.get_observations()
            actions=torch.zeros((a.num_envs,env.num_actions),device=env.device) if a.zero else actor.act_inference(obs)
            obs,_,_,done,_=env.step(actions);s=env.eval_snapshot
            if k*env.dt>=1.:
                angles=s['q'][:,ankle_ids].cpu().numpy()*180/math.pi;angles[~active]=np.nan;ankle_history.append(angles)
            velocity_metrics.observe((k+1)*env.dt,active,s['commands'].cpu().numpy(),s['velocity_local'].cpu().numpy(),s['angular_velocity_local'].cpu().numpy())
            stride_valid=s['stride_qualified'].cpu().numpy()&active[:,None]
            stride_counts+=stride_valid.sum(0);stride_sum+=(s['stride_signed_m'].cpu().numpy()*stride_valid).sum(0)
            completed=s['swing_completed'].cpu().numpy()&active[:,None]&((k+1)*env.dt>1.)
            peaks=s['swing_peak'].cpu().numpy();swing_counts+=completed;swing_above10+=completed&(peaks>.010)
            swing_above20+=completed&(peaks>.020)
            paired=s['swing_paired'].cpu().numpy()&active[:,None]&((k+1)*env.dt>1.)
            differences=s['swing_peak_difference'].cpu().numpy()
            for foot in range(4):
                swing_peaks[foot].extend(peaks[completed[:,foot],foot].tolist())
                paired_errors[foot//2].extend(differences[paired[:,foot],foot].tolist())
            xyz=s['xyz'].cpu().numpy();feet=s['feet'].cpu().numpy();contact=s['force'].cpu().numpy()>1.
            taking_off=prior&~contact;lift[taking_off]=feet[taking_off];takeoff_time[taking_off]=(k+1)*env.dt
            clear_airtime[taking_off]=0.;peak_clear_airtime[taking_off]=0.;clearance=s['clearance'].cpu().numpy()
            air_peak[taking_off]=0.;air_peak=np.where(~contact,np.maximum(air_peak,clearance),air_peak)
            clear_airtime=np.where(~contact&(clearance>=.002),clear_airtime+env.dt,0.)
            peak_clear_airtime=np.maximum(peak_clear_airtime,clear_airtime)
            airtime+=~contact*env.dt
            landing=~prior&contact
            # Suppress initial settling and contact chatter: >=60 ms air and >=5 mm foot travel.
            travel=np.linalg.norm(feet-lift,axis=2)
            qualified=landing&(airtime>=.06)&(peak_clear_airtime>=.06)&(travel>=.005)&active[:,None]&(k*env.dt>.5)
            steps+=qualified
            all20_steps+=qualified&(air_peak>.020)&(takeoff_time>1.)
            priorities.observe((k+1)*env.dt,active,s['q'].cpu().numpy(),s['body_xyz'].cpu().numpy(),s['body_force_norm'].cpu().numpy(),clearance,s['force'].cpu().numpy(),qualified&(takeoff_time>1.),travel,s['knee_clearance'].cpu().numpy())
            airtime[contact]=0.;clear_airtime[contact]=0.;peak_clear_airtime[contact]=0.;prior=contact
            peak_clearance[active]=np.maximum(peak_clearance[active],clearance[active])
            quat=s['base_quat'].cpu().numpy();x,y,z,w=quat.T;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            if k:yaw_total[active]+=np.arctan2(np.sin(yaw[active]-last_yaw[active]),np.cos(yaw[active]-last_yaw[active]))
            last_yaw=yaw;path[active]+=np.linalg.norm(xyz[active,:2]-last_xyz[active,:2],axis=1);last_xyz=xyz.copy()
            end[active]=xyz[active];duration[active]=(k+1)*env.dt
            coupling[active]=np.maximum(coupling[active],s['coupling'].cpu().numpy()[active])
            falls[active]|=s['fall'].cpu().numpy()[active]
            knee_falls[active]|=s['knee_ground_contact'].any(1).cpu().numpy()[active]
            if active[0] and k%5==0:trace.append({'t':(k+1)*env.dt,'body_xyz':s['body_xyz'][0].cpu().tolist(),'force':s['force'][0].cpu().tolist(),'q':s['q'][0].cpu().tolist(),'clearance_m':clearance[0].tolist(),'knee_clearance_m':s['knee_clearance'][0].cpu().tolist(),'command_m_s':s['commands'][0,:3].cpu().tolist(),'actual_velocity_m_s':s['velocity_local'][0].cpu().tolist()})
            active&=~done.cpu().numpy().astype(bool)
            if not active.any():break
    dx=end[:,0]-start[:,0];survive5=~(falls&(duration<=5.+1e-6));survive16=~falls
    summary={'knee_ground_failure_fraction':float(knee_falls.mean()),'survival_5s':float(survive5.mean()),'survival_16s':float(survive16.mean()),
        'median_displacement_m':float(np.median(dx)),'mean_displacement_m':float(dx.mean()),
        'mean_duration_s':float(duration.mean()),'mean_min_foot_steps':float(steps.min(1).mean()),
        'max_coupling_error_rad':float(coupling.max()),
        'moving_all_four_fraction':float(((steps.min(1)>=2)&(dx>.15)&survive16).mean())}
    summary.update(median_abs_heading_change_deg=float(np.median(np.abs(yaw_total))*180/math.pi),median_path_m=float(np.median(path)),mean_peak_clearance_m=peak_clearance.mean(0).tolist())
    summary['score']=summary['median_displacement_m']+.3*summary['survival_16s']+.02*float(np.minimum(steps.min(1),5).mean())
    result={'schema':'sable-independent-evaluation/1','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
        'iteration':int(cp['iter']),'runtime_sha256':cp['infos']['runtime_sha256'],'evaluator_sha256':sha256(__file__),'diagnostics_sha256':sha256(Path(__file__).with_name('priority_metrics.py')),
        'created_unix':time.time(),'wall_seconds':time.time()-t0,'horizon_s':a.seconds,'num_envs':a.num_envs,
        'seed':947,'zero_actions':a.zero,'command_set_m_s':command_set if a.command_profile=='constant' else [0.,.03,.06,.09],'command_profile':a.command_profile,'summary':summary,'accepted_locomotion':False,
        'kinematic_reference_enabled':bool(cfg.reference.enabled),'zero_policy_residual':bool(a.zero and cfg.reference.enabled),
        'mirror_policy':bool(config.get('mirror_policy',False)),
        'reference_strength':cfg.reference.strength if cfg.reference.enabled else 0.,
        'step_definition':'capsule clearance >=2mm and contact-off >=60ms then landing, foot travel >=5mm; first 0.5s excluded',
        'step_definition_zh':'足底离地至少2mm并持续60ms，重新着地时足中心移动至少5mm；最初0.5秒不计。',
        'domain':'flat floor, explicit material_friction settings recorded below, deterministic policy, 0.015rad initial joint perturbation; simulation only',
        'swing_peak_diagnostics':{'definition':'Actual capsule-bottom peak within each complete planned swing after startup; no 60ms duration condition in this peak metric; no-lift swings are counted.',
            'completed_swings_per_foot':swing_counts.sum(0).tolist(),'swings_peak_above10mm_per_foot':swing_above10.sum(0).tolist(),
            'fraction_above10mm_per_foot':(swing_above10.sum(0)/np.maximum(swing_counts.sum(0),1)).tolist(),
            'median_peak_m_per_foot':[float(np.median(v)) if v else None for v in swing_peaks],
            'paired_peak_rmse_m_front_hind':[float(np.sqrt(np.mean(np.square(v)))) if v else None for v in paired_errors],
            'samples':[{'counts':swing_counts[i].tolist(),'above10mm':swing_above10[i].tolist()} for i in range(a.num_envs)]},
        'samples':[{'command_m_s':float(commands[i]) if a.command_profile=='constant' else None,'duration_s':float(duration[i]),'fell':bool(falls[i]),'knee_ground_failure':bool(knee_falls[i]),'displacement_m':float(dx[i]),'steps_per_foot':steps[i].tolist()} for i in range(a.num_envs)],
        'priority_diagnostics':priorities.result(),
        'knee_contact_rule':{'enabled':cfg.termination.knee_contact_enabled,'margin_m':cfg.termination.knee_contact_margin_m,'method':'Existing knee motor collision primitives vs plane z=0, checked each 20ms control step; cylinder support bound conservative for tessellation'},
        'asset_sha256':sha256(cfg.asset.file),'manifest_sha256':sha256(Path(cfg.asset.file).with_name('physical_manifest.json')),
        'body_names':env.body_names,'dof_names':env.dof_names,'foot_order':[env.body_names[i] for i in env.feet_indices.cpu().tolist()],'trace':trace}
    result['material_friction']={'plane':cfg.material_friction.plane,'front_scale':cfg.material_friction.front_scale,'nominal_front_floor_mu':cfg.material_friction.front_scale,'nominal_rear_floor_mu':1.,'combination':'arithmetic mean, validated by isolated sliding deceleration','scope':'Temporary simulation materials; not hardware friction. Both batch arms have matched material remapping; non-foot/foot self-contact friction differs from parent.'}
    hind20_steps=all20_steps[:,2:]
    result['hind20_diagnostics']={'definition':'Each qualifying hind step retains2mm/60ms/5mm and additionally has actual capsule-bottom peak>20mm during the same air phase; takeoff after1s. No requirement to hold20mm for60ms.',
        'steps_per_environment':hind20_steps.tolist(),'total_steps_left_right':hind20_steps.sum(0).tolist(),
        'both_hind_2steps_forward_survive_fraction':float(((hind20_steps.min(1)>=2)&(steps.min(1)>=2)&(dx>.15)&survive16).mean()),
        'completed_swings_peak_above20mm_per_foot':swing_above20.sum(0).tolist(),
        'fraction_completed_swings_peak_above20mm_per_foot':(swing_above20.sum(0)/np.maximum(swing_counts.sum(0),1)).tolist()}
    result['all20_diagnostics']={'definition':'All four feet: valid landed step with actual capsule-bottom peak>20mm in that air phase, clearance>=2mm for>=60ms and foot travel>=5mm; takeoff after1s.',
        'steps_per_environment':all20_steps.tolist(),'total_steps_per_foot':all20_steps.sum(0).tolist(),
        'four_feet_2steps_forward_survive_fraction':float(((all20_steps.min(1)>=2)&(dx>.15)&survive16).mean()),
        'completed_swing_peak_above20mm_fraction_per_foot':(swing_above20.sum(0)/np.maximum(swing_counts.sum(0),1)).tolist()}
    result['stride_progress_diagnostics']={'definition':'Forward displacement along takeoff heading for valid landed steps:20mm peak,2mm clearance60ms,atleast5mm forward placement,body forwardvelocity>5mm/s; startup excluded.','counts_per_foot':stride_counts.tolist(),'mean_forward_stride_m_per_foot':(stride_sum/np.maximum(stride_counts,1)).tolist()}
    if ankle_history:
        ah=np.asarray(ankle_history);median=np.nanmedian(ah,axis=0);rom=np.nanpercentile(ah,95,axis=0)-np.nanpercentile(ah,5,axis=0)
        result['ankle_diagnostics']={'joint_order':['L_ankle_pitch','R_ankle_pitch'],'definition':'All live samples after1s; per-environment median and p95-p05 range, then median over environments','median_angle_deg':np.nanmedian(median,axis=0).tolist(),'median_rom_p95_p05_deg':np.nanmedian(rom,axis=0).tolist(),'per_environment_median_deg':median.tolist(),'per_environment_rom_deg':rom.tolist()}
    result['hip_preferred_max_deg']=getattr(cfg.posture,'hip_preferred_max_deg',None)
    result['velocity_tracking']=velocity_metrics.result()
    result['velocity_tracking']['command_profile']=a.command_profile
    result['velocity_tracking']['step_sequence_m_s']=[0.,.03,.06,.09,.03,0.] if a.command_profile=='steps' else None
    result['velocity_tracking']['step_duration_s']=4. if a.command_profile=='steps' else None
    result['friction_power_diagnostic']={'status':'unavailable_calibration_failed','included_in_reward':False,
        'definition':'Horizontal ground force on foot dotted with actual body horizontal velocity; positive along body motion.',
        'reason':'Legacy net-contact and articulation sensor horizontal-force measurements failed known-load calibration. No per-foot power numbers published.'}
    result['reference_per_foot']={'horizontal_table_scale_per_foot':env.motion_reference.data.get('horizontal_table_scale_per_foot') if env.motion_reference else None,'height_m':cfg.gait.swing_heights_m,'amplitude_multipliers':cfg.reference.amplitude_multipliers,
        'target_m':[h*cfg.reference.strength*m for h,m in zip(cfg.gait.swing_heights_m,cfg.reference.amplitude_multipliers)]}
    result['replay_source_files_sha256']={str(p.relative_to(Path(__file__).parent)):sha256(p) for p in list(Path(__file__).parent.glob('*.py'))+list((Path(__file__).parent/'assets').rglob('*')) if p.is_file()}
    result['runtime_identity_scope']='runtime_sha256 is the checkpoint training runtime; replay_source_files_sha256 identifies actual diagnostic replay sources and assets.'
    atomic_json(a.output,result);print('EVALUATION',json.dumps(summary),flush=True);env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
