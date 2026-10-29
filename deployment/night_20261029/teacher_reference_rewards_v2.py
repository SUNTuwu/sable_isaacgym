"""Exclusive reference-local rewards. No v44 or strict_steps reward is called.

Continuous terms are per-second; event costs are divided by dt here and the
trainer integrates the sum exactly once. All motion commands, including pure
yaw and slow translation, enable contact-event costs.
"""
import math
import torch
from teacher_turn_cycles import signed_tracking,TurnCycles

PROFILE='reference_local_v2'
WEIGHTS={'velocity':4.,'yaw':4.,'foot_local':4.,'cross_motion':-.2,'heading':0.,'translation_yaw_rate':-2.,
         'contact_timing':-2.,'slip':-1.5,'gravity_alignment':-2.,'body_height':-.5,'target_acceleration':-.3,
         'joint_acceleration':-.05,'torque':-.1,'joint_limits':-.5,'symmetry':-.5}
METADATA={
 'task_boundary_settlement':'Only natural physical death or full episode timeout; signed integral(yaw_dot), norm integral(world XY velocity); translations30deg, turns0.20m. No early task-boundary termination.',
 'turn_command':'fixed +/-30deg/s','profile':PROFILE,'weights':WEIGHTS,'reference_amplitude':1.,'control':'q_ref + bounded offset',
 'foot_frame':'fore: shoulder midpoint/chest; hind: hip midpoint/calibrated pelvis; no global XY/yaw imitation',
 'gravity_alignment_definition':'-2*(gravity_angle_error/10deg)^2; exact match zero, no positive baseline; root/chest only',
 'foot_local_definition':'4*mean(exp(-sum((error/[0.040,0.040,0.020])^2))); loose body-local guide',
 'cross_motion_definition':'turns: -2*min((horizontal_speed/0.03)^2,4), cap -8/s; translations unchanged: -0.2*u^2/(1+u^2/4), u=cross_speed/0.06, cap -0.8/s',
 'contact_timing_definition':'-2*mean(actual_contact != (reference_contact_probability>=0.5)); every control step, all four feet; no timing deadband',
 'translation_yaw_rate_definition':'translations only: -2*min((yaw_rate_error/0.10rad_s)^2,4), cap -8/s; pure turns excluded',
 'superseded_turn_boundary_curriculum':'turns only; reset-relative rootXY radius 0.30m -> 0.20m once >=19/20 complete held-out episodes survive without crossing. Translations: reset-relative yaw 30deg -> 20deg -> 10deg, same independent 19/20 full-episode gate; replaces continuous heading-integral penalty. Other previous-version rewards preserved.',
 'turn_task_revision':{'yaw':'6*tanh(1-abs(w-w_cmd)/abs(w_cmd))/tanh(1): rest 0, goal +6/s, reverse negative, overspeed discouraged','foot_weight':2,'foot_sigma_xyz_m':[.060,.060,.025],'period':'unchanged reference phase/contact timing; full-cycle cadence count checked','cycle_event':'up to +2 per completed cycle, drift/phase gated; reverse negative, extra/missing lifts -0.1 each; partial reset cycles excluded','other_six_teachers':'unchanged'},
 'fall_definition':'per fall -(20 + sum of other same-transition negative costs); timeout excluded; displayed rates are not per-event amounts',
 'event_costs':{'liftoff_per_foot':.01,'additional_liftoff_same_reference_cycle':.04,
                'short_swing_max_per_foot':.05,'short_swing_threshold':'0.5 * reference swing duration'},
 'contact_detector':'Force hysteresis: contact >=1N; leave <=0.3N AND capsule clearance >1mm; no long debounce; reset air is not a liftoff',
 'symmetry':'Straight walk/trot only; actual anatomy-local XY compared to mirrored opposite foot half a reference period earlier',
 'aggregation':'sum of continuous rates * dt + event costs; no positive-only clipping',
 'removed':'All v44 reward modules, strict step width/height/distance/angle targets, raw upstroke taxes, cadence budgets, inverse duration, extra alive rewards, joint pose/velocity imitation, hard-coded waist=-40deg termination',
 'termination':'Nonfinite, severe tilt/low body, prohibited body/knee contact, passive coupling; time limit. No cadence termination.'}

def quat_apply(q,v):
    xyz=q[...,:3];t=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,3:4]*t+torch.cross(xyz,t,dim=-1)

def quat_inverse_apply(q,v):
    iq=q.clone();iq[...,:3]*=-1
    return quat_apply(iq,v)

def bounded_exp_square(x,cap=4.):
    return torch.expm1(x.square().clamp(max=math.log1p(cap)))

def validate_config(spec):
    for name in ('motion_reward_module_enabled','adaptive_enabled','translation_guard_enabled','pace_stability_enabled','balance_first'):
        if spec.get(name,False):raise ValueError('reference_local_v2 forbids legacy option: '+name)
    for name in ('step_quality_profile','task_tracking_profile'):
        if spec.get(name,'legacy')!='legacy':raise ValueError('reference_local_v2 forbids legacy option: '+name)
    if spec.get('control_mode')!='reference_plus_residual':raise ValueError('v2 needs reference_plus_residual')
    for name in ('reference_strength','time_scale','turn_speed_scale','turn_reference_time_scale'):
        if spec.get(name,1.)!=1.:raise ValueError('v2 preserves reference and command scale: '+name)
    if spec.get('turn_offset_rate_deg_s',0.)!=0.:raise ValueError('Use the full target smoothness cost, not turn offset rate clamping')

class ReferenceRewards:
    def __init__(self,n,device,dt,period,swing_time,symmetric):
        self.dt=dt;self.period=period;self.swing_time=swing_time;self.symmetric=symmetric
        self.contact=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.ready=torch.zeros(n,device=device,dtype=torch.bool);self.armed=self.contact.clone()
        self.air=torch.zeros(n,4,device=device);self.cycle_count=torch.zeros_like(self.air)
        self.previous_phase=torch.zeros(n,device=device);self.cycle_ready=self.ready.clone()
        self.yaw_integral=torch.zeros(n,device=device);self.age=torch.zeros(n,device=device)
        self.last_target=torch.zeros(n,15,device=device);self.last_target_velocity=torch.zeros_like(self.last_target)
        self.history_steps=int(torch.ceil(period.max()/(2*dt)).item())+2
        self.history=torch.zeros(self.history_steps,n,4,3,device=device);self.index=0
        self.delay=(period/(2*dt)).round().long().clamp(min=1)
        self.sample_count=torch.zeros(n,device=device,dtype=torch.long)
        self.diagnostics={}

    def reset(self,ids):
        for name in ('contact','ready','armed','air','cycle_count','previous_phase','cycle_ready','yaw_integral',
                     'age','last_target','last_target_velocity','sample_count'):
            getattr(self,name)[ids]=0
        self.history[:,ids]=0

    def update(self,s):
        dt=self.dt;n=len(self.ready);force=s['normal_force'];clear=s['clearance']
        previous=self.contact.clone()
        contact=torch.where(previous,~((force<=.3)&(clear>.001)),force>=1.)
        contact=torch.where(self.ready[:,None],contact,force>=1.)
        lift=previous&~contact&self.ready[:,None]
        land=~previous&contact&self.armed&self.ready[:,None]
        self.air+=dt*((~previous)&self.armed)
        duration=self.air.clone();self.armed|=lift
        self.air=torch.where(lift|contact,torch.zeros_like(self.air),self.air)
        wrapped=(s['phase']<self.previous_phase)&self.ready
        self.cycle_count[wrapped]=0;self.cycle_ready|=wrapped
        extra=lift&(self.cycle_count>=1)&self.cycle_ready[:,None]
        self.cycle_count+=lift.float();self.armed&=~land
        moving=(s['command'][:,:2].norm(dim=1)>.001)|(s['command'][:,2].abs()>.001)
        short=(1-duration/(.5*self.swing_time).clamp(min=2*dt)).clamp(min=0,max=1).square()*land
        event=(-.01*lift.sum(1)-.04*extra.sum(1)-.05*short.sum(1))*moving
        actual=s['velocity'];cmd=s['command'];turn=cmd[:,2].abs()>.001
        self.yaw_integral+=actual[:,2]*dt*(~turn)
        speed=cmd[:,:2].norm(dim=1);direction=cmd[:,:2]/speed[:,None].clamp(min=1e-6)
        cross=actual[:,0]*direction[:,1]-actual[:,1]*direction[:,0]
        unwanted=torch.where(turn,actual[:,:2].norm(dim=1),cross)
        error=s['foot_local']-s['reference_local']
        relaxed_sigma=s['foot_sigma']*torch.where(turn[:,None],torch.tensor([1.5,1.5,1.25],device=force.device),torch.ones(3,device=force.device))
        foot_scaled=error/relaxed_sigma[:,None,:]
        velocity_error=(actual[:,:2]-cmd[:,:2]).norm(dim=1)
        yaw_error=actual[:,2]-cmd[:,2]
        target_velocity=(s['target']-self.last_target)/dt
        target_acc=(target_velocity-self.last_target_velocity)/dt
        valid=(self.sample_count>=2).float()
        delayed=self.history[(self.index-self.delay)%self.history_steps,torch.arange(n,device=force.device)]
        mirrored=delayed[:,[1,0,3,2]].clone();mirrored[:,:,1]*=-1
        symmetry_error=(s['foot_local'][:,:,:2]-mirrored[:,:,:2])/.010
        symmetry_valid=self.symmetric&(self.sample_count>=self.delay)
        desired=s['contact_probability']>=.5;phase_mismatch=(contact!=desired).float().mean(1)
        # Contact material-point velocity: allow capsule rolling; do not penalize the toe merely rotating.
        slip=(s['contact_velocity'][:,:,:2].norm(dim=-1)/.05).square().clamp(max=4)
        posture=((s['gravity']-s['reference_gravity']).norm(dim=1)/.35).square().clamp(max=4)
        height=(s['height_error']/.03).square().clamp(max=4)
        limit=(s['limit_excess']/.05).square().clamp(max=4).mean(1)
        gravity_angle=torch.atan2(torch.cross(s['gravity'],s['reference_gravity'],dim=-1).norm(dim=1),(s['gravity']*s['reference_gravity']).sum(1))
        terms={
          'v2_velocity':4*torch.exp(-(velocity_error/s['velocity_sigma']).square()),
          'v2_yaw':torch.where(turn,signed_tracking(actual[:,2],cmd[:,2]),4*torch.exp(-(yaw_error/s['yaw_sigma']).square())),
          'v2_foot_local':torch.where(turn,torch.full_like(yaw_error,2.),torch.full_like(yaw_error,4.))*torch.exp(-foot_scaled.square().sum(-1)).mean(1),
          'v2_cross_motion':torch.where(turn,-2*(unwanted/.03).square().clamp(max=4),-.2*((unwanted/.06).square()/(1+(unwanted/.06).square()/4))),
          'v2_translation_yaw_rate':-2*(yaw_error/.10).square().clamp(max=4)*(~turn),
          'v2_heading_integral':torch.zeros_like(self.yaw_integral),
          'v2_contact_timing':-2*phase_mismatch,
          'v2_contact_slip':-1.5*(slip*contact).mean(1),
          'v2_gravity_alignment':-2*(gravity_angle/math.radians(10)).square(),
          'v2_body_height':-.5*height,
          'v2_target_acceleration':-.3*(target_acc/100.).square().clamp(max=4).mean(1)*valid,
          'v2_joint_acceleration':-.05*(s['joint_acceleration']/150.).square().clamp(max=4).mean(1)*valid,
          'v2_torque':-.1*s['normalized_torque'].square().clamp(max=4).mean(1),
          'v2_joint_limits':-.5*limit,
          'v2_phase_symmetry':-.5*bounded_exp_square(symmetry_error).mean((1,2))*symmetry_valid,
          'v2_liftoff_events':event/dt,
          'v2_task_boundary_event':-10*s.get('task_failure',torch.zeros_like(s['fall'])).float()/dt,
          'v2_fall_event':torch.zeros_like(gravity_angle),
        }
        other_cost=sum((-value).clamp(min=0) for value in terms.values())*dt
        terms['v2_fall_event']=-(20.+other_cost)*s['fall'].float()/dt
        self.diagnostics={'contact_phase_agreement':1-phase_mismatch,'reference_contact':desired.clone(),'gravity_error_rad':gravity_angle,'fall':s['fall'].clone(),'fall_event_cost':(20.+other_cost)*s['fall'].float(),'liftoff':lift.clone(),'touchdown':land.clone(),'extra_liftoff':extra.clone(),
            'short_swing':(short>0).clone(),'swing_duration_s':duration,'foot_error_m':error.norm(dim=-1),
            'contact':contact.clone(),'heading_error_rad':self.yaw_integral.clone(),
            'foot_saturated_fraction':(torch.exp(-foot_scaled.square().sum(-1))>.95).float().mean(1),
            'velocity_error_m_s':velocity_error,'yaw_error_rad_s':yaw_error.abs(),
            'contact_slip_squared':(s['contact_velocity'][:,:,:2].square().sum(-1)*contact).sum(1)}
        self.contact=contact;self.ready[:]=True;self.previous_phase=s['phase'].clone()
        self.last_target=s['target'].clone();self.last_target_velocity=target_velocity.clone()
        self.history[self.index]=s['foot_local'];self.index=(self.index+1)%self.history_steps
        self.sample_count+=1;self.age+=dt
        return terms

def initialize_env(e):
    import numpy as np
    from pathlib import Path
    spec=e.cfg.teacher_bank;data=[np.load(Path(spec['reference'])/(c['id']+'.npz')) for c in spec['clips']]
    for key,name in [('anatomy_toe','toe_local_anatomy'),('contact_probability','reference_contact_probability')]:
        e.ref[key]=torch.tensor(np.stack([d[name] for d in data]),device=e.device,dtype=torch.float32)
    e.teacher_commands[:]=torch.tensor([c['command'] for c in spec['clips']],device=e.device)
    e.raw_reference_commands.copy_(e.teacher_commands)
    from teacher_turn_curriculum import TurnCurriculum
    e.turn_curriculum=TurnCurriculum([c["id"] for c in spec["clips"]],e.ref_duration.tolist())
    e.turn_curriculum.apply(e)
    e.anatomy_calibration=torch.tensor(data[0]['anatomy_frame_calibration'],device=e.device,dtype=torch.float32)
    swing=torch.tensor(np.stack([d['reference_swing_time_s'] for d in data]),device=e.device,dtype=torch.float32)
    symmetric=torch.tensor([c.get('symmetry_enabled',False) for c in spec['clips']],device=e.device)
    e.reference_rewards_v2=ReferenceRewards(e.num_envs,e.device,e.dt,e.ref_duration[e.teacher_ids],swing[e.teacher_ids],symmetric[e.teacher_ids])
    properties=e.gym.get_actor_rigid_body_properties(e.envs[0],e.actor_handles[0])
    e.v2_foot_com=torch.tensor([[properties[i].com.x,properties[i].com.y,properties[i].com.z] for i in e.feet_indices.tolist()],device=e.device)
    e.v2_sigma=torch.tensor([[.040,.040,.020]]*e.teacher_count,device=e.device)
    e.v2_progress=torch.zeros(e.teacher_count,device=e.device)
    e.v2_failures=torch.zeros(e.teacher_count,20,device=e.device)
    e.v2_exposure=torch.zeros_like(e.v2_failures)
    from teacher_turn_boundary import TurnBoundary
    e.turn_boundary=TurnBoundary([c['id'] for c in spec['clips']],e.device)
    contacts=e.ref['contact_probability'][:,:-1]>=.5
    expected=(torch.roll(contacts,1,1)&~contacts).sum(1).float()
    e.turn_cycles=TurnCycles(e.num_envs,e.device,e.ref_duration[e.teacher_ids],expected[e.teacher_ids])

def sample_phase(e,ids):
    risk=((e.v2_failures+1)/(e.v2_exposure+20)).sqrt()
    probabilities=.5/20+.5*risk/risk.sum(1,keepdim=True)
    bins=torch.multinomial(probabilities[e.teacher_ids[ids]],1).squeeze(1)
    return (bins.float()+torch.rand(len(ids),device=e.device))/20

def check_env(e):
    """Replace, rather than append to, the full legacy termination/reward path."""
    ref=e.ref_sample();n=e.num_envs;device=e.device
    tilt=(e.projected_gravity*quat_inverse_apply(ref['quat'],e.gravity_vec)).sum(1)
    body_contact=(e.contact_forces[:,e.termination_contact_indices].norm(dim=-1)>2.).any(1)
    coupling=(e.dof_pos[:,e.hip]+e.dof_pos[:,e.dependent]).abs()>.08
    finite=torch.isfinite(e.root_states).all(1)&torch.isfinite(e.dof_pos).all(1)&torch.isfinite(e.dof_vel).all(1)
    low=(e.root_states[:,2]-e.env_origins[:,2])<torch.maximum(torch.full((n,),.035,device=device),ref['root'][:,2,3]*.45)
    knee=e.knee_ground_contact.any(1)&bool(e.cfg.termination.knee_contact_enabled)
    physical_fall=(tilt<.5)|body_contact|coupling|~finite|low|knee
    turn_failure=e.turn_boundary.evaluate(e) if hasattr(e,'turn_boundary') else torch.zeros_like(physical_fall)
    episode_end=physical_fall|(e.episode_length_buf>=e.max_episode_length)
    task_failure=turn_failure&episode_end
    e.true_fall=physical_fall
    e.time_out_buf=(e.episode_length_buf>=e.max_episode_length)&~physical_fall
    e.reset_buf=e.true_fall|e.time_out_buf;e.extras['time_outs']=e.time_out_buf
    e.terminal_proprio=e._proprio().clone();e.terminal_velocity=e.base_lin_vel.clone()
    e.terminal_critic=e._critic(e.terminal_proprio).clone()
    e.extras.update(terminal_critic=e.terminal_critic,next_clean_proprio=e.terminal_proprio,next_velocity=e.terminal_velocity)
    # Nonfinite state must stop the run instead of being silently hidden by reward clipping.
    if not bool(finite.all()):raise FloatingPointError('Nonfinite physical state in reference_v2')
    tips=[]
    for _,points,radius in e.mapping.capsules:
        distal=int(points.square().sum(-1).argmax());axis=points[distal]-points[1-distal]
        tips.append(points[distal]+radius*axis/axis.norm())
    feet=e.rigid_states[:,e.feet_indices];tip=torch.stack(tips)
    toe=feet[:,:,:3]+quat_apply(feet[:,:,3:7],tip[None].expand(n,-1,-1))
    anchors=e.rigid_states[:,e.event_anchor_indices,:3].mean(2).repeat_interleave(2,dim=1)
    rotations=e.rigid_states[:,e.event_frame_indices,3:7].repeat_interleave(2,dim=1)
    local=quat_inverse_apply(rotations,toe-anchors)
    calibration=e.anatomy_calibration.repeat_interleave(2,dim=0)
    local=torch.einsum('nfi,fij->nfj',local,calibration)
    raw_ref=torch.einsum('nfi,fji->nfj',ref['anatomy_toe'],calibration)
    display_ref=anchors+quat_apply(rotations,raw_ref)
    e.reference_endpoint_error_vector_m=local-ref['anatomy_toe'];e.reference_endpoint_error_m=e.reference_endpoint_error_vector_m.norm(dim=-1)
    com=feet[:,:,:3]+quat_apply(feet[:,:,3:7],e.v2_foot_com[None].expand(n,-1,-1))
    material_velocity=feet[:,:,7:10]+torch.cross(feet[:,:,10:13],e.foot_contact_point-com,dim=-1)
    q=e.dof_pos[:,e.active];limits=e.hard_limits[e.active];target=e.executed_target
    near=(limits[:,0]+.05-q).clamp(min=0)+(q-(limits[:,1]-.05)).clamp(min=0)
    raw=e.current_reference_target+e.current_offset_rad
    near+=(limits[:,0]-raw).clamp(min=0)+(raw-limits[:,1]).clamp(min=0)
    command=e.teacher_commands[e.teacher_ids];actual=torch.cat((e.heading_linear_velocity[:,:2],e.heading_yaw_rate[:,None]),1)
    state={'normal_force':e.contact_forces[:,e.feet_indices,2].clamp(min=0),'clearance':e.foot_clearance,
        'phase':e.ref_phase,'command':command,'velocity':actual,'foot_local':local,'reference_local':ref['anatomy_toe'],
        'foot_sigma':e.v2_sigma[e.teacher_ids],'velocity_sigma':torch.maximum(command[:,:2].norm(dim=1)*.4,torch.full((n,),.020,device=device)),
        'yaw_sigma':torch.maximum(command[:,2].abs()*.4,torch.where(command[:,2].abs()>.001,torch.full((n,),.03,device=device),torch.full((n,),.10,device=device))),
        'target':target,'contact_probability':ref['contact_probability'],'contact_velocity':material_velocity,
        'gravity':e.projected_gravity,'reference_gravity':quat_inverse_apply(ref['quat'],e.gravity_vec),
        'height_error':e.root_states[:,2]-e.env_origins[:,2]-ref['root'][:,2,3],
        'joint_acceleration':(e.dof_vel[:,e.active]-e.last_dof_vel[:,e.active])/e.dt,
        'normalized_torque':e.torques[:,e.active]/e.torque_limits[e.active].clamp(min=1e-3),
        'limit_excess':near,'fall':physical_fall,'task_failure':task_failure}
    terms=e.reference_rewards_v2.update(state)
    if hasattr(e,'turn_cycles'):
        from teacher_translation_guard import root_heading
        diag=e.reference_rewards_v2.diagnostics
        cycle_rate=e.turn_cycles.update(root_heading(e.root_states[:,3:7]),e.root_states[:,:2],e.ref_phase,command,diag['contact_phase_agreement'],diag['liftoff'],e.true_fall,e.dt)
        terms['v2_turn_cycle_task']=cycle_rate
        diag['turn_cycles']=e.turn_cycles.diagnostics
        # Preserve fall dominance after adding the turn-only cycle event.
        terms['v2_fall_event']=terms['v2_fall_event']-(-cycle_rate).clamp(min=0)*physical_fall
    import night_tuning
    terms=night_tuning.apply(e,terms,physical_fall)
    if e.training_env_count:
        count=e.training_env_count;index=e.teacher_ids[:count]*20+(e.ref_phase[:count]*20).long().clamp(max=19)
        e.v2_exposure.mul_(.999);e.v2_failures.mul_(.999)
        e.v2_exposure.view(-1).index_add_(0,index,torch.ones(count,device=device))
        e.v2_failures.view(-1).index_add_(0,index,e.true_fall[:count].float())
    e.teacher_terms=terms;e.teacher_reward=sum(terms.values())*e.dt
    if not bool(torch.isfinite(e.teacher_reward).all()):raise FloatingPointError('Nonfinite reference_v2 reward')
    e.pre_reset_teacher_obs=e.teacher_observations().clone()
    e.pre_reset_teacher_state={'q_rmse':(q-ref['q']).square().mean(1).sqrt(),'xy_error':(actual[:,:2]-command[:,:2]).norm(dim=1),
        'yaw_error':(actual[:,2]-command[:,2]).abs(),'actual_velocity':actual.clone(),'q':q.clone(),'root':e.root_states.clone(),
        'toe':toe.clone(),'reference_toe':display_ref.clone(),'contact_force':state['normal_force'].clone(),
        'contact':e.reference_rewards_v2.contact.clone(),'phase':e.ref_phase.clone(),'fall':e.true_fall.clone(),
        'executed_target_full':target.clone(),'residual_action':e.current_residual_action.clone(),
        'reference_joint_target':e.current_reference_target.clone(),'reference_endpoint_error_vector_m':e.reference_endpoint_error_vector_m.clone(),
        'offset_joint_rad':e.current_offset_rad.clone(),
        'fall_diagnostics':{'tilt_failure':tilt<.5,'height_failure':low,'termination_contact_failure':body_contact,
                            'coupling_failure':coupling,'knee_ground_failure':knee,'task_boundary_failure':task_failure,'integrated_yaw_deg':torch.rad2deg(e.turn_boundary.angle).clone() if hasattr(e,'turn_boundary') else torch.zeros(n,device=device),'integrated_xy_m':e.turn_boundary.displacement.norm(dim=1).clone() if hasattr(e,'turn_boundary') else torch.zeros(n,device=device)},
        'v2_diagnostics':e.reference_rewards_v2.diagnostics}
