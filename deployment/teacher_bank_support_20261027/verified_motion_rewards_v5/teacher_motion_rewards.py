"""Motion-specific costs and direction-adapted v44 walk reward mechanisms.

All returned values are WEIGHTED PER-SECOND terms: integrate once with env.dt.
No positive-only clipping is applied. This module never writes motor targets.
Reference heights provide soft targets; actual contacts define event timing.
"""
import math
import torch
from ankle_events import AnkleStepEvents
from stride_progress import StrideProgress
from hind_landing_width import HindLandingWidth
from waist_acceleration import WaistHeightAcceleration

V44_WEIGHTS={'action_rate':-.01,'ang_vel_xy':-.05,'ankle_event_angle':-2.,'base_height':-10.,
 'collision':-1.,'dof_acc':-2.5e-7,'dof_pos_limits':-2.,'feet_air_time':1.,'forward_speed':2.,
 'gait_clearance':-32.,'gait_contact':3.,'hind_landing_width':-4.,'lin_vel_z':-2.,'orientation':-.5,
 'proximal_acceleration_peak':-4.,'stride_progress':2.,'stride_shortfall':-2.,'stride_variance':-2.,
 'swing_peak_balance':-.5,'swing_peak_shortfall':-2.,'target_limits':-2.,'termination':-250.,
 'torques':-1e-5,'tracking_ang_vel':1.,'tracking_lin_vel':8.,'waist_vertical_acceleration':-1.,'yaw_rate_error':-2.}


def quat_apply(q,v):
    xyz=q[...,:3];twice=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,3:4]*twice+torch.cross(xyz,twice,dim=-1)


def body_forward_xy(quat):
    x,y,z,w=quat.unbind(-1)
    forward=torch.stack((1-2*(y*y+z*z),2*(x*y+w*z)),-1)
    return forward/forward.norm(dim=-1,keepdim=True).clamp(min=1e-6)


def directed_ground_angle_deg(axis_world,forward_xy):
    """0..180 angle of distal foot long-axis to body-forward horizontal ray.

    Keep the full 3D axis normalization. Normalizing only its XY projection would
    destroy pitch information; abs(dot) would incorrectly merge30 and150deg.
    """
    axis=axis_world/axis_world.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    return torch.acos((axis[...,:2]*forward_xy[:,None,:]).sum(-1).clamp(-1,1))*180/math.pi


def canonical_task_commands(commands,clip_ids):
    out=commands.clone()
    for i,ident in enumerate(clip_ids):
        if 'TURN' in ident:out[i,:2]=0.
        elif 'LATERAL' in ident or 'STRAFE' in ident:out[i,0]=0.;out[i,2]=0.
        else:out[i,1:]=0.
    return out


def bounded_square(normalized_error):
    """Smooth nonnegative z/(1+z), z=error^2; stable even at large error."""
    z=normalized_error.square()
    return 1.-1./(1.+z)


def direction_terms(actual_xy,actual_yaw,commands,turn_mask,sigma_velocity=None,sigma_angular=None,velocity_deadband=0.):
    desired=commands[:,:2];speed=desired.norm(dim=-1);translation=~turn_mask
    axis=desired/speed[:,None].clamp(min=1e-6)
    along=(actual_xy*axis).sum(-1);cross=actual_xy[:,0]*axis[:,1]-actual_xy[:,1]*axis[:,0]
    sigma_v=torch.maximum(speed*.5,torch.full_like(speed,.02)) if sigma_velocity is None else sigma_velocity
    yaw_goal=torch.where(turn_mask,commands[:,2],torch.zeros_like(actual_yaw))
    sigma_yaw=torch.maximum(yaw_goal.abs()*.5,torch.full_like(speed,.10)) if sigma_angular is None else sigma_angular
    velocity_error=((actual_xy-desired).norm(dim=-1)-velocity_deadband).clamp(min=0)
    return {'task_velocity_cost':-2*bounded_square(velocity_error/sigma_v),
      'task_yaw_cost':-2*bounded_square((actual_yaw-yaw_goal)/sigma_yaw),
      'task_wrong_translation_cost':-2*bounded_square((-along).clamp(min=0)/.03)*translation,
      'task_cross_axis_cost':-bounded_square(cross/.03)*translation,
      'task_wrong_turn_cost':-2*bounded_square((-actual_yaw*torch.sign(yaw_goal)).clamp(min=0)/sigma_yaw)*turn_mask,
      'task_turn_translation_cost':-bounded_square(actual_xy.norm(dim=-1)/.03)*turn_mask}


class TeacherMotionRewards:
    def __init__(self,env):
        self.env=env;n=env.num_envs;device=env.device;spec=env.cfg.teacher_bank
        self.ids=[c['id'] for c in spec['clips']]
        types=env.teacher_ids
        walk_teacher=torch.tensor([x in ('01_FORWARD_POLICY','09_BACKWARD_WALK') for x in self.ids],device=device)
        back_teacher=torch.tensor([x=='09_BACKWARD_WALK' for x in self.ids],device=device)
        turn_teacher=torch.tensor(['TURN' in x for x in self.ids],device=device)
        self.walk=walk_teacher[types];self.back=back_teacher[types];self.turn=turn_teacher[types]
        self.teacher_commands=canonical_task_commands(env.teacher_commands,self.ids)
        self.command=self.teacher_commands[types]
        self.event_scale=torch.full((n,),float(spec.get('walk_event_weight_scale',.25)),device=device)
        self.style_scale=float(spec.get('walk_reference_style_scale',.2))
        self.ankle=AnkleStepEvents(n,device,env.dt,period=1.,landing_deg=30.,takeoff_deg=110.,
                                  clearance_m=.002,air_time_s=.060,travel_m=.005,error_scale_deg=30.,max_exponent=6.)
        self.ankle.landing_deg=torch.where(self.back,torch.full((n,),110.,device=device),torch.full((n,),30.,device=device))[:,None]
        self.ankle.takeoff_deg=torch.where(self.back,torch.full((n,),30.,device=device),torch.full((n,),110.,device=device))[:,None]
        self.stride=StrideProgress(n,device,1.,env.dt,cap_m=.10,exponent=4.,hind_zero_m=.08,front_zero_m=.08,
                                   variance_normalization_m=.08,variance_max_age_periods=2.)
        self.width=HindLandingWidth(n,device,.06);self.waist=WaistHeightAcceleration(n,device,env.dt)
        self.air=torch.zeros(n,4,device=device);self.last_contact=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.previous_target=torch.zeros(n,15,device=device);self.valid_target=torch.zeros(n,device=device,dtype=torch.bool)
        self.diagnostics={};self.last_counter=None
        self.metadata={'schema':'sable-teacher-motion-rewards/2','walk_reward_source':'stand5_20261027_v44_stride80_all/9470 frozen reward mechanisms; weights only, no checkpoint weight transfer',
          'new_task_cost_mapping':'-weight * z/(1+z), z=normalized_error^2; goals, signs, tolerances and deadbands unchanged',
          'new_cost_max_magnitude_per_second':{'task_velocity_cost':2.,'task_yaw_cost':2.,'task_wrong_translation_cost':2.,
             'task_cross_axis_cost':1.,'task_wrong_turn_cost':2.,'task_turn_translation_cost':1.,'contact_material_slip_proxy':1.},
          'new_cost_sum_bound':'For either translation or turning, applicable direction costs plus material-point slip total at most8 per second; does not bound the separate pre-existing v44 physical/event costs',
          'original_v44_weights':V44_WEIGHTS,'walk_teacher_ids':['01_FORWARD_POLICY','09_BACKWARD_WALK'],
          'foot_angle_targets':{'01_FORWARD_POLICY':{'landing_deg':30.,'liftoff_deg':110.},'09_BACKWARD_WALK':{'landing_deg':110.,'liftoff_deg':30.},
             'definition':'Directed0..180deg acos(distal foot long-axis unit3D dot body-forward horizontal ray); not ankle joint angle',
             'feet':['HL','HR'],'events':'genuine >=2mm clearance continuously60ms and >=5mm unsigned center travel; one landing, then confirmed takeoff latched at previous last contact; startup takeoff excluded'},
          'event_weight_course':{'initial_scale':float(spec.get('walk_event_weight_scale',.25)),'setter':'set_walk_event_scale','targets_unchanged':True},
          'reference_style_scale':self.style_scale,
          'adaptations':['backwalk stride and positive body-velocity gates project along -body-forward; width and foot angle retain +body-forward',
             'heading-frame task vx/vy/yaw; pure translations command yaw0, turns retain signed yaw',
             'body height, orientation and soft clearance/contact targets follow phase reference, not old fixed stand height or old fixed clock',
             'old action-rate acts on complete executed targets, not residual only'],
          'loose_reference_adaptation':{'tolerances':'Read env.adaptive.sigma[teacher_id] columns toe_m/velocity_m_s/yaw_rad_s and clamp to its declared lower/upper bounds',
             'deadbands':'Read env.adaptive.deadband; fallback toe3mm,velocity5mm/s,yaw.025rad/s',
             'soft_terms':['gait_clearance','gait_contact','swing_peak_balance','swing_peak_shortfall'],
             'style_scale':self.style_scale,'semantic_yaw':'Continuous signed yaw cost remains separate; no yaw deadband on this cost',
             'foot_air_time_target':'cfg.rewards.foot_air_time_target; v44 fallback .08s',
             'event_angle_targets_unchanged':True},
          'excluded':['hip/knee preferred posture regions','continuous ankle phase/lag/direction costs','unavailable per-foot tangential-force cancellation'],
          'slip_scope':'Capsule geometric-lowest material-point velocity proxy: native COM velocity + omega cross(point-COM); not solver contact-patch velocity',
          'aggregation':'Weighted per-second values integrated once by dt. No positive-only clipping; direction/yaw costs survive.'}

    def set_walk_event_scale(self,value):
        v=torch.as_tensor(value,device=self.env.device,dtype=self.event_scale.dtype)
        if v.numel()==len(self.ids):v=v[self.env.teacher_ids]
        self.event_scale.copy_(v.expand_as(self.event_scale).clamp(0,1))

    def reset(self,ids):
        for tracker in (self.ankle,self.stride,self.width,self.waist):tracker.reset(ids)
        # v44 reset clears all tensor members. Restore the per-environment
        # configuration tensors after it clears the event state tensors.
        back=self.back[ids]
        self.ankle.landing_deg[ids]=torch.where(back,torch.full_like(back,110.,dtype=torch.float32),torch.full_like(back,30.,dtype=torch.float32))[:,None]
        self.ankle.takeoff_deg[ids]=torch.where(back,torch.full_like(back,30.,dtype=torch.float32),torch.full_like(back,110.,dtype=torch.float32))[:,None]
        self.air[ids]=0;self.last_contact[ids]=False;self.valid_target[ids]=False

    def apply(self,base_terms,ref):
        e=self.env;age=e.episode_length_buf*e.dt;n=e.num_envs
        if self.last_counter==e.common_step_counter:raise RuntimeError('Motion reward apply called twice for one control sample')
        self.last_counter=e.common_step_counter
        forward=body_forward_xy(e.base_quat);normal=e.contact_forces[:,e.feet_indices,2].clamp(min=0)
        contact=normal>1.;hind_contact=contact[:,2:]&(e.foot_clearance[:,2:]<=.001)
        angle=directed_ground_angle_deg(e.foot_long_axis_world[:,2:],forward)
        move_sign=torch.where(self.back,-torch.ones(n,device=e.device),torch.ones(n,device=e.device))
        walk_speed=self.command[:,0].abs()*self.walk
        movement_forward=forward*move_sign[:,None]
        self.ankle.update(hind_contact,e.foot_clearance[:,2:],angle,e.foot_center_world[:,2:,:2],walk_speed,age)
        self.stride.update(e.foot_center_world[:,:,:2],contact,e.foot_clearance,movement_forward,walk_speed,age,e.heading_linear_velocity[:,0]*move_sign)
        self.width.update(e.foot_contact_point[:,2:,:2],hind_contact,self.stride.hind_landing,forward,walk_speed,age)
        self.waist.update(e.rigid_states[:,e.waist_body_index,2])
        feet=e.rigid_states[:,e.feet_indices]
        com=feet[:,:,:3]+quat_apply(feet[:,:,3:7],e.contact_foot_com[None].expand(n,-1,-1))
        point_velocity=feet[:,:,7:10]+torch.cross(feet[:,:,10:13],e.foot_contact_point-com,dim=-1)
        slip_sq=point_velocity[:,:,:2].square().sum(-1)
        out=dict(base_terms)
        # These old generic entries are replaced for walk to avoid double-counting.
        zero=torch.zeros(n,device=e.device)
        for key in ('velocity','yaw','orientation','height','foot_clearance','slip','torque','action_rate'):
            if key in out:out[key]=torch.where(self.walk,zero,out[key])
        adaptive=getattr(e,'adaptive',None)
        if adaptive is not None:
            sigma=adaptive.sigma.clamp(adaptive.lower,adaptive.upper)[e.teacher_ids]
            deadband=adaptive.deadband
        else:
            speed=self.command[:,:2].norm(dim=-1);yaw=self.command[:,2].abs()
            sigma=torch.stack((torch.full_like(speed,.010),torch.maximum(speed*.4,torch.full_like(speed,.010)),
                               torch.maximum(yaw*.4,torch.full_like(yaw,.04))),1)
            deadband=torch.tensor([.003,.005,.025],device=e.device)
        sigma_toe,sigma_velocity,sigma_yaw=sigma[:,0],sigma[:,1],sigma[:,2]
        out.update(direction_terms(e.heading_linear_velocity[:,:2],e.heading_yaw_rate,self.command,self.turn,
                                   sigma_velocity,sigma_yaw,deadband[1]))
        out['contact_material_slip_proxy']=-(bounded_square(point_velocity[:,:,:2].norm(dim=-1)/.03)*contact).mean(1)
        target=e.executed_target;change=((target-self.previous_target)/.35).square().sum(1)*self.valid_target
        self.previous_target.copy_(target);self.valid_target[:]=True
        ref_quat=ref['quat'];inverse=ref_quat.clone();inverse[:,:3]*=-1
        gravity_ref=quat_apply(inverse,e.gravity_vec)
        qe_limit=(e.hard_limits[:,0]-e.dof_pos).clamp(min=0)+(e.dof_pos-e.hard_limits[:,1]).clamp(min=0)
        target_raw=getattr(e,'current_reference_target',ref['q'])+getattr(e,'current_offset_rad',torch.zeros_like(ref['q']))
        active_limits=e.hard_limits[e.active]
        target_excess=(active_limits[:,0]-target_raw).clamp(min=0)+(target_raw-active_limits[:,1]).clamp(min=0)
        force_sigma=float(getattr(e.cfg.gait,'force_sigma_n',15.));speed_sigma=float(getattr(e.cfg.gait,'stance_speed_sigma_m_s',.05))
        surface_min=e.ref['surface'].amin(1)[e.teacher_ids];surface_max=e.ref['surface'].amax(1)[e.teacher_ids]
        height_scale=(surface_max-surface_min).clamp(min=.008)
        swing=((ref['surface']-surface_min)/height_scale).clamp(0,1)
        gait_contact=(swing*torch.exp(-normal.square()/force_sigma**2)+(1-swing)*torch.exp(-slip_sq/speed_sigma**2)).mean(1)
        clearance_excess=((e.foot_clearance-ref['surface']).abs()-deadband[0]).clamp(min=0)
        clearance=(clearance_excess/sigma_toe[:,None]).square().clamp(max=9).mean(1)
        filtered=contact|self.last_contact;first=(self.air>0)&filtered;self.air+=e.dt
        air_time_target=float(getattr(getattr(e.cfg,'rewards',None),'foot_air_time_target',.08))
        air_reward=((self.air-air_time_target)*first).sum(1);self.air*=~filtered;self.last_contact.copy_(contact)
        signed_speed=e.heading_linear_velocity[:,0]*move_sign
        forward_progress=(signed_speed/.12).clamp(0,1)*torch.exp(-e.heading_linear_velocity[:,1].square()/.0025-e.heading_yaw_rate.square()/.25)*(walk_speed>.02)*(~e.true_fall)
        accel=(e.dof_vel-e.last_dof_vel)/e.dt
        proximal=((accel[:,e.proximal_indices].abs()/150.-1).clamp(min=0).square().clamp(max=16)).mean(1)*(e.episode_length_buf>=2)
        collision=(e.contact_forces[:,e.penalised_contact_indices].norm(dim=-1)>.1).float().sum(1)
        peak_target=surface_max.clamp(min=.008)
        peak_deficit=((peak_target-self.stride.peak-deadband[0]).clamp(min=0)/sigma_toe[:,None]).square().clamp(max=9)
        peak_shortfall=(peak_deficit*self.stride.landing).sum(1)/(4*e.dt)
        peak_residual=self.stride.peak-peak_target
        peak_spread=(peak_residual-peak_residual.mean(1,keepdim=True)).abs()
        peak_balance=(((peak_spread-deadband[0]).clamp(min=0)/sigma_toe[:,None]).square().clamp(max=9).mean(1))*self.stride.landing.any(1)
        yaw_error=e.heading_yaw_rate # Walk targets are canonical zero yaw.
        values={'action_rate':change,'ang_vel_xy':e.base_ang_vel[:,:2].square().sum(1),
          'ankle_event_angle':self.ankle.event_cost,'base_height':(e.root_states[:,2]-e.env_origins[:,2]-ref['root'][:,2,3]).square(),
          'collision':collision,'dof_acc':accel.square().sum(1)*(e.episode_length_buf>=2),'dof_pos_limits':qe_limit.sum(1),
          'feet_air_time':air_reward,'forward_speed':forward_progress,'gait_clearance':clearance,'gait_contact':gait_contact,
          'hind_landing_width':self.width.cost,'lin_vel_z':e.root_states[:,9].square(),
          'orientation':(e.projected_gravity-gravity_ref).square().sum(1),'proximal_acceleration_peak':proximal,
          'stride_progress':self.stride.reward*(~e.true_fall),'stride_shortfall':self.stride.shortfall,'stride_variance':self.stride.variance_cost,
          'swing_peak_balance':peak_balance,'swing_peak_shortfall':peak_shortfall,'target_limits':(target_excess/.35).square().sum(1),
          'termination':e.true_fall.float(),'torques':e.torques.square().sum(1),
          'tracking_ang_vel':torch.exp(-(yaw_error.abs()-deadband[2]).clamp(min=0).square()/sigma_yaw.square()),
          'tracking_lin_vel':torch.exp(-((e.heading_linear_velocity[:,:2]-self.command[:,:2]).norm(dim=-1)-deadband[1]).clamp(min=0).square()/sigma_velocity.square()),
          'waist_vertical_acceleration':self.waist.cost(5.),'yaw_rate_error':yaw_error.square()}
        event_names={'ankle_event_angle','stride_progress','stride_shortfall','stride_variance','swing_peak_balance','swing_peak_shortfall'}
        for name,weight in V44_WEIGHTS.items():
            scale=self.event_scale if name in event_names else 1.
            if name in ('gait_clearance','gait_contact','swing_peak_balance','swing_peak_shortfall'):scale=scale*self.style_scale
            out['walk_v44_'+name]=values[name]*weight*self.walk*scale
        self.diagnostics={'foot_angle_deg_hind':angle.clone(),'ankle_event_landed':self.ankle.landed.clone(),
           'ankle_event_lifted':self.ankle.lifted.clone(),'ankle_event_cost_raw':self.ankle.event_cost.clone(),
           'ankle_landing_angle_deg':self.ankle.landing_angle_deg.clone(),
           'ankle_takeoff_angle_deg':self.ankle.takeoff_angle_deg.clone(),
           'walk_event_weight_scale':self.event_scale.clone(),'ankle_landing_target_deg':self.ankle.landing_deg[:,0].clone(),
           'ankle_takeoff_target_deg':self.ankle.takeoff_deg[:,0].clone(),'stride_signed_m':self.stride.last_signed.clone(),
           'stride_landed':self.stride.landing.clone(),'stride_variance_m2':self.stride.variance_m2.clone(),
           'stride_variance_valid':self.stride.variance_event.clone(),'material_point_velocity_proxy':point_velocity.clone(),
           'canonical_task_command':self.command.clone(),'adaptive_sigma_used':sigma.clone(),
           'tracking_deadbands_used':deadband[None].expand(n,-1).clone(),
           'walk_reference_style_scale':torch.full((n,),self.style_scale,device=e.device),
           'foot_air_time_target_s':torch.full((n,),air_time_target,device=e.device)}
        return out
