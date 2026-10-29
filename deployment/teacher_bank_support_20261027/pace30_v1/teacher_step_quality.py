"""Task-directed real steps and strict drift/placement costs.

Two unchanged v44 AnkleStepEvents detectors supply four-foot genuine landings.
Real-air attempts with less than5mm travel are retained separately for short
step penalties; they cannot earn a positive reward or count as a valid step.
"""
import math
import torch
from ankle_events import AnkleStepEvents
from teacher_yaw_integral import YawIntegral,METADATA as YAW_METADATA

PROFILE='strict_steps_v1'
NARROW_HIND_TEACHERS=['01_FORWARD_POLICY','06_BACKWARD','09_BACKWARD_WALK','10_FORWARD_TROT']
METADATA={
 'profile':PROFILE,'revision':'pace30_yaw_integral_exp_v6','feet_order':['FL','FR','HL','HR'],
 'hind_target30_teacher_ids':NARROW_HIND_TEACHERS,
 'hind_target30_profile':{'target_m':.03,'free_band':None,'hard_limit':None,
    'formula':'-expm1(abs(signed_outward_distance-.030)/.010); numerical exponent guard6 only',
    'raw_penalty_at_5mm_error':math.expm1(.5),'scope':'Each hind contact point independently, continuously during stance; average over two hind feet, no free band. Signed own-side distance penalizes crossed legs too. Replaces both previous landing-width and beyond60mm costs for four straight teachers.'},
 'hind_lateral_target_m':.04,'hind_lateral_free_range_m':[.03,.05],
 'hind_lateral_hard_limit_m':.06,'width_scope':'Each hind foot, signed outward distance from current root/body midline along heading-left axis; exponential at real-air landing, extra continuous penalty while contact is beyond60mm',
 'height_threshold_m':.01,'height_reward_cap_m':.03,'low_height_fixed_score':-2.,
 'invalid_stride_below_m':.02,'stride_reward_start_m':.05,'stride_reward_cap_m':.10,
 'invalid_stride_fixed_score':-2.,'height_reward_max':2.,'stride_reward_max':3.,
 'minimum_swing_duration_s':.2,'short_swing_fixed_score':-2.,
 'swing_duration_scope':'Sum of20ms noncontact intervals from observed stance departure to first landing; requires v44 continuous>=2mm clearance for>=60ms. Sub60ms contact chatter and startup in air are excluded; genuine60..180ms flights are penalized.',
 'yaw_integral_exponential':YAW_METADATA,
 'survival_reward_rebalance_per_second':15.,'survival_rebalance_scope':'Additional alive15/s during this stricter reward stage; offsets newly added negative terms to reduce early-termination incentive. Existing motion rewards and costs retain their magnitudes.',
 'joint_acceleration_free_rad_s2':150.,'joint_acceleration_scope':'All15 independent active joints, first two reset samples excluded; simulation threshold, not measured hardware rating',
 'direction_costs':'No cross-axis or translation-yaw deadband; fixed scales20mm/s and0.15rad/s; -4*z/(1+z), z=normalized_error squared. Each cost bounded by4/s, with stronger small-error gradients than the earlier weak adaptive penalties; requested turn yaw remains exempt from zero-yaw cost',
 'detector':'Unchanged v44 AnkleStepEvents x2: prior real contact, age>=1s, continuous>=2mm clearance for60ms, >=5mm unsigned foot-center travel; no startup/chatter event',
 'valid_step':'Genuine landing plus swing duration>=200ms, signed task-direction stride>=20mm, positive actual body task response, not falling',
 'distance_frame':'World foot-center displacement from last contact before flight, projected onto task direction latched at departure; translation uses commanded body XY, turns use yaw-sign tangent about root',
 'height_scope':'Peak actual capsule-bottom terrain clearance during flight; positive height reward requires valid directed stride',
 'event_normalization':'Per-foot score times reference_period/(4*dt), integrated once by dt; width uses two hind feet. Maxima are per nominal four-step cycle, not a cap on events per second',
 'replaced_walk_terms':['hind_landing_width','stride_progress','stride_shortfall','stride_variance','swing_peak_balance','swing_peak_shortfall','dof_acc','proximal_acceleration_peak','yaw_rate_error'],
 'reference_clearance_scope':'Direct phase clearance imitation removed for all teachers; event height rules replace it. Joint/toe references remain loose guides.'}


def exp_ramp(value,start,cap):
    return torch.expm1(3*((value-start)/(cap-start)).clamp(0,1))/math.expm1(3)


def event_scores(height,distance,attempt,valid):
    return {'step_height_reward':2*exp_ramp(height,.01,.03)*valid,
            'step_low_height_cost':-2*(height<.01-1e-7)*attempt,
            'step_distance_reward':3*exp_ramp(distance,.05,.10)*valid,
            'step_invalid_distance_cost':-2*(distance<.02-1e-7)*attempt}


def placement_scores(outward):
    excess=((outward-.04).abs()-.01).clamp(min=0)
    # Numerical protection is far outside the free band; 10mm/20mm excess costs
    # 0.755/2 before event normalization. Crossing never satisfies signed width.
    exponential=-2*torch.expm1((excess/.02).clamp(max=3))/math.expm1(1)
    hard=-2.*(outward.abs()>.06+1e-7)
    return exponential,hard


def placement30_scores(outward):
    return -torch.expm1(((outward-.03).abs()/.01).clamp(max=6))


def strict_direction_terms(actual_xy,yaw,command):
    speed=command[:,:2].norm(dim=1);turn=command[:,2].abs()>.001
    axis=command[:,:2]/speed[:,None].clamp(min=1e-8)
    cross=actual_xy[:,0]*axis[:,1]-actual_xy[:,1]*axis[:,0]
    def cost(error):
        z=error.square()
        return -4*(1-1/(1+z))
    return {'task_cross_axis_cost':cost(cross/.02)*(~turn),
            'translation_zero_yaw_cost':cost(yaw/.15)*(~turn),
            'task_turn_translation_cost':cost(actual_xy.norm(dim=1)/.02)*turn}


class SwingDurationTracker:
    """Time observed departures; retain the v44 continuous60ms air gate."""
    def __init__(self,n,device,dt):
        self.dt=dt
        self.prior=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.active=torch.zeros_like(self.prior)
        self.elapsed=torch.zeros(n,4,device=device)
        self.peak=torch.zeros_like(self.elapsed)
        self.clear_run=torch.zeros_like(self.elapsed);self.max_clear=torch.zeros_like(self.elapsed)
    def reset(self,ids):
        for value in (self.prior,self.active,self.elapsed,self.peak,self.clear_run,self.max_clear):value[ids]=0
    def update(self,contact,clearance):
        departure=self.prior&~contact
        self.active|=departure
        self.elapsed=torch.where(departure,torch.zeros_like(self.elapsed),self.elapsed)
        self.peak=torch.where(departure,torch.zeros_like(self.peak),self.peak)
        self.clear_run=torch.where(departure,torch.zeros_like(self.clear_run),self.clear_run)
        self.max_clear=torch.where(departure,torch.zeros_like(self.max_clear),self.max_clear)
        airborne=self.active&~contact
        self.elapsed+=airborne*self.dt
        self.peak=torch.where(airborne,torch.maximum(self.peak,clearance),self.peak)
        self.clear_run=torch.where(airborne&(clearance>=.002-1e-7),self.clear_run+self.dt,torch.zeros_like(self.clear_run))
        self.max_clear=torch.maximum(self.max_clear,self.clear_run)
        landed=contact&self.active&(self.max_clear>=.06-1e-7)
        duration=self.elapsed.clone()
        short=landed&(duration<.2-1e-7)
        self.active&=~contact;self.prior.copy_(contact)
        return landed,duration,short


class TeacherStepQuality:
    def __init__(self,env,command):
        self.env=env;self.command=command;n=env.num_envs;dev=env.device
        teacher_narrow=torch.tensor([c['id'] in NARROW_HIND_TEACHERS for c in env.cfg.teacher_bank['clips']],device=dev)
        self.narrow_hind=teacher_narrow[env.teacher_ids]
        self.detectors=[AnkleStepEvents(n,dev,env.dt) for _ in range(2)]
        self.swing_duration=SwingDurationTracker(n,dev,env.dt)
        self.yaw_integral=YawIntegral(n,dev,env.dt)
        self.axes=torch.zeros(n,4,2,device=dev);self.peak=torch.zeros(n,4,device=dev)
        self.valid=torch.zeros(n,4,device=dev,dtype=torch.bool)
        self.previous_velocity=torch.zeros(n,15,device=dev);self.acceleration_ready=torch.zeros(n,device=dev,dtype=torch.bool)
        self.diagnostics={}

    def reset(self,ids):
        for detector in self.detectors:detector.reset(ids)
        self.swing_duration.reset(ids)
        self.yaw_integral.reset(ids)
        self.axes[ids]=0;self.peak[ids]=0;self.valid[ids]=False
        self.previous_velocity[ids]=0;self.acceleration_ready[ids]=False

    def apply(self,terms,forward,contact,age):
        e=self.env;n=e.num_envs;period=e.ref_duration[e.teacher_ids];xy=e.foot_center_world[:,:,:2]
        duration_landed,duration,short_duration=self.swing_duration.update(contact,e.foot_clearance)
        left=torch.stack((-forward[:,1],forward[:,0]),1)
        speed=self.command[:,:2].norm(dim=1);turn=self.command[:,2].abs()>.001
        direction=forward*self.command[:,0:1]+left*self.command[:,1:2]
        direction=direction/direction.norm(dim=1,keepdim=True).clamp(min=1e-8)
        radial=xy-e.root_states[:,None,:2]
        tangent=torch.stack((-radial[:,:,1],radial[:,:,0]),-1)*self.command[:,2,None,None].sign()
        tangent=tangent/tangent.norm(dim=2,keepdim=True).clamp(min=1e-8)
        desired=torch.where(turn[:,None,None],tangent,direction[:,None,:].expand(-1,4,-1))
        active=((speed>.001)|turn).float();eligible=(active.bool()&(age>=1.))[:,None]
        genuine=[];attempts=[];distances=[]
        for k,d in enumerate(self.detectors):
            sl=slice(2*k,2*k+2);touch=contact[:,sl];clearance=e.foot_clearance[:,sl]
            departure=d.initialized&d.previous_contact&~touch&eligible&~d.ready_for_landing
            self.axes[:,sl]=torch.where(departure[:,:,None],desired[:,sl],self.axes[:,sl])
            self.peak[:,sl]=torch.where(departure,torch.zeros_like(clearance),self.peak[:,sl])
            self.peak[:,sl]=torch.where((d.candidate|departure)&~touch,torch.maximum(self.peak[:,sl],clearance),self.peak[:,sl])
            candidate=d.candidate.clone()
            d.update(touch,clearance,torch.zeros_like(clearance),xy[:,sl],active,age)
            real_air_attempt=touch&candidate&eligible&(d.max_clear>=.060-1e-7)
            genuine.append(d.landed.clone());attempts.append(real_air_attempt)
            distances.append(((xy[:,sl]-d.start_xy)*self.axes[:,sl]).sum(-1))
        genuine=torch.cat(genuine,1);attempt=torch.cat(attempts,1);distance=torch.cat(distances,1)
        along=(e.heading_linear_velocity[:,:2]*self.command[:,:2]).sum(1)/speed.clamp(min=1e-8)
        moving=torch.where(turn,e.heading_yaw_rate*self.command[:,2].sign()>.005,along>.005)
        self.valid=genuine&(duration>=.2-1e-7)&(distance>=.02-1e-7)&moving[:,None]&(~e.true_fall[:,None])
        scores=event_scores(self.peak,distance,attempt,self.valid)
        for name,score in scores.items():terms[name]=score.sum(1)*period/(4*e.dt)
        terms['step_short_duration_cost']=-2.*short_duration.sum(1)*period/(4*e.dt)
        terms['survival_reward_rebalance']=torch.ones(n,device=e.device)*15.
        terms['yaw_integral_exp_cost']=self.yaw_integral.update(e.heading_yaw_rate,self.command[:,2])
        # Task-specific direction constraints replace their earlier weak terms.
        strict=strict_direction_terms(e.heading_linear_velocity[:,:2],e.heading_yaw_rate,self.command)
        terms.update(strict)
        terms['task_yaw_cost']=terms['task_yaw_cost']*turn
        for name in METADATA['replaced_walk_terms']:
            key='walk_v44_'+name
            if key in terms:terms[key]=torch.zeros(n,device=e.device)
        # Broad phase-pose reference remains, but event clearance is authoritative.
        for name in ('foot_clearance','walk_v44_gait_clearance'):
            if name in terms:terms[name]=torch.zeros(n,device=e.device)
        delta=e.foot_contact_point[:,2:,:2]-e.root_states[:,None,:2]
        y=(delta*left[:,None,:]).sum(-1);outward=y*y.new_tensor([1.,-1.])
        width,hard=placement_scores(outward)
        width=torch.where(self.narrow_hind[:,None],placement30_scores(outward),width)
        hard=torch.where(self.narrow_hind[:,None],torch.zeros_like(hard),hard)
        legacy_width=(width*attempt[:,2:]).sum(1)*period/(2*e.dt)
        stance_width=(placement30_scores(outward)*contact[:,2:]).mean(1)
        terms['hind_landing_lateral_cost']=torch.where(self.narrow_hind,stance_width,legacy_width)
        terms['hind_lateral_over60_cost']=(hard*contact[:,2:]).mean(1)
        velocity=e.dof_vel[:,e.active];accel=(velocity-self.previous_velocity)/e.dt
        accel_valid=self.acceleration_ready&(e.episode_length_buf>=2)
        excess=(accel.abs()/150.-1).clamp(min=0).square().clamp(max=4)
        terms['joint_acceleration_excess_cost']=-2*excess.mean(1)*accel_valid
        self.previous_velocity.copy_(velocity);self.acceleration_ready[:]=True
        self.diagnostics={'quality_attempted_landing':attempt.clone(),'quality_genuine_landing':genuine.clone(),
            'quality_valid_landing':self.valid.clone(),'quality_peak_height_m':self.peak.clone(),
            'quality_signed_stride_m':distance.clone(),'quality_hind_outward_m':outward.clone(),
            'quality_hind_over60_contact':((outward.abs()>.06+1e-7)&contact[:,2:]).clone(),
            'quality_hind_contact':contact[:,2:].clone(),'quality_joint_acceleration_rad_s2':accel.clone(),
            'quality_joint_acceleration_valid':accel_valid.clone()}
        self.diagnostics.update(quality_duration_landed=duration_landed.clone(),
            quality_swing_duration_s=duration.clone(),quality_short_duration_landed=short_duration.clone(),
            quality_yaw_integral_rad=self.yaw_integral.value.clone())
        return terms

    def statistics(self,train_per,teacher_count):
        n=train_per*teacher_count;d=self.diagnostics;a=d['quality_attempted_landing'][:n]
        h=d['quality_peak_height_m'][:n];s=d['quality_signed_stride_m'][:n]
        values=[a.sum(1),d['quality_genuine_landing'][:n].sum(1),d['quality_valid_landing'][:n].sum(1),
            ((h<.01-1e-7)&a).sum(1),((s<.02-1e-7)&a).sum(1),(h*a).sum(1),(s*a).sum(1),
            d['quality_hind_over60_contact'][:n].sum(1),d['quality_hind_contact'][:n].sum(1),
            ((d['quality_joint_acceleration_rad_s2'][:n].abs()>150)&d['quality_joint_acceleration_valid'][:n,None]).sum(1),
            d['quality_joint_acceleration_valid'][:n].float()*15,
            d['quality_short_duration_landed'][:n].sum(1),d['quality_duration_landed'][:n].sum(1),
            (d['quality_swing_duration_s'][:n]*d['quality_duration_landed'][:n]).sum(1),
            d['quality_yaw_integral_rad'][:n],d['quality_yaw_integral_rad'][:n].abs(),torch.ones(n,device=self.env.device),
            (d['quality_hind_outward_m'][:n]*d['quality_hind_contact'][:n]).sum(1),d['quality_hind_contact'][:n].sum(1)]
        return torch.stack(values,1).reshape(teacher_count,train_per,19).sum(1)


def statistics_report(sums):
    x=sums.tolist();attempt=x[0]
    return dict(step_quality_profile=PROFILE,step_quality_config=METADATA,
        quality_attempted_landings=int(x[0]),quality_genuine_landings=int(x[1]),quality_valid_landings=int(x[2]),
        quality_low_height_landings=int(x[3]),quality_short_stride_landings=int(x[4]),
        quality_mean_peak_height_m=x[5]/attempt if attempt else None,
        quality_mean_signed_stride_m=x[6]/attempt if attempt else None,
        quality_hind_over60_contact_fraction=x[7]/x[8] if x[8] else None,
        quality_joint_acceleration_excess_fraction=x[9]/x[10] if x[10] else None,
        quality_short_duration_landings=int(x[11]),quality_physical_landings=int(x[12]),
        quality_mean_swing_duration_s=x[13]/x[12] if x[12] else None,
        quality_mean_yaw_integral_deg=math.degrees(x[14]/x[16]) if x[16] else None,
        quality_mean_abs_yaw_integral_deg=math.degrees(x[15]/x[16]) if x[16] else None,
        quality_mean_hind_outward_contact_mm=1000*x[17]/x[18] if x[18] else None,
        step_quality_scope='Counts and event-weighted means over this rollout; short real-air attempts are not valid steps; widths and acceleration fractions use active contact/joint samples')
