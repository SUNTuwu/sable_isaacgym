"""Pace-only task-space guidance and progressive hind residual slew control.

Physical motor limits and authored references are unchanged. The controller
limits the learned residual, not the reference's legitimate foot motion.
"""
import math
import torch

PACE_IDS=('02_STRAFE_LEFT','03_STRAFE_RIGHT')
HIND_ACTIONS=(1,5,6,10,11,12,13)
METADATA={
 'version':'pace_tip_stability_v2','teacher_ids':list(PACE_IDS),'feet_order':['FL','FR','HL','HR'],
 'reference':'Unchanged smoothed hand-K reference amplitude1 plus bounded20deg residual; no joint-angle imitation',
 'frame':'Distal capsule tips in respective reference/actual root frames, matched by phase',
 'curriculum_updates':60,'initial_residual_rate_deg_s':2000.,
 'final_hind_residual_rate_deg_s':60.,'final_shared_hip_roll_rate_deg_s':120.,
 'controller':'Geometric rate curriculum, hind independent motors only; first reset command warm-starts balance correction. Forelegs and spine unchanged.',
 'endpoint_reward':'6*weighted_mean(exp(-error_squared/sigma_squared))+6*worst_foot_quality; weights[1,1,2,2]',
 'position_cost':'-3*weighted_mean(E/(1+E))-3*max(E/(1+E)), E=(tip_error/sigma)^2; sigma20->15mm',
 'error_velocity_cost':'Reference-relative 3D tip error derivative, scale0.15m/s, bounded3/s; excludes reset warmup',
 'residual_acceleration_cost':'Hind actually executed residual acceleration, scale30rad/s2, bounded1/s; excludes reset warmup',
 'acceptance':{'horizon_s':16.,'survival_fraction':1.,'hind_tip_rmse_mm':10.,
   'hind_reference_relative_jump_p95_mm':3.,'xy_rmse_m_s':.03,'yaw_rmse_rad_s':.15,
   'requires_full_curriculum':True,'sampling_s':.02},
 'pace_static_width_removed':'Only pace: remove old40+/-10mm landing-width and60mm stance-width costs; actual phase reference3D tips are authoritative. Four forward/backward30mm and all rotation constraints remain.',
 'pace_cost_rebalance':'Direct upstroke counting retained; frequency max3/s, old raw world-Z velocity cost replaced by reference-relative 3D error velocity. Rotation constraints unchanged.'}


def tracking_scores(error,sigma):
    energy=error.square().sum(-1)/sigma[:,None].square()
    quality=torch.exp(-energy);bounded=energy/(1+energy)
    weights=error.new_tensor([1.,1.,2.,2.])/6
    return (6*(quality*weights).sum(1)+6*quality.min(1)[0],
            -3*(bounded*weights).sum(1)-3*bounded.max(1)[0])


class PaceStability:
    def __init__(self,env,enabled=False):
        self.env=env;n=env.num_envs;dev=env.device
        self.enabled=bool(enabled)
        types=torch.tensor([c['id'] in PACE_IDS for c in env.cfg.teacher_bank['clips']],device=dev)
        self.mask=types[env.teacher_ids]&self.enabled
        self.hind=torch.tensor(HIND_ACTIONS,device=dev)
        self.strength=torch.zeros(n,device=dev)
        self.previous_error=torch.zeros(n,4,3,device=dev)
        self.previous_offset=torch.zeros(n,15,device=dev)
        self.previous_offset_velocity=torch.zeros_like(self.previous_offset)
        self.ready=torch.zeros(n,device=dev,dtype=torch.long)
        self.error=torch.zeros_like(self.previous_error)
        self.jump=torch.zeros(n,4,device=dev)
        self.offset_delta=torch.zeros(n,7,device=dev)
        self.valid=torch.zeros(n,device=dev,dtype=torch.bool)

    def reset(self,ids):
        for v in (self.previous_error,self.previous_offset,self.previous_offset_velocity,self.ready):v[ids]=0

    def set_strength(self,value,ids=None):
        assert 0<=float(value)<=1
        if ids is None:self.strength[:]=float(value)
        else:self.strength[ids]=float(value)

    def rates(self):
        final=self.strength.new_tensor([120.,60.,60.,60.,60.,60.,60.])
        return 2000.*(final[None]/2000.).pow(self.strength[:,None])*math.pi/180

    def filter(self,previous,requested,first):
        if not self.enabled:return requested
        limit=self.rates()*self.env.dt
        old=previous[:,self.hind];new=requested[:,self.hind]
        limited=old+(new-old).clamp(-limit,limit)
        result=requested.clone()
        result[:,self.hind]=torch.where((self.mask&~first&(self.strength>0))[:,None],limited,new)
        return result

    def apply(self,terms,error,offset):
        self.error=error.clone();delta=error-self.previous_error
        self.valid=(self.ready>=2)&self.mask
        self.jump=delta.norm(dim=-1)*self.valid[:,None]
        offset_velocity=(offset-self.previous_offset)/self.env.dt
        offset_acceleration=(offset_velocity-self.previous_offset_velocity)/self.env.dt
        self.offset_delta=(offset-self.previous_offset)[:,self.hind]*self.valid[:,None]
        sigma=.020-.005*self.strength
        reward,position_cost=tracking_scores(error,sigma)
        ev=(delta.norm(dim=-1)/(self.env.dt*.15)).square()
        b=ev/(1+ev);weights=error.new_tensor([1.,1.,2.,2.])/6
        velocity_cost=(-1.5*(b*weights).sum(1)-1.5*b.max(1)[0])*self.valid
        a=(offset_acceleration[:,self.hind]/30.).square()
        acceleration_cost=-(a/(1+a)).mean(1)*self.valid
        terms['reference_endpoint_reward']=torch.where(self.mask,reward,terms['reference_endpoint_reward'])
        # A static40mm target was competing with the authored lateral foot path.
        for key in ('hind_landing_lateral_cost','hind_lateral_over60_cost'):
            if key in terms:terms[key]=torch.where(self.mask,torch.zeros_like(terms[key]),terms[key])
        terms['pace_endpoint_position_cost']=position_cost*self.mask
        terms['pace_endpoint_error_velocity_cost']=velocity_cost
        terms['pace_residual_acceleration_cost']=acceleration_cost
        self.previous_error.copy_(error);self.previous_offset.copy_(offset)
        self.previous_offset_velocity.copy_(offset_velocity);self.ready+=1
        return terms

    def statistics(self,train_per,teacher_count,alive=None):
        n=train_per*teacher_count;active=self.mask[:n].float()
        if alive is not None:active=active*alive[:n].float()
        valid=self.valid[:n].float()*active
        values=[*[self.error[:n,k].square().sum(1)*active for k in range(4)],active,
                *[self.jump[:n,k].square()*valid for k in range(4)],valid,
                self.offset_delta[:n].square().mean(1)*valid]
        return torch.stack(values,1).reshape(teacher_count,train_per,len(values)).sum(1)


def statistics_report(sums,strength):
    x=sums.tolist()
    return {'pace_stability_strength':float(strength),'pace_stability_config':METADATA,
      'pace_tip_rmse_mm_per_foot':[1000*math.sqrt(x[k]/x[4]) if x[4] else None for k in range(4)],
      'pace_reference_relative_jump_rms_mm_per_foot':[1000*math.sqrt(x[k]/x[9]) if x[9] else None for k in range(5,9)],
      'pace_hind_residual_delta_rms_deg':math.degrees(math.sqrt(x[10]/x[9])) if x[9] else None}


def dense_report(records,dt,strength):
    """records: (error_xyz, executed_offset, alive_before_step); no reset mixing."""
    import numpy as np
    error=np.stack([r[0] for r in records]);offset=np.stack([r[1] for r in records])
    alive=np.stack([r[2] for r in records]).astype(bool)
    rmse=[]
    for foot in range(4):
        value=(error[:,:,foot]**2).sum(-1)[alive]
        rmse.append(float(np.sqrt(value.mean())*1000) if len(value) else None)
    adjacent=alive[1:]&alive[:-1]
    # Exclude the same two reset warmup samples as the online diagnostic.
    adjacent[:1]=False
    jump=np.linalg.norm(np.diff(error,axis=0)[:,:,2:],axis=-1)[adjacent]
    delta=np.abs(np.diff(offset,axis=0)[:,:,HIND_ACTIONS])[adjacent]*180/math.pi
    p95=float(np.percentile(jump,95)*1000) if jump.size else None
    delta95=np.percentile(delta,95,axis=0).tolist() if delta.size else [None]*7
    return {'sampling_s':dt,'curriculum_strength':float(strength),'tip_rmse_mm_per_foot':rmse,
      'hind_reference_relative_jump_p95_mm':p95,'hind_residual_delta_p95_deg':delta95,
      'scope':'All four first episodes, every20ms, pre-reset states; no post-failure samples, reset warmup excluded from jumps',
      'hind_motor_order':list(HIND_ACTIONS)}


def acceptance(report,dense):
    limits=METADATA['acceptance'];tip=dense['tip_rmse_mm_per_foot'];jump=dense['hind_reference_relative_jump_p95_mm']
    checks={'full_curriculum':dense['curriculum_strength']>=1-1e-6,
      'full_horizon':report['horizon_s']>=limits['horizon_s'] and report['executed_simulation_s']>=limits['horizon_s']-1e-6,
      'all_survive':report['survival_fraction']==1.,
      'hind_tips':all(v is not None and v<=limits['hind_tip_rmse_mm'] for v in tip[2:]),
      'reference_relative_jitter':jump is not None and jump<=limits['hind_reference_relative_jump_p95_mm'],
      'xy_tracking':report['xy_rmse_m_s']<=limits['xy_rmse_m_s'],
      'yaw_tracking':report['yaw_rmse_rad_s']<=limits['yaw_rmse_rad_s']}
    return {'passed':all(checks.values()),'checks':checks,'criteria':limits}
