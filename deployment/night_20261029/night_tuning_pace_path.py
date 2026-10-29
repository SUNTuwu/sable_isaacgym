"""Explicit per-teacher live reward revisions, with a fixed physical acceptance gate."""
import json,os,math
from pathlib import Path
import torch
CONTROL=Path(os.environ.get('SABLE_NIGHT_CONTROL_FILE',str(Path(os.environ['SABLE_GYM_ROOT'])/'deployment/night_20261029/control.json')))
def refresh(e):
    try:e.night_control=json.loads(CONTROL.read_text())
    except (OSError,ValueError):return

def signed_cross_distance(displacement_world,start_yaw,command_xy):
    """Signed displacement normal to the commanded line in the episode-start frame."""
    c=start_yaw.cos();s=start_yaw.sin()
    fixed_x=c*displacement_world[:,0]+s*displacement_world[:,1]
    fixed_y=-s*displacement_world[:,0]+c*displacement_world[:,1]
    speed=command_xy.norm(dim=1)
    cross=(fixed_x*command_xy[:,1]-fixed_y*command_xy[:,0])/speed.clamp(min=1e-6)
    return torch.where(speed>1e-6,cross,torch.zeros_like(cross))

def pace_cross_track_cost(e,sigma):
    if not math.isfinite(sigma) or sigma<=0:raise ValueError('cross_track_sigma_m must be finite and positive')
    boundary=getattr(e,'turn_boundary',None)
    if boundary is None or getattr(boundary,'displacement',None) is None:
        raise RuntimeError('cross_track_sigma_m requires the existing per-episode world displacement integral')
    # TurnBoundary.evaluate already integrates once per step and reset clears it.
    # This reward is read-only: no new integrator, termination, or instantaneous-velocity cost.
    cross=signed_cross_distance(boundary.displacement,e.translation_guard.start_yaw,e.teacher_commands[e.teacher_ids,:2])
    u=cross/sigma
    return -8*u.square()/(1+u.square())
def apply(e,terms,fall):
    settings=getattr(e,'night_control',{}).get('teachers',{})
    ids=[c['id'] for c in getattr(e.cfg,'teacher_bank',{}).get('clips',[])]
    for i,name in enumerate(ids):
        cfg=settings.get(name,{})
        mask=e.teacher_ids==i
        if 'TURN' in name:
            # A transferred trot keeps its cadence, not a straight-walking mirror constraint.
            terms['v2_phase_symmetry']=torch.where(mask,torch.zeros_like(terms['v2_phase_symmetry']),terms['v2_phase_symmetry'])
            # Speed is the ground-plane norm: independent of world heading.
            speed=e.root_states[:,7:9].norm(dim=1)
            u=speed/.06
            terms['v2_cross_motion']=torch.where(mask,-8*u.square()/(1+u.square()),terms['v2_cross_motion'])
            sigma=max(.02,min(.10,float(cfg.get('turn_velocity_sigma',.06))))
            terms['v2_velocity']=torch.where(mask,4*torch.exp(-(speed/sigma).square()),terms['v2_velocity'])
        if name in ('02_STRAFE_LEFT','03_STRAFE_RIGHT') and 'cross_track_sigma_m' in cfg:
            value=pace_cross_track_cost(e,float(cfg['cross_track_sigma_m']))
            terms['v2_cross_motion']=torch.where(mask,value,terms['v2_cross_motion'])
        for key,multiplier in cfg.get('multipliers',{}).items():
            if key in terms and key!='v2_fall_event':terms[key]=torch.where(mask,terms[key]*max(.25,min(4.,float(multiplier))),terms[key])
        # A smooth rate cost avoids flat saturated -8/s feedback while still penalizing yaw.
        if cfg.get('yaw_sigma') and 'TURN' not in name:
            u=e.heading_yaw_rate/float(cfg['yaw_sigma']);value=-8*u.square()/(1+u.square())
            value*=cfg.get('multipliers',{}).get('v2_translation_yaw_rate',1.)
            terms['v2_translation_yaw_rate']=torch.where(mask,value,terms['v2_translation_yaw_rate'])
        if cfg.get('straight_frame') and 'TURN' not in name:
            # Commands are anchored to the initial heading, not to the rotating body.
            yaw0=e.translation_guard.start_yaw;c=yaw0.cos();s=yaw0.sin()
            world=e.root_states[:,7:9]
            velocity=torch.stack((c*world[:,0]+s*world[:,1],-s*world[:,0]+c*world[:,1]),1)
            command=e.teacher_commands[e.teacher_ids,:2];speed=command.norm(dim=1).clamp(min=.01)
            error=(velocity-command).square().sum(1)
            sigma=torch.maximum(speed*.5,torch.full_like(speed,.03))
            progress=(velocity*command).sum(1)/speed.square()
            tracking=4*torch.exp(-error/sigma.square())+2*progress.clamp(-1,1)
            multiplier=cfg.get('multipliers',{}).get('v2_velocity',1.)
            terms['v2_velocity']=torch.where(mask,tracking*multiplier,terms['v2_velocity'])
            q=e.root_states[:,3:7];x,y,z,w=q.unbind(1)
            delta=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))-yaw0
            # Soft heading guidance, never an early-termination boundary.
            heading=-4*(1-delta.cos())*cfg.get('multipliers',{}).get('v2_heading_integral',1.)
            terms['v2_heading_integral']=torch.where(mask,heading,terms['v2_heading_integral'])
    # Preserve the invariant that physical fall exceeds all other same-transition costs.
    costs=sum((-v).clamp(min=0) for key,v in terms.items() if key!='v2_fall_event')*e.dt
    terms['v2_fall_event']=-(20+costs)*fall/e.dt
    return terms
