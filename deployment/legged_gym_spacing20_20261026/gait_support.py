"""Clock-conditioned crawl rewards, with clearance from the authored capsules.

Contact shaping follows the method in Improbable-AI/walk-these-ways CoRLRewards.
Our foot size, crawl order, timing and reward scales are simulation hypotheses.
"""
import xml.etree.ElementTree as ET
import torch,math

class GaitSignals:
    def __init__(self,env):
        endpoints=[];radii=[];offsets=[];tree=ET.parse(env.cfg.asset.file)
        order={'L_wrist_fixed':.75,'R_ankle_pitch':.5,'R_wrist_fixed':.25,'L_ankle_pitch':0.}
        for i in env.feet_indices.cpu().tolist():
            name=env.body_names[i];body=next(b for b in tree.iter('body') if b.get('name')==name)
            shapes=[g for g in body.findall('geom') if g.get('type')=='capsule'];assert len(shapes)==1
            vals=[float(x) for x in shapes[0].get('fromto').split()];endpoints.append([vals[:3],vals[3:]])
            radii.append(float(shapes[0].get('size')));offsets.append(order[name])
        self.endpoints=torch.tensor(endpoints,device=env.device);self.radii=torch.tensor(radii,device=env.device)
        self.offsets=torch.tensor(offsets,device=env.device)
        heights=dict(zip(['L_wrist_fixed','R_wrist_fixed','L_ankle_pitch','R_ankle_pitch'],env.cfg.gait.swing_heights_m))
        self.heights=torch.tensor([heights[env.body_names[i]] for i in env.feet_indices.cpu().tolist()],device=env.device)
        self.amplitude_multipliers=torch.tensor(env.cfg.reference.amplitude_multipliers,device=env.device)

    def phase(self,env):
        return torch.remainder(env.episode_length_buf*env.dt/env.cfg.gait.period_s+env.gait_phase_offset,1.)

    def update(self,env):
        state=env.rigid_states[:,env.feet_indices];n=env.num_envs
        v=self.endpoints.unsqueeze(0).expand(n,-1,-1,-1)
        q=state[:,:,None,3:6].expand(-1,-1,2,-1);w=state[:,:,None,6:7]
        rotated=v+2*torch.cross(q,torch.cross(q,v,dim=-1)+w*v,dim=-1)
        env.foot_center_world=state[:,:,:3]+rotated.mean(2)
        env.foot_clearance=state[:,:,2]+rotated[:,:,:,2].min(2).values-self.radii
        delta=rotated[:,:,1]-rotated[:,:,0]
        env.foot_long_axis_world=delta/delta.norm(dim=-1,keepdim=True).clamp(min=1e-8)
        lowest=rotated[:,:,:,2].argmin(2)
        point=torch.gather(rotated,2,lowest[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
        env.foot_contact_point=state[:,:,:3]+point
        env.foot_contact_point[:,:,2]-=self.radii

        # This is a soft foot-body speed cost, not an exact contact-point slip measurement.
        env.foot_stance_speed_sq=state[:,:,7:9].square().sum(2)
        phase=torch.remainder(self.phase(env)[:,None]+self.offsets,1.)
        progress=((phase-env.cfg.gait.stance_fraction)/(1-env.cfg.gait.stance_fraction)).clamp(0,1)
        reference=getattr(env,'motion_reference',None)
        amplitude=torch.ones(n,device=env.device)
        moving=env.commands[:,0].abs()>.02
        if reference is not None:
            amplitude=(reference.speed/.03).clamp(0,1)*reference.ramp(env.episode_length_buf*env.dt)
            moving=reference.speed>.001
        swing=(phase>=env.cfg.gait.stance_fraction)&moving[:,None]
        contact_activation=amplitude
        if reference is not None and env.cfg.gait.contact_phase_independent_of_strength:
            # Motion amplitude changes lift height, not the desired contact phase.
            strength=env.cfg.reference.strength
            contact_activation=(amplitude/strength).clamp(0,1) if strength>0 else torch.zeros_like(amplitude)
        env.gait_swing_weight=(progress/.15).clamp(0,1)*((1-progress)/.15).clamp(0,1)*swing*contact_activation[:,None]
        height=torch.sin(math.pi*progress)
        if env.cfg.gait.height_profile=='sine_squared':height=height.square()
        multipliers=self.amplitude_multipliers if reference is not None else 1.
        env.gait_target_clearance=height*self.heights[None,:]*swing*amplitude[:,None]*multipliers
