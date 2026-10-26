"""Official LeggedRobot adapter with an optional audited kinematic reference."""
import math,json
import numpy as np
import torch
from isaacgym import gymapi,gymtorch
from isaacgym.torch_utils import quat_rotate_inverse,to_torch,torch_rand_float
from legged_gym.envs.base.legged_robot import LeggedRobot
from sable_config import MODEL

class SableRobot(LeggedRobot):
    def _create_ground_plane(self):
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1)
        plane.static_friction=self.cfg.material_friction.plane;plane.dynamic_friction=self.cfg.material_friction.plane;plane.restitution=0.
        self.gym.add_ground(self.sim,plane)

    def _create_envs(self):
        from pathlib import Path
        p=Path(self.cfg.asset.file);opts=gymapi.AssetOptions()
        opts.default_dof_drive_mode=gymapi.DOF_MODE_EFFORT
        opts.collapse_fixed_joints=False;opts.fix_base_link=self.cfg.asset.fix_base_link
        opts.disable_gravity=self.cfg.asset.disable_gravity;opts.flip_visual_attachments=False
        opts.replace_cylinder_with_capsule=False;opts.armature=.000015;opts.thickness=.0002
        opts.angular_damping=0.;opts.linear_damping=0.;opts.override_com=False;opts.override_inertia=False
        a=self.gym.load_asset(self.sim,str(p.parent),p.name,opts);self.robot_asset=a
        self.dof_names=self.gym.get_asset_dof_names(a);self.body_names=self.gym.get_asset_rigid_body_names(a)
        self.num_dof=self.num_dofs=len(self.dof_names);self.num_bodies=len(self.body_names)
        assert self.num_dof==16,self.dof_names
        tendon_count=self.gym.get_asset_tendon_count(a)
        assert tendon_count==1,('Missing physical hip-roll coupling',tendon_count)
        tendons=self.gym.get_asset_tendon_properties(a)
        for t in tendons:
            t.limit_stiffness=1000.;t.damping=1.
        self.gym.set_asset_tendon_properties(a,tendons)
        self._get_env_origins()
        self.base_init_state=to_torch(self.cfg.init_state.pos+self.cfg.init_state.rot+[0.]*6,device=self.device)
        self.envs=[];self.actor_handles=[];self.shape_diagnostics=[]
        for i in range(self.num_envs):
            env=self.gym.create_env(self.sim,gymapi.Vec3(0,0,0),gymapi.Vec3(0,0,0),int(np.sqrt(self.num_envs)))
            pose=gymapi.Transform();pos=self.base_init_state[:3]+self.env_origins[i]
            pose.p=gymapi.Vec3(*pos.cpu().numpy());pose.r=gymapi.Quat(*self.cfg.init_state.rot)
            shapes=self.gym.get_asset_rigid_shape_properties(a)
            ranges=self.gym.get_asset_rigid_body_shape_indices(a)
            # Shared bit => filtered pair. Different limbs have disjoint bits;
            # torso shares all limb bits. Bit 16 filters wrist-block/floor only.
            friction=np.random.uniform(.8,1.2) if self.cfg.domain_rand.randomize_friction else 1.
            for b,n in enumerate(self.body_names):
                bit=1 if n.startswith('L_shoulder') or n.startswith('L_elbow') or n=='L_wrist_fixed' else 2 if n.startswith('R_shoulder') or n.startswith('R_elbow') or n=='R_wrist_fixed' else 4 if n.startswith('L_') or n=='hip_roll' else 8 if n.startswith('R_') else 15
                for k in range(ranges[b].start,ranges[b].start+ranges[b].count):
                    names=MODEL['collision_shapes_by_body'][n]
                    assert len(names)==ranges[b].count,(n,names,ranges[b].count)
                    shape=MODEL['shape_metadata'][names[k-ranges[b].start]]
                    nofloor=(16<<MODEL['foot_bodies'].index(n)) if n in MODEL['foot_bodies'] and not shape['floor'] else 0
                    shapes[k].filter=bit|nofloor
                    front=shape['foot'] and n in ['L_wrist_fixed','R_wrist_fixed']
                    shapes[k].friction=(1.+friction)*(self.cfg.material_friction.front_scale if front else 1.) if shape['foot'] else 2.
                    shapes[k].restitution=0.;shapes[k].rolling_friction=0.;shapes[k].torsion_friction=0.
            self.gym.set_asset_rigid_shape_properties(a,shapes)
            actor=self.gym.create_actor(env,a,pose,'sable',i,0,0)
            props=self.gym.get_actor_dof_properties(env,actor)
            props['driveMode'][:]=gymapi.DOF_MODE_EFFORT;props['stiffness'][:]=0.;props['damping'][:]=.001
            for j,n in enumerate(self.dof_names):
                spec=MODEL['motors'].get(n)
                props['effort'][j]=spec['stall_torque_nm'] if spec else 0.
                # Do not clamp passive/backdriven speed to a motor nameplate value.
                props['velocity'][j]=1000.
            self._process_dof_props(props,i);self.gym.set_actor_dof_properties(env,actor,props)
            # create_actor initializes collision filters; restore the audited
            # limb/torso filtering afterwards. Materials were set on the asset
            # before creation so their physical coefficients already apply.
            self.gym.set_actor_rigid_shape_properties(env,actor,shapes)
            if i in [0,self.num_envs-1]:
                applied=self.gym.get_actor_rigid_shape_properties(env,actor)
                assert all(abs(x.friction-y.friction)<1e-6 and x.filter==y.filter for x,y in zip(applied,shapes))
            if i==0:
                self.import_mass=sum(x.mass for x in self.gym.get_actor_rigid_body_properties(env,actor))
                self.shape_diagnostics=[{'body':n,'start':ranges[j].start,'count':ranges[j].count} for j,n in enumerate(self.body_names)]
                print('IMPORTED_ASSET',json.dumps({'dofs':self.dof_names,'bodies':self.body_names,'tendons':tendon_count,'mass_kg':self.import_mass,'shapes':self.shape_diagnostics,'material_stage':'before_actor_creation','foot_shape_friction':{n:[shapes[k].friction for k in range(ranges[self.body_names.index(n)].start,ranges[self.body_names.index(n)].start+ranges[self.body_names.index(n)].count)] for n in MODEL['foot_bodies']}}))
            self.envs.append(env);self.actor_handles.append(actor)
        def ids(names):return torch.tensor([self.body_names.index(n) for n in names],dtype=torch.long,device=self.device)
        self.feet_indices=ids(MODEL['foot_bodies'])
        # The knee/elbow motor collision meshes are bound to their proximal hip/shoulder bodies.
        self.penalised_contact_indices=ids(['L_shoulder_pitch','R_shoulder_pitch','L_hip_pitch','R_hip_pitch'])
        self.termination_contact_indices=ids(['ROOT','spine_pitch'])

    def _init_buffers(self):
        # Native PhysX ground adds no actor or body to these tensors.
        self.root_all=gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.root_states=self.root_all.view(self.num_envs,13)
        self.robot_actor_indices=torch.arange(self.num_envs,device=self.device,dtype=torch.int32)
        assert all(self.gym.get_actor_index(e,a,gymapi.DOMAIN_SIM)==i for i,(e,a) in enumerate(zip(self.envs,self.actor_handles)))
        self.dof_state=gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.dof_pos=self.dof_state.view(self.num_envs,self.num_dof,2)[...,0]
        self.dof_vel=self.dof_state.view(self.num_envs,self.num_dof,2)[...,1]
        self.base_quat=self.root_states[:,3:7]
        self.contact_all=gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.contact_forces=self.contact_all.view(self.num_envs,self.num_bodies,3)
        self.gym.refresh_actor_root_state_tensor(self.sim);self.gym.refresh_dof_state_tensor(self.sim);self.gym.refresh_net_contact_force_tensor(self.sim)
        self.common_step_counter=0;self.extras={};self.noise_scale_vec=self._get_noise_scale_vec(self.cfg)
        self.gravity_vec=torch.tensor([0.,0.,-1.],device=self.device).repeat(self.num_envs,1)
        self.forward_vec=torch.tensor([1.,0.,0.],device=self.device).repeat(self.num_envs,1)
        self.torques=torch.zeros(self.num_envs,self.num_dof,device=self.device)
        self.actions=torch.zeros(self.num_envs,self.num_actions,device=self.device);self.last_actions=torch.zeros_like(self.actions)
        self.last_dof_vel=torch.zeros_like(self.dof_vel);self.last_root_vel=torch.zeros_like(self.root_states[:,7:13])
        self.commands=torch.zeros(self.num_envs,4,device=self.device)
        self.commands_scale=torch.tensor([self.obs_scales.lin_vel,self.obs_scales.lin_vel,self.obs_scales.ang_vel],device=self.device)
        self.feet_air_time=torch.zeros(self.num_envs,4,device=self.device);self.last_contacts=torch.zeros_like(self.feet_air_time,dtype=torch.bool)
        self.base_lin_vel=quat_rotate_inverse(self.base_quat,self.root_states[:,7:10]);self.base_ang_vel=quat_rotate_inverse(self.base_quat,self.root_states[:,10:13])
        self.projected_gravity=quat_rotate_inverse(self.base_quat,self.gravity_vec);self.measured_heights=0
        self.default_dof_pos=torch.tensor([self.cfg.init_state.default_joint_angles[n] for n in self.dof_names],device=self.device).unsqueeze(0)
        self.p_gains=torch.zeros(self.num_dof,device=self.device);self.d_gains=torch.zeros_like(self.p_gains)
        for i,n in enumerate(self.dof_names):
            for key,value in self.cfg.control.stiffness.items():
                if key in n:self.p_gains[i]=value;self.d_gains[i]=self.cfg.control.damping[key]
        self.active=torch.tensor([self.dof_names.index(n) for n in MODEL['policy_joint_names']],device=self.device)
        self.dependent=self.dof_names.index('R_hip_roll_mimic');self.hip=self.dof_names.index('hip_roll');self.waist=self.dof_names.index('spine_pitch')
        self.p_gains[self.dependent]=0.;self.d_gains[self.dependent]=0.
        self.hard_limits=torch.tensor([next(r['limits_rad'] for r in self.authored_joints if r['name']==n) for n in self.dof_names],device=self.device) if hasattr(self,'authored_joints') else None
        # Read imported hard limits separately from upstream's soft-limit shrink.
        props=self.gym.get_actor_dof_properties(self.envs[0],self.actor_handles[0])
        self.hard_limits=torch.tensor(np.stack([props['lower'],props['upper']],axis=1),device=self.device)
        for name in ['L_hip_pitch','R_hip_pitch']:
            actual=self.hard_limits[self.dof_names.index(name)]
            expected=torch.tensor([math.radians(-85),math.radians(24)],device=self.device)
            assert torch.allclose(actual,expected,atol=1e-5), (name,actual.cpu().tolist())
        self.max_speed=torch.tensor([MODEL['motors'].get(n,{'no_load_rpm':1e6})['no_load_rpm']*2*math.pi/60 for n in self.dof_names],device=self.device)
        self.large=torch.tensor([MODEL['motors'].get(n,{}).get('model')=='3505' for n in self.dof_names],device=self.device)
        self.reference_gravity=quat_rotate_inverse(self.base_init_state[3:7].unsqueeze(0),torch.tensor([[0.,0.,-1.]],device=self.device))[0]
        self.rigid_all=gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.rigid_states=self.rigid_all.view(self.num_envs,self.num_bodies,13)
        # MJCF spine_pitch body origin and its hinge anchor coincide (joint pos=0).
        self.waist_body_index=self.body_names.index('spine_pitch')
        from waist_acceleration import WaistHeightAcceleration
        self.waist_height_acc=WaistHeightAcceleration(self.num_envs,self.device,self.dt)
        self.hip_pitch_indices=torch.tensor([self.dof_names.index(n) for n in ['L_hip_pitch','R_hip_pitch']],device=self.device)
        self.knee_pitch_indices=torch.tensor([self.dof_names.index(n) for n in ['L_knee_pitch','R_knee_pitch']],device=self.device)
        self.true_fall=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        from knee_contact import KneeGroundContact
        self.knee_contact_detector=KneeGroundContact(self)
        self.knee_clearance=torch.zeros(self.num_envs,2,device=self.device)
        self.knee_ground_contact=torch.zeros(self.num_envs,2,dtype=torch.bool,device=self.device)
        self.last_episode_records=[];self.ep_start_x=torch.zeros(self.num_envs,device=self.device)
        self.ep_velocity_error=torch.zeros(self.num_envs,device=self.device);self.ep_steps=torch.zeros(self.num_envs,device=self.device)
        self.ep_touchdowns=torch.zeros(self.num_envs,4,device=self.device)
        self.ep_last_contact=torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
        self.ep_peak_coupling=torch.zeros(self.num_envs,device=self.device)
        self.evaluation=False;self.reset_noise=.03;self.fixed_command=None
        from gait_support import GaitSignals
        self.gait_signals=GaitSignals(self)
        self.gait_phase_offset=torch.zeros(self.num_envs,device=self.device)
        self.foot_clearance=torch.zeros(self.num_envs,4,device=self.device)
        self.motion_reference=None
        if self.cfg.reference.enabled:
            from guided_reference import GuidedReference
            self.motion_reference=GuidedReference(self)
        from swing_peak_balance import SwingPeakBalance
        self.peak_balance=SwingPeakBalance(self.num_envs,self.device,self.cfg.gait.period_s,self.dt)
        self.swing_peak_balance_cost=torch.zeros(self.num_envs,device=self.device)
        from stride_progress import StrideProgress
        self.stride_progress=StrideProgress(self.num_envs,self.device,self.cfg.gait.period_s,self.dt,self.cfg.stride.reward_cap_m,self.cfg.stride.reward_exponent,self.cfg.stride.hind_zero_m,self.cfg.stride.front_zero_m,self.cfg.stride.variance_normalization_m,self.cfg.stride.variance_max_age_periods)
        from ankle_stance import StanceAnkleMetrics
        self.hind_ankle_indices=torch.tensor([self.dof_names.index(n) for n in ['L_ankle_pitch','R_ankle_pitch']],device=self.device)
        self.hind_hip_bodies=torch.tensor([self.body_names.index(n) for n in ['L_hip_pitch','R_hip_pitch']],device=self.device)
        self.ankle_stance=StanceAnkleMetrics(self.num_envs,self.device,self.dt)
        from ankle_events import AnkleStepEvents
        c=self.cfg.ankle_phase
        self.ankle_events=AnkleStepEvents(self.num_envs,self.device,self.dt,self.cfg.gait.period_s,c.landing_angle_deg,c.takeoff_angle_deg,c.event_clearance_m,c.event_air_time_s,c.event_travel_m,c.event_error_scale_deg,c.event_max_exponent)
        from hind_landing_width import HindLandingWidth
        self.hind_width=HindLandingWidth(self.num_envs,self.device,self.cfg.hind_landing_width.minimum_m)
        self.proximal_indices=torch.tensor([self.dof_names.index(n) for n in self.cfg.proximal_acceleration.joint_names],device=self.device)
        self.hind_foot_angle_deg=torch.zeros(self.num_envs,2,device=self.device)
        self.hind_landing_forward_m=torch.zeros_like(self.hind_foot_angle_deg)


    def step(self,actions):
        if self.motion_reference is not None:self.motion_reference.prepare()
        return super().step(actions)

    def _compute_torques(self,actions):
        target=self.default_dof_pos.repeat(self.num_envs,1)
        target[:,self.active]+=actions*self.cfg.control.action_scale
        if self.motion_reference is not None:target+=self.motion_reference.target()
        target=torch.maximum(torch.minimum(target,self.hard_limits[:,1]),self.hard_limits[:,0])
        wanted=self.p_gains*(target-self.dof_pos)-self.d_gains*self.dof_vel
        v=self.dof_vel;rpm=v.abs()*60/(2*math.pi)
        large_limit=torch.where(rpm<=245.1,1.95-(1.95-.78)*rpm/245.1,.78*(421.3-rpm)/(421.3-245.1)).clamp(min=0)
        small_limit=self.torque_limits*(1-v.abs()/self.max_speed).clamp(min=0)
        driving_limit=torch.where(self.large,large_limit,small_limit)
        limit=torch.where(wanted*v<0,self.torque_limits,driving_limit)
        result=torch.maximum(torch.minimum(wanted,limit),-limit);result[:,self.dependent]=0.
        return result

    def compute_observations(self):
        self.obs_buf=torch.cat([self.base_lin_vel*self.obs_scales.lin_vel,self.base_ang_vel*self.obs_scales.ang_vel,
            self.projected_gravity,self.commands[:,:3]*self.commands_scale,
            (self.dof_pos[:,self.active]-self.default_dof_pos[:,self.active])*self.obs_scales.dof_pos,
            self.dof_vel[:,self.active]*self.obs_scales.dof_vel,self.actions],dim=1)
        if self.cfg.gait.enabled:
            phase=2*math.pi*self.gait_signals.phase(self)
            self.obs_buf=torch.cat([self.obs_buf,torch.sin(phase)[:,None],torch.cos(phase)[:,None]],dim=1)
        if self.motion_reference is not None:
            self.obs_buf=torch.cat([self.obs_buf,(self.motion_reference.speed*self.obs_scales.lin_vel)[:,None],
                self.motion_reference.ramp(self.episode_length_buf*self.dt)[:,None]],dim=1)
        if self.add_noise:self.obs_buf+=(2*torch.rand_like(self.obs_buf)-1)*self.noise_scale_vec

    def _reset_dofs(self,ids):
        self.dof_pos[ids]=self.default_dof_pos
        if self.reset_noise:self.dof_pos[ids[:,None],self.active]+=torch_rand_float(-self.reset_noise,self.reset_noise,(len(ids),len(self.active)),device=self.device)
        self.dof_pos[ids,self.dependent]=-self.dof_pos[ids,self.hip]
        self.dof_vel[ids]=0.
        actor_ids=self.robot_actor_indices[ids]
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(actor_ids),len(ids))

    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state;self.root_states[ids,:3]+=self.env_origins[ids]
        actor_ids=self.robot_actor_indices[ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_all),gymtorch.unwrap_tensor(actor_ids),len(ids))

    def _resample_commands(self,ids):
        if self.fixed_command is not None:self.commands[ids,0]=self.fixed_command
        else:
            self.commands[ids,0]=torch_rand_float(*self.command_ranges['lin_vel_x'],(len(ids),1),device=self.device).flatten()
            self.commands[ids[torch.rand(len(ids),device=self.device)<.1],0]=0.
        self.commands[ids,1:]=0.

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback();self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.waist_height_acc.update(self.rigid_states[:,self.waist_body_index,2])
        self.gait_signals.update(self)
        self.foot_pair_distance=torch.stack([(self.foot_center_world[:,0,:2]-self.foot_center_world[:,1,:2]).norm(dim=1),(self.foot_center_world[:,2,:2]-self.foot_center_world[:,3,:2]).norm(dim=1)],dim=1)
        if self.cfg.gait.enabled and self.motion_reference is not None:
            local_phase=torch.remainder(self.gait_signals.phase(self)[:,None]+self.gait_signals.offsets,1.)
            eligible=(self.motion_reference.speed>.02)&(self.episode_length_buf*self.dt>=self.cfg.reference.ramp_s)
            target=self.gait_signals.heights[None,:].expand(self.num_envs,-1)*self.cfg.reference.strength*self.gait_signals.amplitude_multipliers[None,:]
            self.swing_peak_balance_cost=self.peak_balance.update(self.foot_clearance,local_phase>=self.cfg.gait.stance_fraction,eligible,target)
        else:self.swing_peak_balance_cost.zero_()
        q=self.base_quat;x,y,z,w=q.unbind(1)
        forward=torch.stack([1-2*(y*y+z*z),2*(x*y+w*z)],1);forward=forward/forward.norm(dim=1,keepdim=True).clamp(min=1e-6)
        self.stride_progress.update(self.foot_center_world[:,:,:2],self.contact_forces[:,self.feet_indices,2]>1.,
            self.foot_clearance,forward,self.commands[:,0],self.episode_length_buf*self.dt,self.base_lin_vel[:,0])

        axis=self.foot_long_axis_world[:,2:]
        cosine=(axis[:,:,:2]*forward[:,None,:]).sum(-1).clamp(-1.,1.)
        self.hind_foot_angle_deg=torch.acos(cosine)*(180/math.pi)
        self.hind_landing_forward_m=((self.foot_contact_point[:,2:,:2]-self.rigid_states[:,self.hind_hip_bodies,:2])*forward[:,None,:]).sum(-1)
        hind_contact=(self.contact_forces[:,self.feet_indices[2:],2]>1.)&(self.foot_clearance[:,2:]<=.001)
        self.ankle_stance.update(self.dof_pos[:,self.hind_ankle_indices],hind_contact,self.foot_clearance[:,2:],self.hind_foot_angle_deg,self.commands[:,0],self.base_lin_vel[:,0],self.episode_length_buf*self.dt)
        self.ankle_events.update(hind_contact,self.foot_clearance[:,2:],self.hind_foot_angle_deg,self.foot_center_world[:,2:,:2],self.commands[:,0],self.episode_length_buf*self.dt)
        self.hind_width.update(self.foot_contact_point[:,2:,:2],hind_contact,self.stride_progress.hind_landing,forward,self.commands[:,0],self.episode_length_buf*self.dt)
        self.knee_contact_detector.update()
        self.ep_velocity_error+=(self.base_lin_vel[:,0]-self.commands[:,0])**2;self.ep_steps+=1
        contact=self.contact_forces[:,self.feet_indices,2]>1.
        self.ep_touchdowns+=((~self.ep_last_contact)&contact).float();self.ep_last_contact[:]=contact
        self.ep_peak_coupling=torch.maximum(self.ep_peak_coupling,(self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs())

    def check_termination(self):
        tilt=(self.projected_gravity*self.reference_gravity).sum(1)
        contact=torch.any(torch.linalg.norm(self.contact_forces[:,self.termination_contact_indices],dim=-1)>2,dim=1)
        coupling=(self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs()
        self.true_fall=(tilt<.55)|(self.root_states[:,2]<.055)|contact|(coupling>.08)|((self.dof_pos[:,self.waist]+math.radians(40)).abs()>.6)
        self.true_fall|=~torch.isfinite(self.root_states).all(1)|~torch.isfinite(self.dof_pos).all(1)
        if self.cfg.termination.knee_contact_enabled:self.true_fall |= self.knee_ground_contact.any(1)
        self.time_out_buf=(self.episode_length_buf>=self.max_episode_length)&~self.true_fall
        self.reset_buf=self.true_fall|self.time_out_buf
        # Upstream reset_idx publishes this only on a reset. With synchronized,
        # stable episodes that leaves old True flags for hundreds of steps and
        # PPO incorrectly adds timeout bootstrapping to every subsequent reward.
        self.extras['time_outs']=self.time_out_buf
        if getattr(self,'capture_eval',False):
            # Snapshot before upstream reset_idx replaces terminal states.
            self.eval_snapshot={
                'waist_anchor_height_m':self.rigid_states[:,self.waist_body_index,2].clone(),
                'waist_vertical_acceleration_m_s2':self.waist_height_acc.acceleration.clone(),
                'proximal_acceleration':((self.dof_vel[:,self.proximal_indices]-self.last_dof_vel[:,self.proximal_indices])/self.dt).clone(),
                'hind_landing_width':self.hind_width.width.clone(),
                'hind_landing_width_valid':self.hind_width.valid.clone(),
                'ankle_event_landed':self.ankle_events.landed.clone(),
                'ankle_event_lifted':self.ankle_events.lifted.clone(),
                'ankle_event_landing_angle_deg':self.ankle_events.landing_angle_deg.clone(),
                'ankle_event_takeoff_angle_deg':self.ankle_events.takeoff_angle_deg.clone(),
                'ankle_event_landing_time_s':self.ankle_events.landing_time_s.clone(),
                'ankle_event_takeoff_time_s':self.ankle_events.takeoff_time_s.clone(),
                'ankle_event_confirmation_time_s':self.ankle_events.confirmation_time_s.clone(),
                'ankle_event_landing_id':self.ankle_events.landing_id.clone(),
                'ankle_event_takeoff_pair_id':self.ankle_events.takeoff_pair_id.clone(),
                'ankle_event_contact':self.ankle_events.contact.clone(),
                'hind_foot_angle_deg':self.hind_foot_angle_deg.clone(),
                'hind_stance_contact':self.ankle_stance.active.clone(),
                'hind_landed':self.ankle_stance.landed.clone(),
                'hind_landing_forward_m':self.hind_landing_forward_m.clone(),
                'ankle_stance_excursion_deg':((self.ankle_stance.hi-self.ankle_stance.lo)*180/math.pi).clone(),
                'xyz':self.root_states[:,:3].clone(),
                'velocity_local':self.base_lin_vel.clone(),
                'angular_velocity_local':self.base_ang_vel.clone(),
                'commands':self.commands.clone(),
                'fall':self.true_fall.clone(),
                'knee_ground_contact':self.knee_ground_contact.clone(),
                'knee_clearance':self.knee_clearance.clone(),
                'foot_centers':self.foot_center_world.clone(),
                'foot_pair_distance':self.foot_pair_distance.clone(),
                'stride_takeoff_age':self.stride_progress.takeoff_age.clone(),
                'stride_takeoff_xy':self.stride_progress.start.clone(),
                'feet':self.rigid_states[:,self.feet_indices,:3].clone(),
                'body_xyz':self.rigid_states[:,:,:3].clone(),
                'body_force_norm':self.contact_forces.norm(dim=-1).clone(),
                'force':self.contact_forces[:,self.feet_indices,2].clone(),
                'q':self.dof_pos.clone(),
                'clearance':self.foot_clearance.clone(),
                'base_quat':self.base_quat.clone(),
                'stride_qualified':(self.stride_progress.qualified&~self.true_fall[:,None]).clone(),
                'stride_signed_m':self.stride_progress.last_signed.clone(),
                'all_stride_landing':self.stride_progress.landing.clone(),
                'stride_variance_event':self.stride_progress.variance_event.clone(),
                'stride_variance_m2':self.stride_progress.variance_m2.clone(),
                'stride_completed_m':self.stride_progress.completed_stride.clone(),
                'hind_stride_landing':self.stride_progress.hind_landing.clone(),
                'swing_completed':self.peak_balance.completed.clone(),
                'swing_peak':self.peak_balance.last_peak.clone(),
                'swing_paired':self.peak_balance.paired.clone(),
                'swing_peak_difference':self.peak_balance.difference.clone(),
                'coupling':(self.dof_pos[:,self.hip]+self.dof_pos[:,self.dependent]).abs().clone()}
    def _reward_target_limits(self):
        # Penalize only requests outside the current imported physical limits.
        # This lets PPO distinguish actions that otherwise clip to the same target.
        target=self.default_dof_pos.repeat(self.num_envs,1)
        target[:,self.active]+=self.actions*self.cfg.control.action_scale
        if self.motion_reference is not None:target+=self.motion_reference.end
        excess=(self.hard_limits[:,0]-target).clamp(min=0)+(target-self.hard_limits[:,1]).clamp(min=0)
        return (excess[:,self.active]/self.cfg.control.action_scale).square().sum(1)

    def _reward_waist_vertical_acceleration(self):
        return self.waist_height_acc.cost(self.cfg.waist_vertical_acceleration.normalization_m_s2)

    def _reward_ankle_event_angle(self):return self.ankle_events.event_cost
    def _reward_stride_shortfall(self):return self.stride_progress.shortfall



    def _reward_stride_variance(self):return self.stride_progress.variance_cost

    def _reward_forward_speed(self):
        # Positive body-forward velocity, capped at the declared course top speed.
        # Suppress parking, lateral slides, yaw spins and terminal falls.
        forward=self.base_lin_vel[:,0].clamp(0.,.12)/.12
        straight=torch.exp(-self.base_lin_vel[:,1].square()/.0025-self.base_ang_vel[:,2].square()/.25)
        return forward*straight*(self.commands[:,0]>.02)*(~self.true_fall)


    def _reward_hind_landing_width(self):return self.hind_width.cost

    def _reward_proximal_acceleration_peak(self):
        from peak_acceleration import peak_cost
        acceleration=(self.dof_vel[:,self.proximal_indices]-self.last_dof_vel[:,self.proximal_indices])/self.dt
        return peak_cost(acceleration,self.cfg.proximal_acceleration.threshold_rad_s2,self.cfg.proximal_acceleration.maximum_normalized_excess)*(self.episode_length_buf>=2)

    def _reward_stride_progress(self):return self.stride_progress.reward*(~self.true_fall)

    def _reward_swing_peak_shortfall(self):
        target=self.cfg.gait.minimum_completed_swing_peak_m
        shortfall=((target-self.peak_balance.last_peak).clamp(min=0)/target).square()
        # Four completed swing events per period; scale as a per-second cost.
        return (shortfall*self.peak_balance.completed).sum(1)*self.cfg.gait.period_s/(4*self.dt)

    def compute_reward(self):
        # Preserve the official aggregation order except posture: like death,
        # its negative cost survives the legacy positive-only reward clipping.
        self.rew_buf[:] = 0.
        posture=torch.zeros_like(self.rew_buf)
        diagnostic = {} if getattr(self,'capture_eval',False) else None
        unclipped_gait = getattr(getattr(self.cfg,'migration',None),'unclipped_gait_costs',False)
        for name,function in zip(self.reward_names,self.reward_functions):
            value=function()*self.reward_scales[name]
            if diagnostic is not None:diagnostic[name]=value.clone()
            if (name in ('velocity_error','waist_vertical_acceleration','ankle_event_angle','yaw_rate_error','stride_shortfall','stride_variance','hind_landing_width','proximal_acceleration_peak','swing_peak_balance','swing_peak_shortfall')
                    or (unclipped_gait and name in ('gait_clearance','target_limits'))):posture+=value
            else:self.rew_buf+=value
            self.episode_sums[name]+=value
        if self.cfg.rewards.only_positive_rewards:
            if diagnostic is not None:diagnostic['positive_clip_adjustment']=(-self.rew_buf).clamp(min=0.)
            self.rew_buf.clamp_(min=0.)
        self.rew_buf+=posture
        if 'termination' in self.reward_scales:
            value=self._reward_termination()*self.reward_scales['termination']
            self.rew_buf+=value;self.episode_sums['termination']+=value
            if diagnostic is not None:diagnostic['termination']=value.clone()
        if diagnostic is not None:
            diagnostic['total']=self.rew_buf.clone()
            self.eval_snapshot['weighted_rewards']=diagnostic

    def _reward_swing_peak_balance(self):return self.swing_peak_balance_cost

    def _reward_termination(self):return self.true_fall.float()
    def _reward_orientation(self):return ((self.projected_gravity-self.reference_gravity)**2).sum(1)
    def _reward_yaw_rate_error(self):
        # Follow the heading controller, including intentional correction turns.
        return (self.base_ang_vel[:,2]-self.commands[:,2]).square()

    def _reward_tracking_ang_vel(self):
        # Angular error is in rad/s; the small linear-speed sigma must not be
        # reused as an unrealistically tight yaw tolerance for this robot.
        return torch.exp(-(self.commands[:,2]-self.base_ang_vel[:,2])**2/.25)
    def _reward_feet_air_time(self):
        contact=self.contact_forces[:,self.feet_indices,2]>1.
        filtered=contact|self.last_contacts;self.last_contacts[:]=contact
        first=(self.feet_air_time>0.)&filtered;self.feet_air_time+=self.dt
        reward=((self.feet_air_time-getattr(self.cfg.rewards,'foot_air_time_target',.20))*first).sum(1)*(self.commands[:,0]>.02)
        self.feet_air_time*=~filtered
        return reward

    def _reward_gait_contact(self):
        force=self.contact_forces[:,self.feet_indices,2].clamp(min=0)
        swing_ok=torch.exp(-force.square()/self.cfg.gait.force_sigma_n**2)
        stance_ok=torch.exp(-self.foot_stance_speed_sq/self.cfg.gait.stance_speed_sigma_m_s**2)
        weight=self.gait_swing_weight
        return (weight*swing_ok+(1-weight)*stance_ok).mean(1)

    def _reward_gait_clearance(self):
        scale=self.gait_signals.heights
        if self.cfg.gait.clearance_relative_to_target and self.cfg.reference.enabled and self.cfg.reference.strength>0:
            scale=scale*self.cfg.reference.strength*self.gait_signals.amplitude_multipliers
        error=(self.foot_clearance-self.gait_target_clearance)/scale
        return error.square().clamp(max=9.).mean(1)

    def reset_idx(self,ids):
        if len(ids) and hasattr(self,'ep_start_x'):
            steps=self.ep_steps[ids].clamp(min=1)
            rows=torch.stack([self.episode_length_buf[ids]*self.dt,self.true_fall[ids].float(),
                self.root_states[ids,0]-self.ep_start_x[ids],torch.sqrt(self.ep_velocity_error[ids]/steps),
                self.ep_peak_coupling[ids],self.ep_touchdowns[ids].min(1).values,self.knee_ground_contact[ids].any(1).float()],1).detach().cpu().numpy()
            self.last_episode_records.extend(rows.tolist());self.last_episode_records=self.last_episode_records[-4096:]
        super().reset_idx(ids)
        if hasattr(self,'ankle_stance'):self.ankle_stance.reset(ids)
        if hasattr(self,'ankle_events'):self.ankle_events.reset(ids)
        if hasattr(self,'waist_height_acc'):self.waist_height_acc.reset(ids)
        if hasattr(self,'hind_width'):self.hind_width.reset(ids)
        if hasattr(self,'stride_progress'):self.stride_progress.reset(ids)
        if hasattr(self,'peak_balance'):
            self.peak_balance.reset(ids);self.swing_peak_balance_cost[ids]=0.
        if self.motion_reference is not None:self.motion_reference.reset(ids)
        if len(ids) and hasattr(self,'gait_phase_offset'):
            self.gait_phase_offset[ids]=torch.rand(len(ids),device=self.device) if self.cfg.gait.enabled else 0.
        if len(ids) and hasattr(self,'ep_start_x'):
            self.ep_start_x[ids]=self.root_states[ids,0];self.ep_velocity_error[ids]=0.;self.ep_steps[ids]=0.
            self.ep_touchdowns[ids]=0.;self.ep_last_contact[ids]=False;self.ep_peak_coupling[ids]=0.
