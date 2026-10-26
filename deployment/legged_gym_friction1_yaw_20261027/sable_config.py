import json
from pathlib import Path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO
HERE=Path(__file__).resolve().parent
MODEL=json.loads((HERE/'assets/physical_manifest.json').read_text())

class SableCfg(LeggedRobotCfg):
    class material_friction:
        plane=0.
        front_scale=1.
        # PhysX arithmetic combine: shape=2*desired coefficient on plane0.
        # Both batch arms use the same recipe. The rear floor coefficient is
        # unchanged from parent:0.5*(1+U[.8,1.2]) during training,1 at inference.
    class termination:
        knee_contact_enabled=True
        knee_contact_margin_m=.0005  # Existing PhysX contact envelope, not a knee-center height target.
    class posture:
        knee_stand_reference_offset_deg=-10.
        knee_stand_tolerance_deg=10.
        knee_stand_scale_deg=15.
        knee_stand_max_exponent=10.
        hip_stand_scale_deg=15.
        hip_preferred_max_deg=-19.
        hip_stand_tolerance_deg=5.
        review_after_seconds=900.
        hip_stand_max_exponent=10.  # Numerical guard beyond the authored physical range.
    class hind_landing_width:
        minimum_m=.06
    class proximal_acceleration:
        threshold_rad_s2=150.
        maximum_normalized_excess=10.
        joint_names=['L_shoulder_roll','R_shoulder_roll','L_shoulder_pitch','R_shoulder_pitch','hip_roll','L_hip_pitch','R_hip_pitch']
    class stride:
        reward_cap_m=.10
        reward_exponent=4.
        hind_zero_m=.08
    class ankle_phase:
        landing_angle_deg=30.
        takeoff_angle_deg=110.
        landing_min_deg=30.
        lag_tolerance_deg=5.
        lag_scale_deg=30.
        lag_max_cost=16.
        landing_shortfall_scale_deg=10.
        landing_shortfall_max_exponent=3.
    class gait:
        enabled=False;period_s=1.;stance_fraction=.5;swing_height_m=.018
        swing_heights_m=[.023,.023,.023,.023]
        minimum_completed_swing_peak_m=.022
        height_profile='sine'
        contact_phase_independent_of_strength=False;clearance_relative_to_target=False
        force_sigma_n=.5;stance_speed_sigma_m_s=.04
    class reference:
        amplitude_multipliers=[4./3.,4./3.,4./3.,4./3.]
        enabled=False;file=str(HERE/'assets/gait_reference.json');ramp_s=1.;speed_filter_tau_s=.15;strength=1.
    class env(LeggedRobotCfg.env):
        num_envs=1024;num_observations=54;num_actions=14;episode_length_s=20.;env_spacing=1.
    class terrain(LeggedRobotCfg.terrain):
        mesh_type='plane';measure_heights=False;curriculum=False
    class commands(LeggedRobotCfg.commands):
        curriculum=False;heading_command=False;resampling_time=5.
        class ranges:
            lin_vel_x=[.05,.20];lin_vel_y=[0.,0.];ang_vel_yaw=[0.,0.];heading=[0.,0.]
    class init_state(LeggedRobotCfg.init_state):
        pos=MODEL['init_position'];rot=MODEL['init_quat_xyzw'];default_joint_angles=MODEL['default_angles']
    class control(LeggedRobotCfg.control):
        stiffness={'spine_pitch':10.,'spine_roll':7.,'hip_roll':7.,'shoulder_roll':7.,'shoulder_pitch':8.,'hip_pitch':8.,'elbow_pitch':5.,'knee_pitch':5.,'ankle_pitch':5.}
        damping={'spine_pitch':.12,'spine_roll':.10,'hip_roll':.06,'shoulder_roll':.06,'shoulder_pitch':.05,'hip_pitch':.05,'elbow_pitch':.025,'knee_pitch':.025,'ankle_pitch':.025}
        action_scale=.35;decimation=40
    class asset(LeggedRobotCfg.asset):
        file=str(HERE/'assets/sable.xml');name='sable';foot_name='_fixed'
        collapse_fixed_joints=False;replace_cylinder_with_capsule=False;flip_visual_attachments=False
        thickness=.0002;armature=.000015;density=1000.;self_collisions=0
        penalize_contacts_on=['elbow_pitch','knee_pitch'];terminate_after_contacts_on=['ROOT','spine_pitch']
    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots=False;randomize_friction=True;friction_range=[.8,1.2];randomize_base_mass=False
    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma=.01;soft_dof_pos_limit=.9;base_height_target=MODEL['init_position'][2]
        class scales:
            action_rate=-0.01
            ang_vel_xy=-0.05
            ankle_landing_low=-2.0
            ankle_phase_angle=-4.0
            ankle_phase_event=-2.0
            ankle_phase_lag=-8.0
            ankle_phase_reverse=-0.5
            base_height=-10.0
            collision=-1.0
            dof_acc=-2.5e-07
            dof_pos_limits=-2.0
            feet_air_time=1.0
            forward_speed=2.0
            gait_clearance=-32.0
            gait_contact=3.0
            hind_landing_width=-4.0
            hind_stride_shortfall=-2.0
            knee_stand_exp=-2.5
            lin_vel_z=-2.0
            orientation=-0.5
            proximal_acceleration_peak=-4.0
            stride_progress=2.0
            swing_peak_balance=-0.5
            swing_peak_shortfall=-2.0
            target_limits=-2.0
            termination=-250.0
            torques=-1e-05
            tracking_ang_vel=1.0
            yaw_rate_error=-2.0
            tracking_lin_vel=8.0
    class normalization(LeggedRobotCfg.normalization):
        clip_actions=3.
    class sim(LeggedRobotCfg.sim):
        dt=.0005;substeps=1
        class physx(LeggedRobotCfg.sim.physx):
            num_threads=4;num_position_iterations=8;num_velocity_iterations=2
            contact_offset=.0005;rest_offset=0.;bounce_threshold_velocity=.1;max_depenetration_velocity=.5
    class noise(LeggedRobotCfg.noise):
        noise_level=.5

class SablePPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std=.8
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name='sable_stand5_v1';run_name='velocity';max_iterations=1000000;save_interval=50
