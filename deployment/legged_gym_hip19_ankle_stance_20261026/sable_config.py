import json
from pathlib import Path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO
HERE=Path(__file__).resolve().parent
MODEL=json.loads((HERE/'assets/physical_manifest.json').read_text())

class SableCfg(LeggedRobotCfg):
    class material_friction:
        plane=0.
        front_scale=.02
        # PhysX arithmetic combine: shape=2*desired coefficient on plane0.
        # Both batch arms use the same recipe. The rear floor coefficient is
        # unchanged from parent:0.5*(1+U[.8,1.2]) during training,1 at inference.
    class termination:
        knee_contact_enabled=True
        knee_contact_margin_m=.0005  # Existing PhysX contact envelope, not a knee-center height target.
    class posture:
        hip_stand_scale_deg=15.
        hip_preferred_max_deg=-19.
        hip_stand_tolerance_deg=5.
        review_after_seconds=900.
        hip_stand_max_exponent=10.  # Numerical guard beyond the authored physical range.
    class ankle_stance:
        excursion_target_deg=15.
        foot_angle_free_max_deg=89.
        foot_angle_error_scale_deg=30.
    class gait:
        enabled=False;period_s=.5;stance_fraction=.75;swing_height_m=.018
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
        class scales(LeggedRobotCfg.rewards.scales):
            orientation=-.5;base_height=-10.;dof_pos_limits=-2.;termination=-250.;target_limits=-2.;forward_speed=2.;hind_foot_angle=-4.;ankle_stance_motion=2.;ankle_stance_reversal=-.5;hip_stand_exp=-2.5;stride_progress=2.;swing_peak_balance=-.5;swing_peak_shortfall=-2.;
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
