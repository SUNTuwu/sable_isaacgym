"""CPU-only reward and exported-asset integration checks. Never imports Isaac Gym."""
import json,math,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from teacher_reference_rewards_v2 import ReferenceRewards,check_env,validate_config,quat_apply
from model_mapping import ModelMapping
from action_schema import ACTION_NAMES

def quaternion(m):
    w=math.sqrt(1+float(np.trace(m)))/2
    return np.array([(m[2,1]-m[1,2])/(4*w),(m[0,2]-m[2,0])/(4*w),(m[1,0]-m[0,1])/(4*w),w])

def tensor(x):return torch.tensor(np.asarray(x).tolist(),dtype=torch.float32)

def engine(dt=.02,symmetric=False):
    return ReferenceRewards(1,'cpu',dt,torch.ones(1),torch.ones(1,4)*.4,torch.tensor([symmetric]))

def state():
    z=lambda *shape:torch.zeros(*shape)
    return dict(normal_force=torch.ones(1,4)*2,clearance=z(1,4),phase=z(1),command=tensor([[.075,0,0]]),
        velocity=tensor([[.075,0,0]]),foot_local=z(1,4,3),reference_local=z(1,4,3),foot_sigma=torch.ones(1,3)*.02,
        velocity_sigma=torch.ones(1)*.03,yaw_sigma=torch.ones(1)*.1,target=z(1,15),contact_probability=torch.ones(1,4),
        contact_velocity=z(1,4,3),gravity=tensor([[0,0,-1]]),reference_gravity=tensor([[0,0,-1]]),height_error=z(1),
        limit_excess=z(1,15),joint_acceleration=z(1,15),normalized_torque=z(1,15),fall=torch.tensor([False]))

class RewardsTest(unittest.TestCase):
    def test_perfect_tracking_and_error(self):
        e=engine();s=state();r=e.update(s)
        self.assertAlmostEqual(float(sum(r.values())),9.5,places=5)
        s['foot_local']+=.02;r=e.update(s);self.assertLess(float(r['v2_foot_local']),1.)
        s['velocity'][:,1]=.06;r=e.update(s);self.assertAlmostEqual(float(r['v2_cross_motion']),-.16,places=6)
    def test_gravity_alignment_separate_from_height(self):
        e=engine();s=state();r=e.update(s)
        self.assertEqual(float(r['v2_gravity_alignment']),0.)
        s['height_error'][:]=.03;r=e.update(s)
        self.assertEqual(float(r['v2_gravity_alignment']),0.)
        self.assertAlmostEqual(float(r['v2_body_height']),-.5,places=5)
        angle=math.radians(15);s['gravity']=tensor([[0,math.sin(angle),-math.cos(angle)]])
        r=e.update(s);self.assertAlmostEqual(float(r['v2_gravity_alignment']),-4.5,places=4)
        self.assertNotIn('v2_posture',r)
    def test_fall_dominates_other_costs(self):
        for dt in (.01,.02,.04):
            e=engine(dt);s=state();s['gravity']=-s['reference_gravity'];s['fall'][:]=True
            r=e.update(s);other=sum(float((-v).clamp(min=0))*dt for k,v in r.items() if k!='v2_fall_event')
            self.assertAlmostEqual(-float(r['v2_fall_event'])*dt,20.+other,places=3)
            self.assertGreater(-float(r['v2_fall_event'])*dt,other)
    def test_contact_reference_phase_mismatch(self):
        e=engine();s=state();r=e.update(s);self.assertEqual(float(r['v2_contact_timing']),0.)
        s['contact_probability'][:,0]=0.;r=e.update(s);self.assertAlmostEqual(float(r['v2_contact_timing']),-.5)
        self.assertAlmostEqual(float(e.diagnostics['contact_phase_agreement']),.75)
        s['contact_probability'][:]=0.;r=e.update(s);self.assertAlmostEqual(float(r['v2_contact_timing']),-2.)
    def test_turn_translation_cost_restored(self):
        e=engine();s=state();s['command']=tensor([[0,0,.087266]])
        s['velocity']=tensor([[.03,0,.087266]]);r=e.update(s);self.assertEqual(float(r['v2_cross_motion']),0.)
        s['velocity']=tensor([[0,.06,.087266]]);r=e.update(s);self.assertEqual(float(r['v2_cross_motion']),0.)
        s['velocity']=tensor([[0,0,.087266]]);r=e.update(s);self.assertEqual(float(r['v2_cross_motion']),0.)
    def test_translation_yaw_penalty(self):
        e=engine();s=state();s['velocity'][:,2]=.1;r=e.update(s);self.assertAlmostEqual(float(r['v2_translation_yaw_rate']),-2.,places=5)
        s['velocity'][:,2]=-.2;r=e.update(s);self.assertAlmostEqual(float(r['v2_translation_yaw_rate']),-8.,places=5)
        s['command']=tensor([[0,0,.087266]]);r=e.update(s);self.assertEqual(float(r['v2_translation_yaw_rate']),0.)
    def test_task_boundaries(self):
        from teacher_reference_rewards_v2 import boundary_errors
        root=torch.zeros(2,13);root[:,6]=1.;root[0,5]=math.sin(math.radians(10.1)/2);root[0,6]=math.cos(math.radians(10.1)/2);root[1,0]=.301
        command=tensor([[.075,0,0],[0,0,.087]])
        hf,tf,*_=boundary_errors(root,torch.zeros(2),torch.zeros(2,2),command,torch.ones(2,dtype=torch.bool))
        self.assertEqual(hf.tolist(),[True,False]);self.assertEqual(tf.tolist(),[False,True])
        hf,tf,*_=boundary_errors(root,tensor([math.radians(10.1),0]),root[:,:2],command,torch.ones(2,dtype=torch.bool))
        self.assertFalse(bool((hf|tf).any()))
        e=engine();s=state();s['task_failure']=torch.tensor([True]);r=e.update(s)
        self.assertAlmostEqual(float(r['v2_task_boundary_event'])*e.dt,-10.);self.assertEqual(float(r['v2_fall_event']),0.)
    def test_reset_no_fake_liftoff(self):
        e=engine();s=state();s['normal_force']*=0;s['clearance']+=.02
        for _ in range(10):self.assertEqual(float(e.update(s)['v2_liftoff_events']),0.)
        s['normal_force']+=2;s['clearance']*=0
        self.assertEqual(float(e.update(s)['v2_liftoff_events']),0.)
    def test_contact_hysteresis(self):
        e=engine();s=state();e.update(s)
        s['normal_force'][:]=.5;s['clearance'][:]=.002
        self.assertEqual(float(e.update(s)['v2_liftoff_events']),0.)
        s['normal_force'][:,0]=0;r=e.update(s)
        self.assertAlmostEqual(float(r['v2_liftoff_events'])*e.dt,-.01,places=6)
        e.reset(torch.tensor([0]));self.assertEqual(float(e.update(s)['v2_liftoff_events']),0.)
    def test_event_dt_and_turn_gate(self):
        for dt in (.01,.02,.04):
            e=engine(dt);s=state();s['command']=tensor([[0,0,.8]]);e.update(s)
            s['normal_force'][:,0]=0;s['clearance'][:,0]=.02
            self.assertAlmostEqual(float(e.update(s)['v2_liftoff_events'])*dt,-.01,places=6)
            self.assertEqual(float(e.yaw_integral),0.)
    def test_short_swing_and_normal_cadence(self):
        e=engine();s=state();e.update(s)
        s['phase'][:]=.9;e.update(s);s['phase'][:]=.0;e.update(s)
        s['normal_force'][:,0]=0;s['clearance'][:,0]=.02;e.update(s)
        for _ in range(15):e.update(s)
        s['normal_force'][:,0]=2;s['clearance'][:,0]=0
        self.assertEqual(float(e.update(s)['v2_liftoff_events']),0.)
        s['normal_force'][:,0]=0;s['clearance'][:,0]=.02
        self.assertAlmostEqual(float(e.update(s)['v2_liftoff_events'])*e.dt,-.05,places=6)
        s['normal_force'][:,0]=2;s['clearance'][:,0]=0
        self.assertLess(float(e.update(s)['v2_liftoff_events']),0.)
    def test_complete_target_smoothness(self):
        e=engine();s=state();e.update(s);e.update(s)
        s['target']+=.2
        self.assertLess(float(e.update(s)['v2_target_acceleration']),-1.)
    def test_phase_symmetry(self):
        e=engine(symmetric=True);s=state();s['foot_local'][:,:,1]=tensor([[.04,-.04,.03,-.03]])
        for _ in range(30):r=e.update(s)
        self.assertEqual(float(r['v2_phase_symmetry']),0.)
        s['foot_local'][:,0,0]=.05
        self.assertLess(float(e.update(s)['v2_phase_symmetry']),0.)
        e.reset(torch.tensor([0]));self.assertEqual(float(e.update(s)['v2_phase_symmetry']),0.)
    def test_exclusive_config(self):
        base={'control_mode':'reference_plus_residual'};validate_config(base)
        for option in ('motion_reward_module_enabled','adaptive_enabled','translation_guard_enabled','pace_stability_enabled'):
            with self.assertRaises(ValueError):validate_config(dict(base,**{option:True}))
    def test_flat_reward_is_not_acceptance(self):
        from teacher_reference_evaluation_v2 import summarize
        report={'mean_command':[.075,0,0],'survival_fraction':1.,'xy_rmse_m_s':.075,'yaw_rmse_rad_s':0.}
        history=[]
        for _ in range(5):result=summarize(report,[0]*4,100,[4]*4,0,100,history)
        self.assertFalse(result['accepted']);self.assertEqual(result['plateau_status'],'stalled_not_solved')
    def test_export_in_simulator_fk_and_rotated_world(self):
        reference=BASE.parent/'reference';manifest=json.loads((reference/'manifest.json').read_text(encoding='utf8'))
        model=ModelMapping(BASE/'assets/sable.xml',BASE/'assets/physical_manifest.json')
        names=list(ACTION_NAMES)+['R_hip_roll_mimic','L_wrist_fixed','R_wrist_fixed']
        joints=[j[0] for node in model.nodes for j in node[4]]
        self.assertTrue(set(joints)<=set(names))
        max_error=0.
        for clip in manifest['clips']:
            d=np.load(reference/(clip['id']+'.npz'));self.assertTrue(np.isfinite(d['joint_position_policy_order']).all())
            self.assertLess(float(abs(d['joint_position_policy_order'][0]-d['joint_position_policy_order'][-1]).max()),1e-7)
            self.assertLess(float(abs(d['joint_velocity_policy_order'][0]-d['joint_velocity_policy_order'][-1]).max()),1e-7)
            for k in (0,25,50,75,100):
                root=d['root_transform_world'][k];q=torch.zeros(1,len(names));q[0,:15]=tensor(d['joint_position_policy_order'][k]);q[0,15]=-q[0,1]
                p,r=model.forward(tensor(root[:3,3][None]),tensor(quaternion(root[:3,:3])[None]),q,names)
                toes=[]
                for index,pts,radius in model.capsules:
                    distal=int(pts.square().sum(-1).argmax());axis=pts[distal]-pts[1-distal]
                    toes.append(p[:,index]+quat_apply(r[:,index],(pts[distal]+radius*axis/axis.norm())[None]))
                toes=torch.stack(toes,1)
                error=float((toes-tensor(d['toe_actual_world'][k][None])).abs().max());max_error=max(error,max_error)
                self.assertLess(error,2e-5,(clip['id'],k,error))
                anchor_ids=torch.tensor([[model.body_names.index(n) for n in pair] for pair in [('L_shoulder_pitch','R_shoulder_pitch'),('L_hip_pitch','R_hip_pitch')]])
                frame_ids=torch.tensor([model.body_names.index('ROOT'),model.body_names.index('spine_roll')])
                from teacher_reference_rewards_v2 import quat_inverse_apply
                local=quat_inverse_apply(r[:,frame_ids].repeat_interleave(2,1),toes-p[:,anchor_ids].mean(2).repeat_interleave(2,1))
                local=torch.einsum('nfi,fij->nfj',local,tensor(d['anatomy_frame_calibration']).repeat_interleave(2,0))
                self.assertLess(float((local-tensor(d['toe_local_anatomy'][k][None])).abs().max()),2e-5)
                # Rigidly change global heading and origin: local reference must not move.
                yaw=tensor([[0,0,math.sin(.7),math.cos(.7)]]);offset=tensor([[5,-3,0]])
                transformed_toe=quat_apply(yaw[:,None].expand(-1,4,-1),toes)+offset[:,None]
                anchors=p[:,anchor_ids].mean(2).repeat_interleave(2,1)
                transformed_anchor=quat_apply(yaw[:,None].expand(-1,4,-1),anchors)+offset[:,None]
                from model_mapping import quat_multiply
                transformed_rot=quat_multiply(yaw[:,None].expand(-1,4,-1),r[:,frame_ids].repeat_interleave(2,1))
                actual=quat_inverse_apply(transformed_rot,transformed_toe-transformed_anchor)
                actual=torch.einsum('nfi,fij->nfj',actual,tensor(d['anatomy_frame_calibration']).repeat_interleave(2,0))
                self.assertLess(float((actual-local).abs().max()),2e-6)
                if k==0:
                    # Execute the complete adapter, including reward/termination
                    # snapshots, against native asset FK without starting physics.
                    rigid=torch.zeros(1,len(model.nodes),13);rigid[:,:,:3]=p;rigid[:,:,3:7]=r
                    root_state=torch.zeros(1,13);root_state[:,:3]=tensor(root[:3,3][None]);root_state[:,3:7]=tensor(quaternion(root[:3,:3])[None])
                    sample={'root':tensor(root[None]),'quat':root_state[:,3:7],
                        'q':q[:,:15],'anatomy_toe':tensor(d['toe_local_anatomy'][k][None]),
                        'contact_probability':tensor(d['reference_contact_probability'][k][None])}
                    e=NS(num_envs=1,device='cpu',ref_sample=lambda:sample,root_states=root_state,dof_pos=q,dof_vel=torch.zeros_like(q),
                        hip=1,dependent=15,projected_gravity=quat_inverse_apply(root_state[:,3:7],tensor([[0,0,-1]])),gravity_vec=tensor([[0,0,-1]]),
                        contact_forces=torch.zeros(1,len(model.nodes),3),termination_contact_indices=torch.tensor([0]),
                        knee_ground_contact=torch.zeros(1,2,dtype=torch.bool),cfg=NS(termination=NS(knee_contact_enabled=False)),
                        env_origins=torch.zeros(1,3),episode_length_buf=torch.tensor([1]),max_episode_length=1200,extras={},
                        _proprio=lambda:torch.zeros(1,51),base_lin_vel=torch.zeros(1,3),_critic=lambda _:torch.zeros(1,92),
                        mapping=model,rigid_states=rigid,feet_indices=torch.tensor([x[0] for x in model.capsules]),
                        event_anchor_indices=anchor_ids,event_frame_indices=frame_ids,anatomy_calibration=tensor(d['anatomy_frame_calibration']),
                        v2_foot_com=torch.zeros(4,3),foot_contact_point=toes,active=torch.arange(15),hard_limits=torch.tensor([[-4.,4.]]*len(names)),
                        executed_target=q[:,:15],current_reference_target=q[:,:15],current_offset_rad=torch.zeros(1,15),
                        current_residual_action=torch.zeros(1,15),teacher_commands=tensor(d['command'][None]),teacher_ids=torch.tensor([0]),
                        heading_linear_velocity=tensor(d['command'][None,:2]),heading_yaw_rate=tensor(d['command'][None,2]),
                        foot_clearance=tensor(d['foot_capsule_min_z'][k][None]),ref_phase=torch.zeros(1),v2_sigma=torch.ones(1,3)*.02,
                        last_dof_vel=torch.zeros_like(q),dt=.02,torques=torch.zeros_like(q),torque_limits=torch.ones(len(names)),
                        translation_guard=NS(start_yaw=torch.zeros(1),start_xy=root_state[:,:2].clone(),ready=torch.ones(1,dtype=torch.bool)),reference_rewards_v2=engine(),teacher_observations=lambda:torch.zeros(1,94),training_env_count=0)
                    check_env(e)
                    self.assertFalse(bool(e.true_fall.any()))
                    self.assertLess(float(e.reference_endpoint_error_m.max()),2e-5)
                    self.assertTrue(torch.isfinite(e.teacher_reward).all())
        print('asset FK maximum foot coordinate error (m):',max_error)

if __name__=='__main__':unittest.main(verbosity=2)
