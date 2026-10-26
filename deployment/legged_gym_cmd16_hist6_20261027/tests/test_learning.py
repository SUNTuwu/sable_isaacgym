import sys
import copy
import json
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from command_schema import CommandSpec
from observation_schema import pack
from history_policy import HistoryActorCritic
from ppo_history import HistoryPPO, timeout_bootstrap
from step_events import StepEvents
from acceptance import assess,assess_velocity_transitions
from evaluation_scenarios import scenarios,scheduled_commands,transition_segments
from curriculum import next_frequency_range


class LearningTests(unittest.TestCase):
    def test_frequency_expansion_requires_completed_episodes(self):
        self.assertEqual(next_frequency_range('S5',[.9,1.1],0,1.,600),[.9,1.1])
        self.assertEqual(next_frequency_range('S5',[.9,1.1],300,.89,600),[.9,1.1])
        self.assertEqual(next_frequency_range('S5',[.9,1.1],300,1.,60),[.9,1.1])
        self.assertEqual(next_frequency_range('S3',[.9,1.1],300,1.,600),[.9,1.1])
        self.assertEqual(next_frequency_range('S5',[.9,1.1],300,.95,600),[.8,1.2])
        self.assertEqual(next_frequency_range('S5',[.9,1.1],300,.95,600,1024),[.9,1.1])

    def test_acceptance_rejects_missing_events_and_nonresponsive_clocks(self):
        baseline = {'teacher_sha256':'t'*64,'regression_limits':{'xy_rmse_max_m_s':.053,'forward_rmse_max_m_s':.03},
            'new_capability_limits':{'evaluation_seeds':[947,1147,1347],'environments_per_seed':32,
                'xy_rmse_m_s':.035,'yaw_rmse_rad_s':.3,'mean_command_gain_min':.5,
                'frequency_error_hz':.1,'frequency_switch_difference_min_hz':.25}}
        reports = []
        for seed in baseline['new_capability_limits']['evaluation_seeds']:
            report = {'seed':seed,'teacher_control':False,'teacher_sha256':'t'*64,'checkpoint_sha256':'c'*64,
                'reference_execution_calls':0,'num_envs':60,'horizon_s':28,'suite':'S5','scenarios':[], 'frequency_scenarios':[]}
            for name, command in scenarios('S5'):
                report['scenarios'].append({'name':name,'survival_fraction':1.,'xy_rmse_m_s':.01,
                    'yaw_rmse_rad_s':.1,'mean_command':command[:3],'mean_actual_xyz_command_units':command[:3],
                    'four_feet_stepping_fraction':1.,'steady_frequency_per_foot_hz':[command[3]]*4})
            for name, segment, f in [('frequency08',0,.8),('frequency12',0,1.2),('frequency_switch',0,.8),('frequency_switch',8,1.2),('frequency_switch',20,.8)]:
                report['frequency_scenarios'].append({'name':name,'segment_start_s':segment,
                    'feet':[{'foot':i,'measured_frequency_hz':f,'interval_count':10,'environments':4} for i in range(4)]})
            report['velocity_segments']=[]
            commands=dict(scenarios('S5'))
            for name,times in transition_segments('S5').items():
                for start in times:
                    value,_=scheduled_commands(name,commands[name],start)
                    report['velocity_segments'].append({'name':name,'segment_start_s':start,'samples':20,
                        'mean_command':value[:3],'mean_actual':value[:3], 'xy_rmse_m_s':.01,'yaw_rmse_rad_s':.1})
            reports.append(report)
        self.assertTrue(assess('S5',reports,baseline)['passed'])
        broken = copy.deepcopy(reports)
        broken[0]['frequency_scenarios'][-1]['feet'].pop()
        self.assertFalse(assess('S5',broken,baseline)['passed'])
        broken=copy.deepcopy(reports)
        for segment in broken[0]['velocity_segments']:
            if segment['name']=='speed_switch':segment['mean_actual']=[.105,0.,0.]
        self.assertTrue(any('insufficient_change' in f for f in assess('S5',broken,baseline)['failures']))
        broken = copy.deepcopy(reports)
        for report in broken:
            for row in report['frequency_scenarios']:
                for foot in row['feet']:foot['measured_frequency_hz']=1.
        self.assertFalse(assess('S5',broken,baseline)['passed'])
        broken = copy.deepcopy(reports)
        for row in broken[0]['scenarios']:
            if row['name']=='left':row['mean_actual_xyz_command_units']=[0.,0.,0.]
        self.assertFalse(assess('S5',broken,baseline)['passed'])
        self.assertFalse(assess('S5',reports[:1],baseline)['passed'])
        broken = copy.deepcopy(reports)
        for row in broken[0]['scenarios']:
            if row['name']=='turn_left':row['four_feet_stepping_fraction']=0.
        self.assertFalse(assess('S5',broken,baseline)['passed'])

    def test_actual_step_requires_continuous_air_and_unsigned_travel(self):
        detector = StepEvents(2, 'cpu', .02)
        xy = torch.zeros(2, 4, 2)
        contact = torch.zeros(2, 4, dtype=torch.bool)
        clear = torch.full((2, 4), .003)
        age, moving = torch.ones(2), torch.ones(2, dtype=torch.bool)
        for _ in range(3):
            detector.update(xy, contact, clear, age, moving)
        # Lateral and backward travel qualify, whereas stationary feet do not.
        xy[0, :, 1] = -.006
        landed = detector.update(xy, ~contact, clear*0, age, moving)
        self.assertTrue(landed[0].all())
        self.assertFalse(landed[1].any())
        detector.reset(torch.tensor([0, 1]))
        detector.update(xy, contact, clear, age*.1, moving)
        for _ in range(3):
            detector.update(xy, contact, clear, age, moving)
        xy += .01
        self.assertFalse(detector.update(xy, ~contact, clear*0, age, moving).any())

    def test_moving_start_curriculum_preserves_goal_and_direct_samples(self):
        from curriculum import MovingStartCurriculum
        c=MovingStartCurriculum(5,'cpu')
        target=torch.tensor([[0.,.04,0.],[-.04,0.,0.],[0.,0.,.35],[0.,0.,0.],[.105,0.,0.]])
        c.resample(torch.arange(5),target,torch.zeros(5),draws=torch.tensor([0.,1.,0.,0.,0.]))
        first=c.value(torch.zeros(5))
        torch.testing.assert_close(first[[0,2]],torch.tensor([[.105,0.,0.],[.105,0.,0.]]))
        torch.testing.assert_close(first[[1,3,4]],target[[1,3,4]])
        torch.testing.assert_close(c.value(torch.full((5,),4.)),target)
        torch.testing.assert_close(c.value(torch.full((5,),3.)),(first+target)*.5)
        torch.testing.assert_close(c.target,target)

    def test_velocity_sampler_includes_all_direct_motion_families(self):
        from curriculum import sample_velocity_commands
        torch.manual_seed(123)
        commands=torch.zeros(20000,16)
        sample_velocity_commands(commands,torch.arange(len(commands)),'S3',
            {'vx':[-.04,.12],'vy':[-.04,.04],'yaw_rate':[-.35,.35],'frequency':[.9,1.1]})
        x,y,w=commands[:,:3].unbind(1)
        pure_reverse=(x<=-.02)&(y==0)&(w==0)
        pure_lateral=(x==0)&(y.abs()>=.02)&(w==0)
        pure_yaw=(x==0)&(y==0)&(w.abs()>=.15)
        for mask in (pure_reverse,pure_lateral,pure_yaw):
            self.assertGreater(float(mask.float().mean()),.13)
            self.assertLess(float(mask.float().mean()),.17)
        self.assertTrue((x>=-.04).all() and (x<=.12).all())
        self.assertTrue((y.abs()<=.04).all() and (w.abs()<=.35).all())
        self.assertTrue((commands[:,5]==1.).all())

    def test_novel_exploration_preserves_forward_stop_and_deterministic_actor(self):
        policy=HistoryActorCritic(actor_hidden_dims=(32,16),critic_hidden_dims=(32,16),init_noise_std=.02)
        c=torch.tensor([CommandSpec.defaults(.08,.1,.07,-.69)]).repeat(5,1)
        c[:,:3]=torch.tensor([[.105,0.,0.],[0.,0.,0.],[-.04,0.,0.],[0.,.04,0.],[0.,0.,.35]])
        raw=pack(torch.randn(5,6,51),c,torch.zeros(5,4));original=policy.act_inference(raw)
        policy.novel_command_std_floor=.04
        policy.update_distribution(raw)
        torch.testing.assert_close(policy.action_std[:2],torch.full((2,15),.02))
        torch.testing.assert_close(policy.action_std[2:],torch.full((3,15),.04))
        torch.testing.assert_close(policy.act_inference(raw),original,atol=0,rtol=0)
        self.assertTrue((policy.act(raw)[:,-1]==0.).all())

    def test_terminal_bootstrap_and_actual_ppo_update(self):
        torch.manual_seed(11)
        policy = HistoryActorCritic(actor_hidden_dims=(32, 16), critic_hidden_dims=(32, 16))
        ppo = HistoryPPO(policy, gamma=.9, num_learning_epochs=1, num_mini_batches=1, auxiliary_policy_kl_limit=0.)
        ppo.init_storage(4, 2, [326], [92], [15])
        command = torch.tensor([CommandSpec.defaults(.08, .1, .07, -.69)]).repeat(4, 1)
        raw = pack(torch.randn(4, 6, 51), command, torch.randn(4, 4))
        critic = torch.randn(4, 92)
        critic[:, 51:67] = CommandSpec.normalize(command)
        terminal = critic*0
        terminal[:, 51:67] = CommandSpec.normalize(command)
        original = {k:v.clone() for k,v in policy.state_dict().items()}
        for step in range(2):
            with torch.no_grad():
                ppo.act(raw, critic)
                done = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
                timeout = torch.tensor([1, 0, 0, 0], dtype=torch.bool)
                values = policy.evaluate(terminal)
                ppo.observe_auxiliary(torch.randn(4, 3), torch.randn(4, 51), done)
                ppo.process_env_step(torch.ones(4), done, {'time_outs':timeout, 'terminal_critic':terminal})
                expected = timeout_bootstrap(torch.ones(4), timeout, values, .9)
                torch.testing.assert_close(ppo.storage.rewards[step, :, 0], expected)
                self.assertEqual(float(ppo.storage.rewards[step, 1]), 1.)
        with torch.no_grad():
            ppo.compute_returns(critic)
        loss = ppo.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(loss))))
        self.assertTrue(any(not torch.equal(original[k], v) for k,v in policy.state_dict().items() if k.startswith('actor.')))
        self.assertTrue(any(not torch.equal(original[k], v) for k,v in policy.state_dict().items() if k.startswith('critic.')))
        self.assertTrue(all(p.grad is None for p in policy.history_encoder.parameters()) is False)
        self.assertTrue(torch.isfinite(policy.std).all() and (policy.std > 0).all())
        self.assertEqual(len(ppo.auxiliary_rollout), 0)
        self.assertFalse(ppo.last_auxiliary_metrics['encoder_update_accepted'])
        self.assertEqual(ppo.last_auxiliary_metrics['auxiliary_learning_rate'],2.5e-5)
        for name,value in policy.state_dict().items():
            if name.startswith('history_encoder.'):
                torch.testing.assert_close(value,original[name],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
