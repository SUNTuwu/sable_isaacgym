import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from action_schema import ActionSpec, ACTION_NAMES
from command_schema import CommandSpec
from observation_schema import HistoryBuffer, pack, mirror_raw, mirror_critic, proprioception
from gait_clock import GaitClock, clocks, local_phases, shortest_phase_delta
from history_policy import HistoryActorCritic, DeploymentPolicy
from model_mapping import ModelMapping, heading_velocity


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.commands = torch.tensor([CommandSpec.defaults(.08, .1, .07, -.6981317)]).repeat(7, 1)

    def test_slices_and_history_reset(self):
        p = torch.arange(51.).repeat(7, 1)
        h = HistoryBuffer(7, 'cpu')
        h.sample(p, 0)
        self.assertTrue(torch.equal(h.data, p[:, None].expand(-1, 6, -1)))
        h.sample(p + 100, 1)
        saved = h.data.clone()
        h.sample(p + 200, 1)
        self.assertTrue(torch.equal(h.data, saved))
        h.reset(torch.tensor([1, 5]))
        h.sample(p + 300, 2)
        self.assertTrue(torch.equal(h.data[[1, 5]], (p + 300)[[1, 5], None].expand(-1, 6, -1)))
        self.assertTrue(torch.equal(h.data[0, 1], p[0] + 100))
        raw = pack(h.data, self.commands, torch.zeros(7, 4))
        self.assertEqual(raw.shape, (7, 326))
        self.assertTrue(torch.equal(raw[:, :51], p + 300))
        self.commands[:, 0] = .12
        updated = pack(h.data, self.commands, torch.zeros(7, 4))
        self.assertTrue(torch.equal(raw[:, :306], updated[:, :306]))
        self.assertFalse(torch.equal(raw[:, 306], updated[:, 306]))

    def test_mirror_and_clock_rebasing(self):
        self.commands[:, 6:9] = torch.rand(7, 3)
        phase = torch.rand(7)
        c = clocks(phase, self.commands)
        mirrored_commands = CommandSpec.mirror(self.commands)
        c_mirror = clocks(torch.remainder(phase + self.commands[:, 6], 1), mirrored_commands)
        torch.testing.assert_close(c_mirror, c[:, [1, 0, 3, 2]], atol=2e-6, rtol=2e-6)
        raw = pack(torch.randn(7, 6, 51), self.commands, c)
        torch.testing.assert_close(mirror_raw(mirror_raw(raw)), raw, atol=1e-6, rtol=1e-6)
        critic = torch.randn(7, 92)
        critic[:, 51:67] = CommandSpec.normalize(self.commands)
        torch.testing.assert_close(mirror_critic(mirror_critic(critic)), critic, atol=1e-6, rtol=1e-6)
        q = torch.randn(7, 15)
        torch.testing.assert_close(ActionSpec.mirror(ActionSpec.mirror(q)), q)
        self.assertTrue(torch.equal(ActionSpec.mirror(q)[:, 1], q[:, 1]))

    def test_integrated_frequency_and_stop(self):
        clock = GaitClock(7, 'cpu')
        old_phase = torch.rand(7)
        clock.reset_from_teacher(torch.arange(7), old_phase)
        expected = torch.remainder(old_phase[:, None] + torch.tensor([.75, .25, 0., .5]), 1)
        torch.testing.assert_close(local_phases(clock.phase, self.commands), expected, atol=1e-6, rtol=1e-6)
        before = clock.phase.clone()
        self.commands[:, 5] = 1.1
        clock.advance(self.commands)
        torch.testing.assert_close(shortest_phase_delta(clock.phase, before), torch.full((7,), .022), atol=1e-6, rtol=1e-6)
        self.commands[:, :3] = 0
        clock.advance(self.commands, True)
        self.assertTrue((clock.values(self.commands, True) == 0).all())
        self.commands[:, 2] = .3
        clock.advance(self.commands, True)
        self.assertTrue(clock.moving.all())

    def test_full_target_and_frozen_waist(self):
        default = torch.linspace(-1, 1, 15)
        limits = torch.stack((default - .6, default + .7), -1)
        action = torch.full((7, 15), 4.)
        target = ActionSpec.target(action, default, limits, freeze_waist=True)
        torch.testing.assert_close(target[:, :14], (default + .7)[None, :14].expand(7, -1))
        torch.testing.assert_close(target[:, 14], default[14].expand(7))
        torch.testing.assert_close(ActionSpec.offsets(target, default)[:, :14], torch.full((7, 14), 2.))
        self.assertNotIn('R_hip_roll_mimic', ACTION_NAMES)
        p = proprioception(torch.ones(7, 3), torch.zeros(7, 3), default.expand(7, -1),
                           torch.zeros(7, 15), default, target)
        torch.testing.assert_close(p[:, 36:51], ActionSpec.offsets(target, default))

    def test_policy_symmetry_privilege_and_export(self):
        model = HistoryActorCritic(actor_hidden_dims=(32, 24), critic_hidden_dims=(32, 24))
        self.commands[:, 6:9] = torch.rand(7, 3)
        raw = pack(torch.randn(7, 6, 51), self.commands, torch.randn(7, 4))
        a = model.act_inference(raw)
        self.assertEqual(a.shape, (7, 15))
        torch.testing.assert_close(model.act_inference(mirror_raw(raw)), ActionSpec.mirror(a), atol=2e-6, rtol=2e-6)
        critic = torch.randn(7, 92)
        model.evaluate(critic)
        model.evaluate(critic * 100)
        torch.testing.assert_close(model.act_inference(raw), a)
        samples = model.act(raw)
        self.assertTrue((samples[:, -1] == 0).all())
        self.assertTrue((model.action_std > 0).all())
        self.assertTrue(torch.isfinite(model.get_actions_log_prob(samples)).all())
        model.zero_grad()
        a.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.history_encoder.parameters()))
        loss, stats = model.history_encoder.auxiliary_loss(raw[:, :306], torch.randn(7, 3),
                                                          torch.randn(7, 51), torch.tensor([0, 1, 0, 1, 0, 0, 1], dtype=torch.bool))
        loss.backward()
        self.assertEqual(int(stats['valid_pairs']), 4)
        self.assertGreater(float(model.history_encoder.encoder[-1].weight.grad[3:].abs().sum()), 0)
        traced = torch.jit.trace(DeploymentPolicy(model.eval()), raw, check_trace=True)
        torch.testing.assert_close(traced(raw[:3]), model.act_inference(raw[:3]), atol=1e-6, rtol=1e-6)

    def test_fk_and_heading_semantics(self):
        base = Path(__file__).resolve().parents[1] / 'assets'
        m = ModelMapping(base / 'sable.xml', base / 'physical_manifest.json')
        defaults = m.default_measurements()
        self.assertGreater(defaults['shoulder_height_m'], .05)
        self.assertGreater(defaults['hip_height_m'], .05)
        self.assertGreater(defaults['hind_width_m'], 0)
        q = torch.tensor([[0., 0., math.sin(.3), math.cos(.3)]])
        v = torch.tensor([[math.cos(.6), math.sin(.6), 0.]])
        xy, yaw = heading_velocity(q, v, torch.tensor([[0., 0., .4]]))
        torch.testing.assert_close(xy, torch.tensor([[1., 0.]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(yaw, torch.tensor([.4]), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
