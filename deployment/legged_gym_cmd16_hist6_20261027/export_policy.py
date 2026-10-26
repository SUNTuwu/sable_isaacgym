"""Export the complete history/mirror policy and explicit full-target contract."""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from torch import nn
from history_policy import HistoryActorCritic, DeploymentPolicy
from action_schema import ACTION_NAMES, ActionSpec
from command_schema import COMMAND_NAMES, COMMAND_SCALES, CommandSpec
from model_mapping import ModelMapping
from observation_schema import pack
from gait_clock import clocks
from run_io import atomic_json, sha256


class FullTargetPolicy(nn.Module):
    def __init__(self, policy, defaults, limits):
        super().__init__()
        self.policy = policy
        self.register_buffer('default_angles', defaults)
        self.register_buffer('limits', limits)

    def forward(self, raw):
        action = self.policy.act_inference(raw)
        target = self.default_angles + .35*action
        return torch.maximum(torch.minimum(target, self.limits[:, 1]), self.limits[:, 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('An export is immutable; choose a new output directory')
    a.output.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    manifest = json.loads((source/'assets/physical_manifest.json').read_text())
    mapping = ModelMapping(source/'assets/sable.xml', source/'assets/physical_manifest.json')
    m = mapping.default_measurements()
    defaults = torch.tensor([manifest['default_angles'][name] for name in ACTION_NAMES])
    # These are exported editable simulation limits, not measured motor calibration.
    import xml.etree.ElementTree as ET
    root = ET.parse(source/'assets/sable.xml')
    ranges = {j.get('name'):[float(x) for x in j.get('range').split()] for j in root.iter('joint') if j.get('name') in ACTION_NAMES}
    limits = torch.tensor([ranges[name] for name in ACTION_NAMES])
    policy = HistoryActorCritic().eval()
    checkpoint = torch.load(str(a.checkpoint), map_location='cpu')
    policy.load_state_dict(checkpoint['model_state_dict'], strict=True)
    commands = torch.tensor([CommandSpec.defaults(m['shoulder_height_m'],m['hip_height_m'],m['width0_m'],float(defaults[-1]))]).repeat(32,1)
    raw = pack(torch.randn(32,6,51)*.2, commands, clocks(torch.rand(32), commands))
    wrappers = {'actor_raw326.ts':DeploymentPolicy(policy),'target_raw326.ts':FullTargetPolicy(policy,defaults,limits)}
    errors = {}
    for name, wrapper in wrappers.items():
        traced = torch.jit.trace(wrapper, raw, check_trace=True)
        traced.save(str(a.output/name))
        loaded = torch.jit.load(str(a.output/name)).eval()
        error = max(float((loaded(raw[:n])-wrapper(raw[:n])).abs().max()) for n in (1,7,32))
        assert error < 1e-6
        errors[name] = error
    # End-to-end inference timing includes the encoder and both mirror branches.
    target_policy = torch.jit.load(str(a.output/'target_raw326.ts')).eval()
    with torch.no_grad():
        for _ in range(20):target_policy(raw[:1])
        start = time.perf_counter()
        for _ in range(500):target_policy(raw[:1])
        elapsed = (time.perf_counter()-start)/500
    accepted = None
    if a.acceptance:
        accepted = json.loads(a.acceptance.read_text())
        assert accepted['passed'] and accepted['stage']=='S5' and accepted['checkpoint_sha256']==sha256(a.checkpoint)
        shutil.copy2(a.acceptance,a.output/'acceptance.json')
    for name in ('command_schema.py','action_schema.py','observation_schema.py','gait_clock.py'):
        shutil.copy2(source/name,a.output/name)
    contract = dict(schema='sable-cmd16-hist6-export/1',checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=sha256(a.checkpoint),iteration=checkpoint['iter'],acceptance_passed=bool(accepted),
        raw_dimension=326,history_frames=6,proprio_dimension=51,actor_internal_dimension=90,critic_dimension=92,
        raw_slices={'history_current_to_past':[0,306],'normalized_commands':[306,322],'current_clock':[322,326]},
        proprio_slices={'root_angular_velocity_times_0.25':[0,3],'projected_gravity':[3,6],
            'q_minus_default_radians':[6,21],'q_velocity_times_0.05':[21,36],
            'previous_executed_clamped_target_minus_default_divided_by_0.35':[36,51]},
        command_names=list(COMMAND_NAMES),command_scales=list(COMMAND_SCALES),command_defaults=commands[0].tolist(),
        action_names=list(ACTION_NAMES),default_angles_rad=defaults.tolist(),physical_limits_rad=limits.tolist(),
        action_scale_rad=.35,waist_frozen=policy.freeze_waist,passive_joint='R_hip_roll_mimic',passive_relation='q_right=-q_hip_roll; no action and zero motor torque',
        coordinates='meters, seconds, radians; +X forward,+Y left,+Z up; S0-S2 teacher body-frame compatibility, S3/S5 velocity commands use heading-aligned horizontal frame',
        history_reset='Repeat current fresh proprioception six times; previous executed target resets to default; append once per20ms control sample only',
        phase_update='Integrate phase += frequency*0.02 modulo1; do not reset phase on a frequency command change',
        clocks='Four current sin(2*pi*u) values; u=local/(2*beta) in stance, else0.5+(local-beta)/(2*(1-beta)); phase offsets FL0,FRcommand6,HLcommand7,HRcommand8; use provided GaitClock stop hysteresis',
        control_period_s=.02,physics_period_s=.0005,physics_substeps_per_control=40,target_interpolation='hold full joint target throughout control interval',
        aux_reward_coeff=1.,output='actor_raw326 returns15 dimensionless offsets; target_raw326 returns15 limited full absolute joint targets in radians',
        asset_sha256=sha256(source/'assets/sable.xml'),manifest_sha256=sha256(source/'assets/physical_manifest.json'),
        numerical_max_errors=errors,cpu_complete_policy_latency_seconds=elapsed,
        limitations='Simulation export; mass, contact and actuator calibration are not hardware validation. No claim that all16 command dimensions have been learned.',
        created_unix=time.time())
    atomic_json(a.output/'contract.json',contract)
    atomic_json(a.output/'files.json',{str(f.relative_to(a.output)):sha256(f) for f in a.output.iterdir() if f.is_file()})
    print(json.dumps({'output':str(a.output),'acceptance_passed':bool(accepted),'max_errors':errors,'cpu_latency_seconds':elapsed}))


if __name__=='__main__':main()
