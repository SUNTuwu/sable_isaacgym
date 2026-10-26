"""Validated continuation, with a fixed02:00 deadline and immutable runtime."""
import datetime,fcntl,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
TEST=OUT/'friction_adapt_20261026_preflight';SOURCE=TEST/'runtime'
PARENT=ROOT/'runs/stand5_20261025_v23_friction_low_front/checkpoints/model_005552.pt'
RUN=ROOT/'runs/stand5_20261026_v24_friction002_adapt'
DEADLINE='2026-10-26T02:00:00+08:00'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)

def main():
    from isaacgym import gymapi
    import torch
    assert time.time()<datetime.datetime.fromisoformat(DEADLINE).timestamp()-600
    assert '4090' in torch.cuda.get_device_name(0)
    lock=(ROOT/'runs/.adaptation_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    current=Path(read(ROOT/'runs/current.json')['run_dir'])
    if (current/'status.json').exists():
        state=read(current/'status.json')
        assert state['status'] not in ['starting','training','evaluating'] or (current/'completion.json').exists(),'Existing active mainline'
    assert not RUN.exists()
    parent_backup=read(PARENT.parent.parent/'backup_receipt.json')
    assert sha(Path(parent_backup['archive']))==parent_backup['sha256']
    assert sha(PARENT)==parent_backup['checkpoint']['sha256']=='69e273db4a4664c346c4951a516b1fdf4895a946c60e59293ead51f821929d23'
    smoke=TEST/'ppo_smoke';parent=torch.load(str(PARENT),map_location='cpu')
    initial=torch.load(str(smoke/'checkpoints/model_005552.pt'),map_location='cpu')
    final=torch.load(str(smoke/'checkpoints/model_005553.pt'),map_location='cpu')
    assert read(smoke/'status.json')['status']=='iterations_complete'
    assert all(torch.equal(v,initial['model_state_dict'][k]) for k,v in parent['model_state_dict'].items() if k!='std')
    assert not initial['optimizer_state_dict']['state'] and final['optimizer_state_dict']['state']
    assert torch.allclose(initial['model_state_dict']['std'],torch.full_like(initial['model_state_dict']['std'],.04))
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    assert bool((final['model_state_dict']['std']>=.04).all())
    for prefix in ['actor.','critic.']:
        assert any(not torch.equal(v,final['model_state_dict'][k]) for k,v in initial['model_state_dict'].items() if k.startswith(prefix))
    baseline=read(TEST/'parent_baseline.json');standing=read(TEST/'standing.json')
    assert standing['passed'] and standing['seconds']==16 and standing['front_friction_scale']==.02
    assert baseline['horizon_s']==16 and baseline['num_envs']==24 and baseline['summary']['survival_16s']>=.9
    assert baseline['material_friction']['nominal_front_floor_mu']==.02
    assert baseline['checkpoint_sha256']==sha(smoke/'checkpoints/model_005552.pt')
    assert baseline['evaluator_sha256']==sha(SOURCE/'evaluate.py')
    manifest=read(smoke/'runtime_manifest.json')['files']
    assert all(sha(SOURCE/n)==h for n,h in manifest.items())
    for p in (SOURCE/'assets').rglob('*'):
        if p.is_file():assert sha(p)==sha(PARENT.parent.parent/'runtime/assets'/p.relative_to(SOURCE/'assets'))
    proof_source=OUT/'friction_batch_preflight_v2/runtime/verification'
    deadline_proof=read(proof_source/'deadline_validation.json')
    assert deadline_proof['passed'] and deadline_proof['supervisor_sha256']==sha(SOURCE/'supervise.py')
    proof={'passed':True,'created_unix':time.time(),'parent_sha256':sha(PARENT),'one_real_ppo_update':True,
           'actor_critic_exact_transfer':True,'fresh_optimizer':True,'finite_update':True,
           'standing_passed':True,'baseline':baseline['summary'],'deadline':DEADLINE,
           'runtime_files':manifest,'same_supervisor_deadline_proof':deadline_proof}
    atomic(TEST/'launch_validation.json',proof)
    RUN.mkdir();runtime=RUN/'runtime';runtime.mkdir()
    for p in SOURCE.glob('*.py'):shutil.copy2(p,runtime/p.name)
    shutil.copytree(SOURCE/'assets',runtime/'assets')
    (RUN/'checkpoints').mkdir();(RUN/'evaluations').mkdir();preflight=RUN/'preflight';preflight.mkdir()
    for name in ['parent_baseline.json','standing.json','launch_validation.json']:shutil.copy2(TEST/name,preflight/name)
    shutil.copytree(smoke,preflight/'transfer_smoke')
    shutil.copytree(proof_source,preflight/'deadline_validation')
    shutil.copy2(TEST/'parent_baseline.json',RUN/'baseline_evaluation.json')
    intent={'created_unix':time.time(),'deadline':DEADLINE,'front_floor_mu':.02,'target_front_floor_mu':0.,
            'hind_peak_goal_m':.020,'current_swing_target_m':.0135,'reference_strength':.75,
            'parent':str(PARENT),'parent_sha256':sha(PARENT),'power_reward_enabled':False,
            'power_measurement_validated':False,'mode':'validated_staged_friction_adaptation',
            'scope':'Intermediate friction course; not zero-friction or20mm-hind success.'}
    atomic(RUN/'stage_intent.json',intent)
    shutil.copy2(Path(__file__),RUN/'launch_used.py')
    cmd=[sys.executable,'-u',str(runtime/'supervise.py'),'--run',str(RUN),'--deadline',DEADLINE,
         '--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy',
         '--front-friction-scale','.02','--num-envs','1024']
    with (RUN/'supervisor.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(200):
        if (RUN/'status.json').exists():break
        assert child.poll() is None,'Supervisor exited during launch'
        time.sleep(.1)
    assert (RUN/'status.json').exists(),'No trainer startup status'
    atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
    plan=dict(intent,run=str(RUN),status='training',supervisor_pid=child.pid,started_unix=time.time(),stages=[str(RUN)])
    atomic(OUT/'adaptation_round_plan_20261026.json',plan)
    atomic(OUT/'active_training_plan.json',dict(plan,run=RUN.name,stage_status='training',web_port=11111,
           progress_interval_s=300,knee_ground_death=True,tolerance_deg=5,automatic_resume=False))
    print(json.dumps(plan,ensure_ascii=False))

if __name__=='__main__':main()
