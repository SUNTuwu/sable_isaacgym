"""Validate first, archive v25, then start requested stride and hip<=-9 course."""
import datetime,hashlib,json,shutil,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
TEST=OUT/'stride_hip9_preflight_20261026';SOURCE=TEST/'runtime'
OLD=ROOT/'runs/stand5_20261026_v25_speedtrack20'
PARENT=OLD/'checkpoints/model_005800.pt'
RUN=ROOT/'runs/stand5_20261026_v26_stride_hip9'
DEADLINE='2026-10-26T02:00:00+08:00'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2));temp.replace(p)

def main():
    from isaacgym import gymapi
    import torch
    assert not RUN.exists() and Path(read(ROOT/'runs/current.json')['run_dir'])==OLD
    assert time.time()<datetime.datetime.fromisoformat(DEADLINE).timestamp()-900
    smoke=TEST/'ppo_smoke';initial=torch.load(str(smoke/'checkpoints/model_005800.pt'),map_location='cpu')
    updated=torch.load(str(smoke/'checkpoints/model_005801.pt'),map_location='cpu');parent=torch.load(str(PARENT),map_location='cpu')
    assert read(smoke/'status.json')['status']=='iterations_complete'
    assert all(torch.equal(v,initial['model_state_dict'][k]) for k,v in parent['model_state_dict'].items() if k!='std')
    assert not initial['optimizer_state_dict']['state'] and updated['optimizer_state_dict']['state']
    assert bool((updated['model_state_dict']['std']>=.04).all()) and all(torch.isfinite(v).all() for v in updated['model_state_dict'].values())
    for prefix in ['actor.','critic.']:assert any(not torch.equal(v,updated['model_state_dict'][k]) for k,v in initial['model_state_dict'].items() if k.startswith(prefix))
    cfg=read(smoke/'config.json');assert cfg['mirror_policy'] and cfg['env']['env']['num_observations']==52
    assert cfg['env']['rewards']['scales']['hip_stand_exp']==-2.5 and cfg['env']['rewards']['scales']['tracking_lin_vel']==6
    assert cfg['env']['termination']['knee_contact_enabled'] and cfg['env']['material_friction']['front_scale']==.02
    assert cfg['env']['gait']['swing_heights_m']==[.023]*4 and cfg['env']['reference']['amplitude_multipliers']==[4./3.]*4
    standing=read(TEST/'standing_reused.json');baseline=read(TEST/'parent_baseline.json');static=read(TEST/'fullstride_static.json');reward=read(TEST/'reward_checks.json');new_symmetry=read(TEST/'symmetry_geometry.json')
    assert standing['passed'] and standing['front_friction_scale']==.02
    assert baseline['num_envs']==24 and baseline['horizon_s']==16 and baseline['summary']['survival_16s']>=.9
    assert baseline['evaluator_sha256']==sha(SOURCE/'evaluate.py') and baseline['checkpoint_sha256']==sha(smoke/'checkpoints/model_005800.pt')
    assert static['passed'] and all(sha(SOURCE/n)==h for n,h in static['runtime_files'].items() if n=='guided_reference.py' or n.startswith('assets/'))
    assert reward['passed'] and all(sha(SOURCE/n)==h for n,h in reward['source_sha256'].items())
    assert new_symmetry['passed'] and new_symmetry['checker_sha256']==sha(SOURCE/'check_symmetry_geometry.py')
    assert cfg['env']['posture']['hip_preferred_max_deg']==-9 and cfg['env']['rewards']['scales']['stride_progress']==2
    assert baseline['all20_diagnostics']['four_feet_2steps_forward_survive_fraction']>=.9
    manifest=read(smoke/'runtime_manifest.json')['files'];assert all(sha(SOURCE/n)==h for n,h in manifest.items())
    for name in ['sable.xml','physical_manifest.json','authored_model.json']:
        assert sha(SOURCE/'assets'/name)==sha(OLD/'runtime/assets'/name)
    proof_src=ROOT/'deployment/legged_gym_speedtrack20_20261026/verification'
    deadline=read(proof_src/'deadline_validation.json');symmetry=read(proof_src/'symmetry_geometry.json')
    assert deadline['passed'] and deadline['supervisor_sha256']==sha(SOURCE/'supervise.py') and symmetry['passed']
    proof={'passed':True,'created_unix':time.time(),'parent_checkpoint_sha256':sha(PARENT),
       'one_real_ppo_update':True,'exact_network_transfer':True,'fresh_optimizer':True,
       'standing':standing['passed'],'baseline':baseline['summary'],'all20_baseline':baseline['all20_diagnostics'],
       'source_files':manifest,'deadline':DEADLINE,'hip_change':'User requests hip<=-9deg: one-sided exponential cost above-9deg, physical limits unchanged. Valid forward landing stride reward2; full horizontal reference.'}
    atomic(TEST/'launch_validation.json',proof)
    atomic(OLD/'stage_transition_request.json',{'created_unix':time.time(),'reason':'User requested low friction, hip<=-9deg preference and forward stride reward','next_run':RUN.name,'new_stage_validation':str(TEST/'launch_validation.json'),'global_deadline':DEADLINE})
    (OLD/'STOP').write_text('User-requested speed/all-feet20mm course; validated replacement ready.\n')
    print('OLD_STAGE_STOP_REQUESTED',flush=True)
    end=time.time()+420
    while not (OLD/'backup_receipt.json').exists():
        assert time.time()<end,'Old stage archive wait exceeded; inspect without launching duplicate training'
        time.sleep(3)
    backup=read(OLD/'backup_receipt.json');complete=read(OLD/'completion.json')
    assert complete['training_exit_code']==0 and complete['reason']=='training_exited'
    assert sha(Path(backup['archive']))==backup['sha256']
    verified=0
    with tarfile.open(backup['archive']) as tar:
        for m in tar.getmembers():
            if m.isfile() and Path(m.name).name!='supervisor.log':
                assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(OLD.parent/m.name),m.name;verified+=1
    final=read(Path(complete['final_evaluation']['report']))
    assert complete['final_evaluation']['return_code']==0 and final['checkpoint_sha256']==backup['checkpoint']['sha256']
    atomic(OUT/'v25_final_verification_20261026.json',{'passed':True,'verified_files':verified,'archive':backup,'summary':final['summary']})
    RUN.mkdir();rt=RUN/'runtime';rt.mkdir();(RUN/'checkpoints').mkdir();(RUN/'evaluations').mkdir()
    for p in SOURCE.glob('*.py'):shutil.copy2(p,rt/p.name)
    shutil.copytree(SOURCE/'assets',rt/'assets');pre=RUN/'preflight';pre.mkdir()
    for n in ['standing_reused.json','parent_baseline.json','fullstride_static.json','launch_validation.json','reward_checks.json','symmetry_geometry.json']:shutil.copy2(TEST/n,pre/n)
    shutil.copytree(smoke,pre/'transfer_smoke');shutil.copytree(proof_src,pre/'deadline_and_symmetry')
    shutil.copy2(TEST/'parent_baseline.json',RUN/'baseline_evaluation.json')
    intent={'created_unix':time.time(),'deadline':DEADLINE,'front_floor_mu':.02,'target_front_floor_mu':0.,
       'hind_peak_goal_m':.020,'all_feet_peak_goal_m':.020,'current_swing_target_m':.023,'reference_strength':.75,
       'parent':str(PARENT),'parent_sha256':sha(PARENT),'hip_posture_penalty_weight':-2.5,'hip_preferred_max_deg':-9.,'physical_hip_limits_deg':[-85,24],'stride_reward_weight':2.,'horizontal_reference_scale':1.,
       'velocity_reward_weight':6.,'velocity_sigma':.000625,'swing_peak_shortfall_weight':-2.,
       'power_reward_enabled':False,'power_measurement_validated':False,
       'headline':'当前主线：低前足摩擦 · 髋≤−9° · 加大有效步长',
       'description':'前足对地摩擦0.02、后足1；四足参考峰值23mm，水平步幅100%。髋≤−9°无姿态惩罚，超过才指数扣分；物理范围−85°～+24°。奖励真正离地后的向前步长，按指令所需步长封顶；膝部触地终止，不使用功率奖励。',
       'scope':'Validated initial full-stride stability. New hip preference and landing stride rewards must be assessed after learning; physical limits unchanged.'}
    atomic(RUN/'stage_intent.json',intent);shutil.copy2(Path(__file__),RUN/'launch_used.py')
    cmd=[sys.executable,'-u',str(rt/'supervise.py'),'--run',str(RUN),'--deadline',DEADLINE,'--resume',str(PARENT),
       '--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale','.02','--num-envs','1024']
    with (RUN/'supervisor.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(200):
        if (RUN/'status.json').exists():break
        assert child.poll() is None,'New supervisor exited';time.sleep(.1)
    assert (RUN/'status.json').exists()
    atomic(ROOT/'runs/current.json',{'run_dir':str(RUN)})
    prior=read(OUT/'adaptation_round_plan_20261026.json');plan=dict(intent,run=str(RUN),status='training',supervisor_pid=child.pid,started_unix=time.time(),stages=prior['stages']+[str(RUN)])
    atomic(OUT/'adaptation_round_plan_20261026.json',plan)
    atomic(OUT/'active_training_plan.json',dict(plan,run=RUN.name,stage_status='training',web_port=11111,progress_interval_s=300,knee_ground_death=True,automatic_resume=False))
    print('NEW_STAGE_STARTED',json.dumps(plan,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
