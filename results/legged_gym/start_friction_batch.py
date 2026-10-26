"""User-requested two-arm parallel PPO experiment, each with its own supervisor."""
import datetime,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
from friction_batch_status import read,atomic,publish
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym'
TEST=OUT/'friction_batch_preflight_v2';SOURCE=TEST/'runtime'
PARENT=ROOT/'runs/stand5_20261025_v22_highlift_speed/checkpoints/model_005314.pt'
PARENT_SHA='03d0ce03ac5834ea93cdd8bc27acbc77c5442ad33567d30c0d768c06d550593b'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate():
    from isaacgym import gymapi
    import torch
    assert '4090' in torch.cuda.get_device_name(0)
    assert digest(PARENT)==PARENT_SHA
    for p in (SOURCE/'assets').rglob('*'):
        if p.is_file():assert digest(p)==digest(PARENT.parent.parent/'runtime/assets'/p.relative_to(SOURCE/'assets'))
    completed=read(TEST/'validation_processes.json');assert completed
    deadline_proof=read(SOURCE/'verification/deadline_validation.json')
    assert deadline_proof['passed'] and deadline_proof['supervisor_sha256']==digest(SOURCE/'supervise.py')
    parent=torch.load(str(PARENT),map_location='cpu')['model_state_dict']
    report={'passed':True,'parent_sha256':PARENT_SHA,'arms':{},'source_files':{}}
    for label,scale in [('control',1.),('low_front',.05)]:
        smoke=TEST/(label+'_smoke');cfg=read(smoke/'config.json')
        initial=torch.load(str(smoke/'checkpoints/model_005314.pt'),map_location='cpu')
        final=torch.load(str(smoke/'checkpoints/model_005315.pt'),map_location='cpu')
        assert all(torch.equal(v,initial['model_state_dict'][k]) for k,v in parent.items() if k!='std')
        assert not initial['optimizer_state_dict']['state'] and final['optimizer_state_dict']['state']
        assert torch.allclose(initial['model_state_dict']['std'],torch.full_like(initial['model_state_dict']['std'],.04))
        assert final['iter']==5315 and all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
        assert bool((final['model_state_dict']['std']>=.04).all())
        assert all(any(not torch.equal(v,final['model_state_dict'][k]) for k,v in initial['model_state_dict'].items() if k.startswith(prefix)) for prefix in ['actor.','critic.'])
        assert cfg['mirror_policy'] and cfg['env']['env']['num_observations']==52 and cfg['env']['env']['num_actions']==12
        assert cfg['env']['material_friction']['front_scale']==scale
        assert cfg['env']['rewards']['scales']==read(ROOT/'runs/stand5_20261025_v22_highlift_speed/config.json')['env']['rewards']['scales']
        assert cfg['env']['reference']['strength']==.75 and cfg['env']['termination']['knee_contact_enabled']
        standing=read(TEST/(label+'_standing.json'));baseline=read(TEST/(label+'_baseline.json'))
        assert standing['passed'] and standing['front_friction_scale']==scale
        assert baseline['num_envs']==24 and baseline['horizon_s']==16 and baseline['checkpoint_sha256']==digest(smoke/'checkpoints/model_005314.pt')
        assert baseline['evaluator_sha256']==digest(SOURCE/'evaluate.py')
        if label=='control':assert baseline['summary']['survival_16s']>=.9,'Control parent stability failed'
        assert read(smoke/'status.json')['status']=='iterations_complete'
        manifest=read(smoke/'runtime_manifest.json')
        for p in list(SOURCE.glob('*.py'))+list((SOURCE/'assets').rglob('*')):
            if p.is_file():
                name=str(p.relative_to(SOURCE));assert manifest['files'][name]==digest(p),name
                report['source_files'][name]=digest(p)
        report['arms'][label]={'one_actual_ppo_update':True,'exact_parent_network_transfer':True,'fresh_optimizer':True,'standing':True,'baseline':baseline['summary'],'front_scale':scale,'initial_survival_gate_passed':baseline['summary']['survival_16s']>=.9,'mode':'validated' if baseline['summary']['survival_16s']>=.9 else 'experimental_user_requested_friction_ablation'}
    physics=read(OUT/'native_plane_zero_friction_calibration.json')
    control=read(TEST/'control_baseline.json');low=read(TEST/'low_front_baseline.json')
    max_difference=max(abs(a['displacement_m']-b['displacement_m']) for a,b in zip(control['samples'],low['samples']))
    assert max_difference>.01,'Material edit had no meaningful physical effect; refuse a fake control experiment'
    report['material_effect_max_paired_displacement_difference_m']=max_difference
    for shape,expected in [(.1,.05),(2.,1.)]:
        found=next(r for r in physics['rows'] if r['shape_friction']==shape)
        assert abs(found['effective_dynamic_mu_from_deceleration']-expected)<.002
    atomic(TEST/'batch_launch_validation.json',report)
    return report

def verify_archive(run):
    receipt=read(run/'backup_receipt.json');complete=read(run/'completion.json')
    assert receipt and complete['training_exit_code']==0 and complete['reason']=='deadline'
    archive=Path(receipt['archive']);assert digest(archive)==receipt['sha256']
    count=0
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            if not item.isfile():continue
            target=run.parent/item.name
            if target.name in ['supervisor.log','dashboard.log']:continue
            assert target.is_file() and hashlib.sha256(tar.extractfile(item).read()).hexdigest()==digest(target),item.name
            count+=1
    final=read(Path(complete['final_evaluation']['report']))
    assert complete['final_evaluation']['return_code']==0 and final['checkpoint_sha256']==receipt['checkpoint']['sha256']
    return {'run':run.name,'passed':True,'verified_files':count,'archive':receipt,'summary':final['summary']}

def main():
    lock=(ROOT/'runs/.friction_batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=validate()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError):continue
        scripts=[a.decode(errors='replace') for a in args if a.endswith(b'.py')]
        assert not any(str(ROOT) in s and Path(s).name in ['train.py','supervise.py'] for s in scripts),'Existing project training process: '+proc.name
    now=time.time();deadline=datetime.datetime.fromtimestamp(now+1800,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    plan={'id':'front_friction_20261025','started_unix':now,'deadline':deadline,'status':'training','num_envs_per_arm':512,'primary_arm':'low_front',
          'parent_sha256':PARENT_SHA,'arms':[],
          'authorization':'User explicitly requested a parallel batch using the previous version and very low forelimb friction.',
          'experimental_scope':'Two same-seed runs; one replication per condition, not statistical significance or hardware validation.'}
    jobs=[]
    for label,scale in [('control',1.),('low_front',.05)]:
        run=ROOT/'runs'/('stand5_20261025_v23_friction_'+label);assert not run.exists();run.mkdir()
        runtime=run/'runtime';runtime.mkdir()
        for p in SOURCE.glob('*.py'):shutil.copy2(p,runtime/p.name)
        shutil.copytree(SOURCE/'assets',runtime/'assets');(run/'checkpoints').mkdir();(run/'evaluations').mkdir()
        evidence=run/'preflight';evidence.mkdir()
        for name in [label+'_standing.json',label+'_baseline.json','batch_launch_validation.json']:
            shutil.copy2(TEST/name,evidence/name)
        shutil.copytree(TEST/(label+'_smoke'),evidence/'transfer_smoke')
        shutil.copytree(SOURCE/'verification',evidence/'deadline_validation')
        for p in OUT.glob('native_plane*friction_calibration.json'):shutil.copy2(p,evidence/p.name)
        shutil.copy2(TEST/(label+'_baseline.json'),run/'baseline_evaluation.json')
        arm={'label':label,'front_scale':scale,'run':str(run)};plan['arms'].append(arm)
        atomic(run/'stage_intent.json',{'batch':plan['id'],'arm':label,'parent_checkpoint':str(PARENT),'parent_sha256':PARENT_SHA,'front_floor_friction':scale,'rear_floor_friction':1.,'plane_friction':0.,'num_envs':512,'deadline':deadline,'reward_change':False,'power_reward_enabled':False,'material_caveat':'The common plane0/shape2 material remapping preserves foot-floor and body-floor friction; body-body friction is not identical to the parent. Both batch arms use the same remapping.','validation':proof['arms'][label]})
        command=[sys.executable,'-u',str(runtime/'supervise.py'),'--run',str(run),'--deadline',deadline,'--resume',str(PARENT),'--phase','guided','--reference-strength','.75','--mirror-policy','--front-friction-scale',str(scale),'--num-envs','512']
        log=(run/'supervisor.log').open('a');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        arm['supervisor_pid']=child.pid;jobs.append((run,child))
    atomic(OUT/'friction_batch_plan.json',plan)
    atomic(ROOT/'runs/current.json',{'run_dir':plan['arms'][1]['run']})
    publish(plan,agent_read=True)
    print('BATCH_STARTED',json.dumps(plan),flush=True)
    while any(child.poll() is None for _,child in jobs):
        publish(plan);time.sleep(10)
    assert all(child.returncode==0 for _,child in jobs),'One supervisor failed; inspect that run only'
    verified=[verify_archive(run) for run,_ in jobs]
    post=ROOT/'postrun'/plan['id'];post.mkdir(exist_ok=True)
    atomic(post/'training_archives_verified.json',{'created_unix':time.time(),'arms':verified})
    plan['status']='cross_evaluating';atomic(OUT/'friction_batch_plan.json',plan);publish(plan)
    for arm in plan['arms']:
        run=Path(arm['run']);cp=read(run/'checkpoint.json')['path'];opposite=.05 if arm['front_scale']==1. else 1.
        output=post/(arm['label']+'_cross_friction.json')
        with (post/(arm['label']+'_cross.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(run/'runtime/evaluate.py'),'--checkpoint',cp,'--output',str(output),'--front-friction-scale',str(opposite)],stdout=log,stderr=subprocess.STDOUT,timeout=240)
        assert result.returncode==0,'Cross-domain evaluation failed: '+arm['label']
    plan['status']='complete';plan['completed_unix']=time.time();plan['postrun']=str(post)
    atomic(OUT/'friction_batch_plan.json',plan);publish(plan)
    print('BATCH_COMPLETE',json.dumps(plan),flush=True)

if __name__=='__main__':main()
