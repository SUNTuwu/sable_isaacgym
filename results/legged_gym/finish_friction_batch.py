"""Finalize dashboard evidence only after both runs, cross tests and final video."""
import argparse,fcntl,hashlib,json,sys,time
from pathlib import Path
from friction_batch_status import ROOT,OUT,read,atomic,publish

parser=argparse.ArgumentParser();parser.add_argument('--keep-monitoring',action='store_true');args=parser.parse_args()
plan=read(OUT/'friction_batch_plan.json')
assert plan['status']=='complete','Wait for real stop, verification and cross evaluations'
post=ROOT/'postrun'/plan['id']
verified=read(post/'training_archives_verified.json')
assert len(verified['arms'])==2 and all(x['passed'] for x in verified['arms'])
video=read(OUT/'side_video_preview.json');finals={}
for arm in plan['arms']:
    run=Path(arm['run']);complete=read(run/'completion.json');backup=read(run/'backup_receipt.json')
    assert complete['training_exit_code']==0 and complete['reason']=='deadline'
    assert hashlib.sha256(Path(backup['archive']).read_bytes()).hexdigest()==backup['sha256']
    report=read(Path(complete['final_evaluation']['report']))
    cross=read(post/(arm['label']+'_cross_friction.json'))
    assert report['checkpoint_sha256']==cross['checkpoint_sha256']==backup['checkpoint']['sha256']
    if arm['label']=='low_front':
        assert video['run']==run.name and video['iteration']==report['iteration']
        assert video['checkpoint_sha256']==report['checkpoint_sha256']
        assert hashlib.sha256(Path(video['path']).read_bytes()).hexdigest()==video['sha256']
    finals[arm['label']]={'run':run.name,'iteration':report['iteration'],
        'updates':read(run/'status.json')['updates_this_run'],
        'summary':report['summary'],'cross_summary':cross['summary'],
        'priority':report['priority_diagnostics']['median_across_environments'],
        'swing_peak':{k:v for k,v in report['swing_peak_diagnostics'].items() if k!='samples'},
        'archive_sha256':backup['sha256'],'checkpoint_sha256':report['checkpoint_sha256']}
receipt={'created_unix':time.time(),'batch':plan['id'],'deadline':plan['deadline'],
         'primary_arm':'low_front','arms':finals,'final_video':video,'training_stopped':True,
         'archive_verification':str(post/'training_archives_verified.json'),
         'power_reward_enabled':False,'automatic_resume':False,
         'scope':'Remote server-local evidence. No Windows transfer or hardware validation claimed.'}
atomic(post/'completion_receipt.json',receipt)
atomic(OUT/'completion_receipt_front_friction_20261025.json',receipt)
active=read(OUT/'active_training_plan.json',{})
active.update(stage_status='complete',status='complete',completed_unix=time.time(),
              final_iteration=finals['low_front']['iteration'],
              completion_receipt=str(post/'completion_receipt.json'),automatic_resume=False)
atomic(OUT/'active_training_plan.json',active)
objectives=read(OUT/'next_training_objectives.json',{})
objectives.setdefault('status',{}).update(training='Both friction runs stopped at their deadline; archives and final/cross evaluations verified.',
    video='Final low-front-friction mainline checkpoint side video published; two unselected nominal-friction samples.',
    friction_experiment='Complete. Detailed two-domain results in completion_receipt_front_friction_20261025.json; low-front-friction remains the selected mainline.')
objectives['status']['higher_clearance']='v23 final5552 peaks>10mm fractions: '+str(finals['low_front']['swing_peak']['fraction_above10mm_per_foot'])+'; newly requested hind20mm criterion not yet met.'
objectives['status']['pair_peak_symmetry']='v23 final5552 front/hind RMS(m): '+str(finals['low_front']['swing_peak']['paired_peak_rmse_m_front_hind'])
objectives['status']['speed_stride']='v23 final5552 atfrontmu0.05: median16s forward'+str(finals['low_front']['summary']['median_displacement_m'])+'m,24/24survived. See cross-domain evidence separately.'
objectives['status']['remaining']='New request: frontmu0,hindpeak>20mm and diagnostic friction power; new configuration validation is separate. Force calibration remains failed.'
objectives['updated_unix']=time.time();atomic(OUT/'next_training_objectives.json',objectives)
sys.path.insert(0,str(ROOT/'deployment/legged_gym_knee_termination_20261025'))
from operator_feedback import append_message
low=finals['low_front'];control=finals['control']
def s(x):
    v=x['summary'];return '%d轮（新增%d），16秒存活%d/24、四足前进%d/24，前移中位%.3fm，膝盖触地%d/24'%(x['iteration'],x['updates'],round(v['survival_16s']*24),round(v['moving_all_four_fraction']*24),v['median_displacement_m'],round(v.get('knee_ground_failure_fraction',0)*24))
append_message('assistant','本批已按23:36截止停止，最终权重和两份完整训练归档已核验。低前足摩擦主线：'+s(low)+'；原摩擦对照：'+s(control)+'。交叉回放：低摩擦策略回到原摩擦，前移%.3fm、存活%d/24；对照策略放到低前足摩擦，前移%.3fm、存活%d/24。'%(low['cross_summary']['median_displacement_m'],round(low['cross_summary']['survival_16s']*24),control['cross_summary']['median_displacement_m'],round(control['cross_summary']['survival_16s']*24))+'网页已换成主线最终检查点录像，每个速度各一个未筛选样本。你新提出的后足峰值>20mm要求尚未达标，当前四足数据仍按本批2mm/60ms/5mm判据。功率仅作诊断但测力校准未通过，尚无可靠的每腿正负功数据，也没有加入奖励。主线选择保留为低前足摩擦，没有自动续训；留言仍可保存。',request_id='friction-batch-final-verified')
publish(plan)
with (OUT/'operator_monitor_status.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    status=read(OUT/'operator_monitor_status.json',{})
    status.update(last_checked_unix=time.time(),finished=not args.keep_monitoring,finished_unix=None if args.keep_monitoring else time.time(),
                  completion_receipt=str(post/'completion_receipt.json'))
    atomic(OUT/'operator_monitor_status.json',status)
print(json.dumps({'complete':True,'arms':finals,'receipt':str(post/'completion_receipt.json')},ensure_ascii=False))
