"""Apply the completed frozen A/B review without interrupting GPU training."""
import datetime, hashlib, json, os, py_compile, shutil, signal, subprocess, time
from pathlib import Path

P=Path('/root/gpufree-data/projects/sable-isaacgym')
D=P/'deployment/night_20261029'

def read(p): return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_name(p.name+'.turnreview.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def main():
    ab=read(D/'turn_netprogress_ab.json')
    assert ab['complete'] and all(r['survival_fraction']==1 for r in ab['reports'])
    candidate=D/'monitor_turn_cap.py'
    py_compile.compile(str(candidate),doraise=True)
    assert "min(1.2 if capped else 4." in candidate.read_text()
    pid=read(D/'monitor_pid.json')['pid']
    cmd=Path('/proc',str(pid),'cmdline')
    assert cmd.exists() and str(D/'monitor.py').encode() in cmd.read_bytes()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    review_dir=D/('turn_weight_review_'+stamp);review_dir.mkdir()
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not cmd.exists() or not cmd.read_bytes():break
        time.sleep(.1)
    else: raise RuntimeError('CPU monitor did not exit; no control changes made')
    try:
        shutil.copy2(D/'monitor.py',review_dir/'monitor_before.py')
        shutil.copy2(candidate,D/'monitor.py')
        before=read(D/'control.json');write(review_dir/'control_before.json',before)
        control=json.loads(json.dumps(before));baseline=read(D/'baseline_sources.json')
        bank=Path(read(P/'results/teacher_bank/current.json')['bank'])
        for ident in ('04_TURN_LEFT','05_TURN_RIGHT'):
            evaluation=read(bank/ident/'latest_evaluation.json')
            write(review_dir/(ident+'_evaluation_before.json'),evaluation)
            write(review_dir/(ident+'_checkpoint_before.json'),read(bank/ident/'checkpoint.json'))
            cfg=control['teachers'][ident]
            assert cfg['reward_profile']=='cycle_net_progress_gamma099_v1'
            cfg['multipliers']['v2_yaw']=1.2
            cfg['reason']='Completed paired 24s frozen A/B: weight1.2 ranks rotating baseline above degenerated walking policy under PPO discount; restart from selected trot baseline.'
            cfg['reload']=dict(baseline[ident]['checkpoint'],token='turn_weight_'+stamp+'_'+ident)
        control['revision']+=1
        write(D/'control.json',control);write(review_dir/'control_after.json',control)
        receipt=dict(time=time.time(),type='turn_netprogress_weight_review',bank=str(bank),
            changed_teachers=['04_TURN_LEFT','05_TURN_RIGHT'],yaw_multiplier=1.2,
            monitor_automatic_yaw_cap=1.2,ab_complete=True,ab_sha256=hashlib.sha256((D/'turn_netprogress_ab.json').read_bytes()).hexdigest(),
            evidence_scope='Frozen paired policy returns, not post-update training acceptance',
            before_discounted_returns=[-2.8178886771202087,-2.242828130722046],
            reweighted_discounted_returns=[-.37867771089076996,-1.030689150094986],
            baseline_reload_requested=True,gpu_trainers_interrupted=False)
        write(review_dir/'receipt.json',receipt)
        with (D/'manual_reward_reviews.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    finally:
        with (D/'monitor.py.log').open('a') as log:
            child=subprocess.Popen(['python3',str(D/'monitor.py')],cwd=str(P),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(D/'monitor_pid.json',dict(pid=child.pid,profile='stable_objective_turn_yaw_cap_v4'))
    print(json.dumps(dict(review=str(review_dir),monitor_pid=child.pid,control_revision=control['revision'])))

if __name__=='__main__':main()
