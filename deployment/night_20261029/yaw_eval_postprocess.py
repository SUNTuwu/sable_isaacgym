"""CPU-only sidecar for first-episode yaw evaluation. Never edits trainer output.

python yaw_eval_postprocess.py --receipt /.../deployment/yaw_sine_20261029/launch_receipt.json
Use --once to process one stable pair. Default polling interval is30seconds.
Only corrected_evaluation.json is written beneath the selected run directory.
"""
import argparse
import hashlib
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path
import numpy as np

SCHEMA='yaw_first_episode_v1'

def summarize_yaw_evaluation(data, budget, settle, episode, dt):
    age = np.asarray(data['age'])
    fall = np.asarray(data['fall'], dtype=bool)
    error = np.asarray(data['error'])
    if age.ndim != 2 or fall.shape != age.shape or error.shape != age.shape:
        raise ValueError('Expected matching [time, environment] age/fall/error arrays')
    if not len(age):
        raise ValueError('Replay is empty')
    reset_start = np.zeros_like(fall)
    reset_start[1:] = age[1:] < age[:-1] - dt*.5
    # Keep the terminal failure sample. Exclude every subsequent reset episode.
    previous_fall = np.cumsum(fall, axis=0) - fall
    first_episode = (previous_fall == 0) & (np.cumsum(reset_start, axis=0) == 0)
    valid = first_episode & (age >= settle)
    initial_ok = np.isfinite(age[0]) & (age[0] > 0) & (age[0] <= 1.5*dt)
    first_age = np.where(first_episode, age, 0).max(axis=0)
    survived = ~(fall & first_episode).any(axis=0)
    complete = initial_ok & survived & (first_age >= episode-1e-5)
    counts = np.where(first_episode[:, :, None], data['count'], 0).max(axis=0)
    sample_count = int(valid.sum())
    finite = bool(sample_count and np.isfinite(error[valid]).all())
    rmse = float(np.rad2deg(np.sqrt(np.square(error[valid]).mean()))) if finite else None
    peak = float(np.rad2deg(np.abs(error[valid]).max())) if finite else None
    first_failure_s = []
    per_env_rmse = []
    for i in range(age.shape[1]):
        indices = np.flatnonzero(fall[:, i] & first_episode[:, i])
        first_failure_s.append(float(age[indices[0], i]) if len(indices) else None)
        values = error[valid[:, i], i]
        per_env_rmse.append(float(np.rad2deg(np.sqrt(np.square(values).mean())))
                            if len(values) and np.isfinite(values).all() else None)
    accepted = bool(finite and complete.all() and np.isfinite(counts).all()
                    and (counts <= budget).all() and rmse <= 5 and peak <= 10)
    report = dict(
        evaluation_metric_scope='Initial episode only, including first terminal sample; post-reset episodes excluded',
        angle_rmse_deg=rmse, peak_error_deg=peak, valid_tracking_samples=sample_count,
        survival_fraction=float(survived.mean()), completed_fraction=float(complete.mean()),
        max_liftoffs_per_foot=counts.max(axis=0).tolist(),
        liftoffs_per_environment=counts.tolist(), angle_rmse_per_environment_deg=per_env_rmse,
        first_failure_s=first_failure_s, initial_episode_duration_s=first_age.tolist(),
        excluded_post_reset_samples=int((~first_episode).sum()),
        finite_tracking_data=finite, accepted=accepted,
        criteria='16 environments, 2 full sine cycles, all survive, RMSE<=5deg, peak<=10deg, each foot<=%d liftoff in entire episode' % budget)
    return report, first_episode, valid


def signature(path):
    stat=path.stat()
    return [stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev]


def safe_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    return value


def process_receipt(receipt_path,stable_seconds=2.,max_pair_gap=60.):
    receipt_path=Path(receipt_path)
    receipt_before=signature(receipt_path)
    receipt_bytes=receipt_path.read_bytes()
    receipt=json.loads(receipt_bytes)
    run=Path(receipt['run']).resolve()
    paths={name:run/name for name in ('evaluation.json','latest_replay.npz','config.json')}
    if not all(path.exists() for path in paths.values()):
        return {'state':'pending','run':str(run),'reason':'evaluation artifacts incomplete'}
    before={name:signature(path) for name,path in paths.items()}
    report_ns=before['evaluation.json'][1];replay_ns=before['latest_replay.npz'][1]
    pair_gap=(replay_ns-report_ns)/1e9
    if pair_gap<0 or pair_gap>max_pair_gap:
        return {'state':'pending','run':str(run),'reason':'report/replay publication timestamps do not match','pair_gap_s':pair_gap}
    if time.time()-max(report_ns,replay_ns)/1e9<stable_seconds:
        return {'state':'pending','run':str(run),'reason':'artifacts have not settled'}
    payload={name:path.read_bytes() for name,path in paths.items()}
    if before!={name:signature(path) for name,path in paths.items()} or receipt_before!=signature(receipt_path):
        return {'state':'pending','run':str(run),'reason':'artifacts changed while reading'}
    report=json.loads(payload['evaluation.json']);cfg=json.loads(payload['config.json'])
    checkpoint=report.get('checkpoint_iteration')
    if not isinstance(checkpoint,int) or isinstance(checkpoint,bool) or checkpoint<0:
        return {'state':'pending','run':str(run),'reason':'report checkpoint identity missing'}
    provenance={name:dict(path=str(paths[name]),sha256=hashlib.sha256(raw).hexdigest(),
                          size=before[name][0],mtime_ns=before[name][1],stat_signature=before[name])
                for name,raw in payload.items()}
    output=run/'corrected_evaluation.json'
    if output.exists():
        try:
            old=json.loads(output.read_bytes())
            if old.get('postprocessor_schema')==SCHEMA and old.get('source_checkpoint_iteration')==checkpoint and old.get('source_files')==provenance:
                return {'state':'unchanged','run':str(run),'checkpoint':checkpoint}
        except (OSError,ValueError):pass
    with np.load(io.BytesIO(payload['latest_replay.npz']),allow_pickle=False) as archive:
        required=('age','fall','error','count')
        data={key:archive[key] for key in required}
        embedded_cp=int(np.asarray(archive['checkpoint_iteration']).item()) if 'checkpoint_iteration' in archive else None
    if embedded_cp is not None and embedded_cp!=checkpoint:
        return {'state':'pending','run':str(run),'reason':'embedded replay checkpoint disagrees with report'}
    # The current artifact has no embedded checkpoint. Record bounded timestamp
    # pairing explicitly instead of claiming content-level checkpoint proof.
    amplitude=float(report.get('amplitude_deg',cfg['amplitude_deg']))
    budget=int(report.get('liftoff_budget',cfg['liftoff_budget_per_foot_per_episode']))
    settle=1.;dt=.02
    episode=settle+4*math.pi*math.radians(amplitude)/math.radians(float(cfg.get('peak_rate_deg_s',10)))
    corrected,_,_=summarize_yaw_evaluation(data,budget,settle,episode,dt)
    # A failed original acceptance can never become a pass through postprocessing.
    corrected['recomputed_first_episode_accepted']=corrected['accepted']
    corrected['accepted']=bool(report.get('accepted',False) and corrected['accepted'])
    result=dict(report)
    result.update(corrected)
    result.update(postprocessor_schema=SCHEMA,source_run=str(run),source_checkpoint_iteration=checkpoint,
                  source_files=provenance,source_receipt=dict(path=str(receipt_path),sha256=hashlib.sha256(receipt_bytes).hexdigest(),stat_signature=receipt_before),
                  provenance=dict(source_report_sha256=provenance['evaluation.json']['sha256'],
                      source_replay_sha256=provenance['latest_replay.npz']['sha256'],
                      source_report_mtime_ns=report_ns,source_replay_mtime_ns=replay_ns),
                  original_report=report,original_report_text=payload['evaluation.json'].decode('utf-8'),
                  source_pair_validation=dict(report_precedes_replay=True,publish_gap_s=pair_gap,
                      max_publish_gap_s=max_pair_gap,stable_seconds=stable_seconds,unchanged_during_read=True,
                      embedded_checkpoint_verified=embedded_cp is not None,
                      matching_method='embedded_checkpoint_and_stable_timestamps' if embedded_cp is not None else 'stable_ordered_publication_timestamps; replay_has_no_embedded_checkpoint'),
                  amplitude_deg=amplitude,liftoff_budget=budget,episode_s=episode,control_dt_s=dt,
                  created_unix=time.time())
    if before!={name:signature(path) for name,path in paths.items()} or receipt_before!=signature(receipt_path):
        return {'state':'pending','run':str(run),'reason':'artifacts changed before publication'}
    temporary=output.with_name(output.name+'.%s.tmp'%os.getpid())
    temporary.write_text(json.dumps(safe_json(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,output)
    return {'state':'written','run':str(run),'checkpoint':checkpoint,'accepted':result['accepted'],
            'angle_rmse_deg':result['angle_rmse_deg'],'first_episode_samples':result['valid_tracking_samples']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=30.)
    parser.add_argument('--stable-seconds',type=float,default=2.)
    parser.add_argument('--max-pair-gap',type=float,default=60.)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.interval<=0 or args.stable_seconds<0 or args.max_pair_gap<=0:parser.error('invalid timing limits')
    previous=None
    while True:
        try:status=process_receipt(args.receipt,args.stable_seconds,args.max_pair_gap)
        except (OSError,ValueError,KeyError,TypeError,EOFError,zipfile.BadZipFile) as exc:status={'state':'pending','reason':type(exc).__name__+': '+str(exc)}
        if status!=previous:print(json.dumps(status,ensure_ascii=False),flush=True);previous=status
        if args.once:return
        time.sleep(args.interval)


if __name__=='__main__':main()
