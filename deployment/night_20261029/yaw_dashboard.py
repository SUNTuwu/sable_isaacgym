"""Read-only projection of the independent yaw task into the shared dashboard."""
import json,time,os,hashlib,math
from pathlib import Path
import numpy as np
P=Path('/root/gpufree-data/projects/sable-isaacgym')
NAMES=['spine_roll','hip_roll','L_shoulder_roll','L_shoulder_pitch','L_elbow_pitch','L_hip_pitch','L_knee_pitch','R_shoulder_roll','R_shoulder_pitch','R_elbow_pitch','R_hip_pitch','R_knee_pitch','L_ankle_pitch','R_ankle_pitch','spine_pitch']
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
def matched_evaluation(run,original):
    """Use the sidecar only while it still describes these exact source files."""
    corrected=read(run/'corrected_evaluation.json')
    if not corrected or corrected.get('postprocessor_schema')!='yaw_first_episode_v1':return original
    if corrected.get('source_checkpoint_iteration')!=original.get('checkpoint_iteration'):return original
    if corrected.get('source_run')!=str(run.resolve()):return original
    try:
        sources=corrected['source_files']
        raw=(run/'evaluation.json').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sources['evaluation.json']['sha256']:return original
        replay=(run/'latest_replay.npz').stat();recorded=sources['latest_replay.npz']
        if replay.st_size!=recorded['size'] or replay.st_mtime_ns!=recorded['mtime_ns']:return original
    except (OSError,KeyError,TypeError):return original
    return corrected
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False));t.replace(p)
last=None
while True:
    try:
        receipt=read(P/'deployment/yaw_sine_20261029/launch_receipt.json');b=Path(receipt['run']);s=read(b/'status.json');cfg=read(b/'config.json');report=matched_evaluation(b,read(b/'evaluation.json'))
        out=b/'dashboard/YAW_SINE_STANCE';metrics={'mean_reward_per_control_step':s.get('mean_reward_per_step'),'angle_rmse_deg':report.get('angle_rmse_deg'),'peak_error_deg':report.get('peak_error_deg'),'survival_fraction':report.get('survival_fraction'),
            'amplitude_deg':cfg.get('amplitude_deg'),'peak_rate_deg_s':10,'period_s':cfg.get('period_s'),'episode_s':cfg.get('episode_s'),'liftoff_budget':cfg.get('liftoff_budget_per_foot_per_episode'),'exploration_std_deg':s.get('exploration_std_deg')}
        write(out/'status.json',dict(teacher_id='YAW_SINE_STANCE',status=s.get('status','starting'),pid=s.get('pid'),iteration=s.get('iteration'),updated_unix=s.get('updated_unix',time.time()),metrics=metrics))
        write(out/'config.json',cfg)
        cp=read(b/'checkpoint.json')
        if cp:
            original=Path(cp['path']);dest=out/'checkpoints'/original.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if original.exists() and not dest.exists():os.link(original,dest)
            if dest.exists():write(out/'checkpoint.json',dict(cp,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        if report:
            write(out/'latest_evaluation.json',dict(report,teacher_id='YAW_SINE_STANCE',iteration=report.get('checkpoint_iteration'),independent=True,num_envs=16,horizon_s=cfg.get('episode_s'),created_unix=(b/'evaluation.json').stat().st_mtime,acceptance_scope='Two full sine cycles, angle RMSE<=5deg, peak<=10deg, all survive, per-foot episode liftoff budget'))
        npz=b/'latest_replay.npz';key=(str(b),npz.stat().st_mtime if npz.exists() else 0)
        if npz.exists() and key!=last and report:
            data=np.load(npz);frames=[];origin=data['root'][0,0,:3].copy();origin[2]=0
            for k in range(0,len(data['q']),2):
                root=data['root'][k,0];frames.append(dict(time_s=k*.02,q=data['q'][k,0].tolist(),root_position=(root[:3]-origin).tolist(),root_quaternion_xyzw=root[3:7].tolist(),toe_world=(data['toe'][k,0]-origin).tolist(),active=not bool(data['fall'][:k+1,0].any()),yaw_target_deg=float(np.rad2deg(data['target'][k,0])),yaw_actual_deg=float(np.rad2deg(data['yaw'][k,0]))))
            path=out/'videos/latest_trace.json';write(path,dict(teacher_id='YAW_SINE_STANCE',iteration=report.get('checkpoint_iteration'),joint_names=NAMES,type='kinematic_trace',frames=frames))
            write(out/'video_receipt.json',dict(type='kinematic_trace',path=str(path),iteration=report.get('checkpoint_iteration'),created_unix=npz.stat().st_mtime,source='Independent deterministic sine tracking evaluation'));last=key
    except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
    time.sleep(5)
