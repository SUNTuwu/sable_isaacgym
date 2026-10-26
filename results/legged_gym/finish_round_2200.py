"""Assemble postrun evidence without changing the sealed PPO run or weights."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/root/gpufree-data/projects/sable-isaacgym')
OUT = ROOT / 'results/legged_gym'
NAME = 'stand5_20261025_v22_highlift_speed'
POST = ROOT / 'postrun' / NAME
VIDEO = OUT / 'videos/v22_5314_final_360p24'
sys.path.insert(0, str(ROOT / 'deployment/legged_gym_knee_termination_20261025'))
from operator_feedback import append_message, messages
from run_io import atomic_json


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = read(OUT / 'completion_receipt_20261025_2200.json')
video = read(VIDEO / 'video_receipt.json')
recording = read(VIDEO / 'recording_receipt.json')
checkpoint = Path(receipt['archive']['checkpoint']['path'])
assert receipt['passed'] and receipt['final_iteration'] == 5314
assert sha(checkpoint) == video['checkpoint_sha256'] == receipt['archive']['checkpoint']['sha256']
assert sha(Path(video['path'])) == video['sha256']
assert sha(OUT / 'record_side_video_360p.py') == recording['recorder_sha256']
assert sha(OUT / 'publish_side_video.py') == video['packager_sha256']
assert video['probe']['nb_read_frames'] == '768'
assert all(not item['fell'] for item in recording['videos'])

plan = read(OUT / 'active_training_plan.json')
plan.update(updated_unix=time.time(), stage_status='complete', final_iteration=5314,
            completion_receipt='results/legged_gym/completion_receipt_20261025_2200.json',
            automatic_resume=False)
atomic_json(OUT / 'active_training_plan.json', plan)
objectives = read(OUT / 'next_training_objectives.json')
objectives['interpretation'] = ('Each foot swing is assessed separately. Actual capsule-bottom peak above10mm '
                               'has no60ms condition; strict steps retain the separate2mm/60ms/5mm criterion.')
objectives['updated_unix'] = time.time()
objectives['status'] = {
    'video': 'Final5314 side video published:640x360,24fps,32s; .03/.09m/s unselected samples, both survived.',
    'higher_clearance': 'Final5314 completed-swing peaks>10mm: LF446/446, RF444/444, LH444/448, RH440/442.',
    'pair_peak_symmetry': 'Final5314 corresponding-peak RMS: front0.20554mm, hind0.57483mm.',
    'speed_stride': 'Final5314:24/24 survived16s and met four-foot forward criterion at .03/.06/.09m/s; actual median speeds .02815/.05836/.08698m/s.',
    'training': 'Stopped at22:00; final archive and checkpoint verified. No automatic restart.',
    'remaining': 'Low-speed hind clearance occasionally<10mm; hip time in +/-5deg band about79/83%; hind foot travel smaller than front; nominal simulation only.',
    'web_access': 'Local loopback and container interface return200; authenticated browser/video pass. User external URL/error still needed.'
}
atomic_json(OUT / 'next_training_objectives.json', objectives)
append_message('assistant',
    '本轮已于22:00停训，最终5314轮新增314次PPO更新，完整训练归档和最终权重校验通过。'
    '最终24环境、16秒独立回放：24/24站稳并达到四足前进判据，膝触地0；髋中位角约-12.0/-12.3°，'
    '肘活动范围约26°。完整摆动峰值>10mm比例：前足100%/100%，后足99.1%/99.5%；'
    '后足仍有6/890次不足10mm，不能称全部达标。左右峰值差RMS前/后约0.21/0.57mm。'
    '0.03/0.06/0.09m/s指令的实际净速度中位约0.028/0.058/0.087m/s。'
    '\n网页视频已更新为最终5314轮，前16秒0.03m/s、后16秒0.09m/s，360p24fps正常速度，两个未筛选单样本均未跌倒；视频不代替24环境评估。'
    '\n训练归档在服务器数据盘；没有自动复制到Windows。当前PPO已停止，留言服务保持运行。'
    '服务器内11111登录、数据、视频播放检查正常；用户报告的外部访问问题仍待具体网址和报错定位，不声称已经修复。'
    '本轮定期训练监控结束，后续留言仍保存，需下一次活动会话读取。',
    reply_to=objectives['user_message_id'], request_id='final5314-results-and-video')
rows = messages()
answered = {r.get('reply_to') for r in rows if r['role'] == 'assistant'}
waiting = [r for r in rows if r['role'] == 'user' and r['id'] not in answered]
assert not waiting, 'New operator instructions require review before finalizing'
atomic_json(OUT / 'operator_monitor_status.json', {
    'last_checked_unix': time.time(), 'pending_count': 0, 'active_until': plan['deadline'],
    'finished': True, 'reason': 'PPO stopped; final replay, backup and video complete; no unattended assistant reader.'})

POST.mkdir(parents=True, exist_ok=True)
files = ['completion_receipt_20261025_2200.json', 'verify_round_completion.py',
         'round_acceptance_comparison.json', 'round_acceptance_comparison.png',
         'round_acceptance_comparison_receipt.json', 'summarize_final_round.py',
         'active_training_plan.json', 'next_training_objectives.json',
         'operator_monitor_status.json', 'operator_feedback.jsonl',
         'record_side_video_360p.py', 'publish_side_video.py', 'check_video_browser.cjs',
         'side_video_preview.json', 'finish_round_2200.py']
for name in files:
    shutil.copy2(OUT / name, POST / name)
shutil.copytree(OUT / 'video_ui', VIDEO / 'ui_validation', dirs_exist_ok=True)
for name in ['record_side_video_360p.py', 'publish_side_video.py']:
    shutil.copy2(OUT / name, VIDEO / name)
shutil.copy2(ROOT / 'runs' / NAME / 'final_verification.json', POST / 'final_verification.json')
(POST / 'HANDOFF.txt').write_text(
    'Final run: ' + NAME + '\n'
    'Stopped2026-10-25T22:00:02+08:00, iteration5314,314new PPO updates. No automatic restart.\n'
    'Training archive: ' + receipt['archive']['archive'] + '\n'
    'Archive SHA256: ' + receipt['archive']['sha256'] + '\n'
    'Final checkpoint SHA256: ' + sha(checkpoint) + '\n'
    'Final independent evidence: runs/' + NAME + '/evaluations/final_005314.json\n'
    'Video and this postrun evidence were created after the sealed training archive and are separate.\n'
    'Offline HTML uses a relative videos/... MP4 path: keep that directory with the HTML.\n'
    'All copies are on the remote data disk; no Windows copy is claimed.\n'
    'Internal dashboard access is verified; external user access still awaits URL/error diagnosis.\n'
    'See next_training_objectives.json for remaining low-speed clearance/posture/stride limitations.\n')
print(json.dumps({'postrun': str(POST), 'final_checkpoint_verified': True, 'feedback_pending': len(waiting)}))
