"""Export a read-only dashboard snapshot from local evidence; no simulation."""
import json,time,hashlib
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym')
OUT=ROOT/'results/legged_gym'
def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default
def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
run=Path(read(ROOT/'runs/current.json')['run_dir'])
state=read(run/'status.json',{});state['stale']=time.time()-state.get('updated_unix',0)>180
data={'/api/status':{'run':state,'checkpoint':read(run/'checkpoint.json'),'evaluation':read(run/'latest_evaluation.json'),
    'best':read(run/'best_evaluation.json'),'milestones':lines(run/'milestones.jsonl')[-30:],
    'progress':read(run/'progress_5min.json'),'progress_history':lines(run/'progress_5min.jsonl')[-12:],
    'backup':read(run/'backup_receipt.json'),'server_time_unix':time.time()},
    '/api/learning-curve':lines(run/'metrics.jsonl')[-5000:],
    '/api/feedback':{'messages':lines(OUT/'operator_feedback.jsonl')[-500:],'storage_path':str(OUT/'operator_feedback.jsonl'),
        'monitor':read(OUT/'operator_monitor_status.json',{}),'server_time_unix':time.time()}}
html=(ROOT/'deployment/legged_gym_knee_termination_20261025/dashboard.html').read_text()
begin=html.index('async function get(path)');end=html.index('\nfunction setup',begin)
encoded=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
html=html[:begin]+'const OFFLINE_DATA='+encoded+';\nasync function get(path){return OFFLINE_DATA[path]}'+html[end:]
html=html.replace('<title>鼬科机器人 · 训练反馈</title>','<title>鼬科机器人 · 离线反馈快照</title>')
html=html.replace('<textarea id="feedback-input"','<textarea disabled id="feedback-input"').replace('<button id="feedback-send"','<button disabled id="feedback-send"')
html=html.replace('记录观察、改善建议或调整要求','离线快照只读；留言请打开 11111 端口实时页面')
html=html.replace('每 10 秒刷新；过期状态会明确标注。','离线快照生成于 '+time.strftime('%Y-%m-%d %H:%M:%S')+'，数据不会继续更新。')
output=OUT/'offline_feedback_20261025_1900.html'
tmp=output.with_suffix('.partial');tmp.write_text(html);tmp.replace(output)
receipt={'created_unix':time.time(),'run':run.name,'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
    'bytes':output.stat().st_size,'status':state.get('status'),'iteration':state.get('iteration'),
    'feedback_read_only':True,'scope':'Server-local offline evidence snapshot; no automatic Windows transfer'}
(OUT/'offline_feedback_20261025_1900_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False))
