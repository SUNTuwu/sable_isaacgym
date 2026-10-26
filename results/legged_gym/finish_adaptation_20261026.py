"""Publish final status only after actual stop, verified archives and final video."""
import fcntl,hashlib,json,sys,time
from pathlib import Path
ROOT=Path('/root/gpufree-data/projects/sable-isaacgym');OUT=ROOT/'results/legged_gym';POST=ROOT/'postrun/adaptation_speedtrack20_20261026'
def read(p):return json.loads(p.read_text())
def atomic(p,d):
 t=p.with_suffix('.partial');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=read(OUT/'adaptation_round_plan_20261026.json');run=Path(plan['run']);verified=read(POST/'training_archives_verified.json')
 assert verified['passed'] and all(s['passed'] for s in verified['stages'])
 final=verified['stages'][-1];assert final['run']==run.name
 baseline=read(run/'baseline_evaluation.json')
 complete=read(run/'completion.json');ev=read(Path(complete['final_evaluation']['report']));video=read(OUT/'side_video_preview.json')
 assert complete['reason']=='deadline' and complete['training_exit_code']==0
 assert video['run']==run.name and video['iteration']==ev['iteration'] and video['checkpoint_sha256']==ev['checkpoint_sha256']
 assert sha(Path(video['path']))==video['sha256']
 step=read(OUT/'speed_step_preview_20261026.json');assert step['run']==run.name and step['iteration']==ev['iteration']
 assert sha(Path(step['report']))==step['report_sha256']
 assert step.get('heading_control',{}).get('step_command_update')=='preserve_heading_feedback'
 evidence=read(OUT/'speedtrack_evidence_20261026.json');assert any(r['kind']=='final' and r['iteration']==ev['iteration'] for r in evidence['reports'])
 receipt={'created_unix':time.time(),'deadline':plan['deadline'],'run':run.name,'training_stopped':True,'automatic_resume':False,
   'final_iteration':ev['iteration'],'updates':final['updates'],'total_environment_steps':final['total_environment_steps'],
   'checkpoint_sha256':ev['checkpoint_sha256'],'retained_transfer_baseline':{'checkpoint':baseline['checkpoint'],'checkpoint_sha256':baseline['checkpoint_sha256'],'summary':baseline['summary'],'stride':baseline.get('stride_progress_diagnostics'),'scope':'Transfer baseline with full reference; not newly learned performance.'},'summary':ev['summary'],'all20':ev['all20_diagnostics'],
   'priority':ev['priority_diagnostics']['median_across_environments'],'stride_progress':ev.get('stride_progress_diagnostics'),'velocity_tracking':ev['velocity_tracking'],'speed_step_report':step['report'],'speed_step_heading_control':step['heading_control'],'speed_step_summary':step['summary'],'speed_step_priority':step.get('priority_diagnostics'),
   'video':video,'stages':[{'run':s['run'],'archive':s['archive'],'verified_files':s['exact_files_verified']} for s in verified['stages']],
   'front_floor_mu':plan['front_floor_mu'],'hip_posture_penalty_weight':plan.get('hip_posture_penalty_weight'),'hip_preferred_max_deg':plan.get('hip_preferred_max_deg'),'stride_reward_weight':plan.get('stride_reward_weight'),'physical_hip_limits_deg':[-85,24],
   'power_reward_enabled':False,'force_measurement_validated':False,
   'goal_assessment':{'all_completed_swing_peaks_above20mm':all(v==1. for v in ev['all20_diagnostics']['completed_swing_peak_above20mm_fraction_per_foot']),'hip_medians_le_minus9':all(v<=-9. for v in ev['priority_diagnostics']['median_across_environments']['hip_angle_median_deg']),'hind_stride_improved_vs_transfer_baseline':all(v>=b for v,b in zip(ev['stride_progress_diagnostics']['mean_forward_stride_m_per_foot'][2:],baseline['stride_progress_diagnostics']['mean_forward_stride_m_per_foot'][2:])),'full_acceptance':False},
   'scope':'Simulation only; verified server-local training archives. No Windows transfer or hardware validation claimed.'}
 assert not (POST/'completion_receipt.json').exists(),'Final receipt already exists'
 atomic(POST/'completion_receipt.json',receipt);atomic(OUT/'completion_receipt_speedtrack20_20261026.json',receipt)
 for name in ['adaptation_round_plan_20261026.json','active_training_plan.json']:
  p=OUT/name;d=read(p);d.update(status='complete',stage_status='complete',completed_unix=time.time(),final_iteration=ev['iteration'],completion_receipt=str(POST/'completion_receipt.json'),automatic_resume=False);atomic(p,d)
 p=OUT/'next_training_objectives.json';d=read(p);d['updated_unix']=time.time();d['status'].update(training='Stopped at02:00 and archives verified. No automatic restart.',video='Final current-stage checkpoint video published; one unselected sample per speed.',higher_clearance='Final all-feet20mm success fraction='+str(ev['all20_diagnostics']['four_feet_2steps_forward_survive_fraction']),remaining='Front friction remains0.02; zero-friction and reliable per-foot work measurement remain unvalidated. Broader robustness and hardware tests not completed.');atomic(p,d)
 sys.path.insert(0,str(ROOT/'deployment/legged_gym_knee_termination_20261025'))
 from operator_feedback import append_message
 s=ev['summary'];a=ev['all20_diagnostics'];v=ev['velocity_tracking']['groups']
 text='本轮已按02:00停止PPO并完成最终评估与归档核验。最终第%d轮（本阶段新增%d次）：24环境16秒存活%d/24，四足>20mm有效迈步并前进%d/24，膝部触地%d/24。'%(ev['iteration'],final['updates'],round(s['survival_16s']*24),round(a['four_feet_2steps_forward_survive_fraction']*24),round(s['knee_ground_failure_fraction']*24))
 text+='髋实际角度中位（左/右）：'+('/'.join('%.1f°'%v for v in ev['priority_diagnostics']['median_across_environments']['hip_angle_median_deg']))+'。'
 if ev.get('stride_progress_diagnostics'):text+='四足有效向前步长均值（左前/右前/左后/右后）：'+('/'.join('%.1f'% (v*1000) for v in ev['stride_progress_diagnostics']['mean_forward_stride_m_per_foot']))+'mm。'
 text+='匀速实测：'+'；'.join('指令%.3f→实际均速%.3fm/s，RMSE%.3f'%(g['command_forward_m_s'],g['mean_actual_velocity_xyz_m_s'][0],g['rmse_velocity_xyz_m_s'][0]) for g in v)+'。'
 text+='最终24秒变速/停车诊断（已修正为保留航向反馈）%d/24完整存活；网页可切换查看响应曲线，最终检查点侧面视频也已更新。'%(round(step['summary']['survival_16s']*24))
 text+='变速测试转向绝对角中位%.2f°，仍有方向偏移。髋姿态、后足大步长与速度跟随仍未达到期望，较好的迁移初值也已保留。'%step['summary']['median_abs_heading_change_deg']
 text+='前足摩擦仍为0.02；髋角≤−9°不受姿态惩罚，超过后指数扣分，并奖励有效向前步长；物理限位−85°～+24°保留。没有加入功率奖励，测力尚未通过校准。全部为仿真结果，归档当前保存在服务器，未声称已回传Windows。本轮监控结束，无自动续训；后续留言仍会保存。'
 append_message('assistant',text,request_id='speedtrack20-final-verified-20261026')
 with (OUT/'operator_monitor_status.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);m=read(OUT/'operator_monitor_status.json');m.update(finished=True,finished_unix=time.time(),last_checked_unix=time.time(),completion_receipt=str(POST/'completion_receipt.json'))
  m['batch'].update(status='complete',updated_unix=time.time(),speed_step=step);atomic(OUT/'operator_monitor_status.json',m)
 print(json.dumps({'complete':True,'iteration':ev['iteration'],'receipt':str(POST/'completion_receipt.json')},ensure_ascii=False))
if __name__=='__main__':main()
