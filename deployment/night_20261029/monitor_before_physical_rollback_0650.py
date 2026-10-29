"""15-minute per-teacher review. Decisions are logged with physical evidence."""
import json,time,math,statistics,datetime,subprocess,hashlib,sys
from pathlib import Path
from translation_failure_adjustments import translation_failure_adjustments
P=Path('/root/gpufree-data/projects/sable-isaacgym');D=P/'deployment/night_20261029'
END=datetime.datetime(2026,10,29,8,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}

def read_yaw_evaluation(run):
    # Read source bytes once: checkpoint identity and SHA must describe this exact report.
    try:
        raw=(run/'evaluation.json').read_bytes();original=json.loads(raw)
    except (OSError,ValueError):return {}
    if not isinstance(original,dict):return {}
    corrected=read(run/'corrected_evaluation.json')
    if not isinstance(corrected,dict):return original
    provenance=corrected.get('provenance')
    checkpoint=original.get('checkpoint_iteration')
    if (checkpoint is not None and corrected.get('source_checkpoint_iteration')==checkpoint
            and isinstance(provenance,dict)
            and provenance.get('source_report_sha256')==hashlib.sha256(raw).hexdigest()):
        return corrected
    return original
def write(p,v):
    f=p.with_suffix('.tmp');f.write_text(json.dumps(v,ensure_ascii=False,indent=2));f.replace(p)
def alive(pid,run):
    try:return str(run) in Path('/proc',str(pid),'cmdline').read_bytes().decode()
    except OSError:return False
def median(xs):return statistics.median(xs) if xs else None

def adjusted_multiplier(ident,cfg,key,current,factor):
    # This turn potential can refund at most 12*w at terminal (T=1, Phi>=-2).
    # Keep that single positive refund below the physical-fall base cost of 20.
    capped=(ident in ('04_TURN_LEFT','05_TURN_RIGHT') and key=='v2_yaw'
            and cfg.get('reward_profile')=='cycle_net_progress_gamma099_v1')
    return round(max(.25,min(1.2 if capped else 4.,current*factor)),4)
def basic(e,turn):
    if not e:return False
    cmd=e.get('mean_command',[0,0,0]);actual=e.get('mean_actual',[0,0,0])
    if e.get('survival_fraction',0)<.95:return False
    if turn:return actual[2]*cmd[2]>0 and .6*abs(cmd[2])<=abs(actual[2])<=1.4*abs(cmd[2]) and math.hypot(*actual[:2])<.03 and e.get('max_turn_radius_m',99)<=.20
    speed=math.hypot(*cmd[:2]);progress=sum(a*c for a,c in zip(actual[:2],cmd[:2]))/max(speed*speed,1e-6)
    return progress>=.6 and e.get('max_abs_final_heading_deg',99)<=15 and e.get('maximum_cross_track_m',99)<=.10 and e.get('yaw_rmse_rad_s',99)<=.25 and e.get('xy_rmse_m_s',99)<=max(.025,.6*speed)
def objective_epoch(settings, amplitude=None, reload_epoch=None):
    objective={k:v for k,v in settings.items() if k not in ('reload','reason')}
    payload=dict(objective=objective,amplitude=round(float(amplitude),6) if amplitude is not None else None,reload_epoch=reload_epoch)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]

def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def finite_vector(value,size=3):
    return isinstance(value,(list,tuple)) and len(value)>=size and all(finite_number(x) for x in value[:size])

def yaw_comparable_rows(rows,metrics):
    # A moving exploration ceiling changes the sampled objective. Compare only
    # the stationary window after the ramp, rather than calling it a plateau.
    schedule=metrics.get('exploration_schedule') or {}
    if schedule and not schedule.get('complete',False):return []
    settled=schedule.get('completed_iteration')
    return [row for row in rows if row['iteration']>=settled] if settled is not None else rows

def yaw_exploration_decision(settings,metrics,evaluation):
    """Stage noise down before increasing rewards; never raise an existing cap."""
    if evaluation.get('survival_fraction',0)<.95:
        return dict(action='retain_survival_priority',reason='survival gate has not passed')
    if evaluation.get('accepted'):
        return dict(action='retain_passing',reason='yaw acceptance already passed')
    if not finite_number(evaluation.get('angle_rmse_deg')):
        return dict(action='await_valid_yaw_evaluation',reason='no valid initial-episode angle metric')
    if not metrics.get('exploration_control_supported'):
        return dict(action='await_yaw_exploration_runtime',reason='current runtime does not acknowledge live exploration ceilings')
    schedule=metrics.get('exploration_schedule') or {}
    if schedule and not schedule.get('complete',False):
        return dict(action='await_yaw_std_anneal',reason='exploration ceiling is still changing')
    if schedule.get('completed_iteration') is not None and (evaluation.get('checkpoint_iteration') is None or evaluation['checkpoint_iteration']<schedule['completed_iteration']):
        return dict(action='await_post_anneal_evaluation',reason='latest held-out checkpoint predates the settled ceiling')
    std=metrics.get('exploration_std_active_deg',metrics.get('exploration_std_deg'))
    if not finite_vector(std,14):return dict(action='await_yaw_std_telemetry',reason='effective 14-joint exploration is unavailable')
    effective=max(std[:14]);configured=(settings.get('exploration') or {}).get('max_std_deg')
    if finite_number(configured) and configured<=1. and effective<=1.05:
        return dict(action='ready_for_angle_tuning',reason='noise reduction has settled and a new held-out evaluation remains insufficient')
    target=1.5 if effective>1.55 and (not finite_number(configured) or configured>1.5) else 1.
    if finite_number(configured):target=min(target,configured)
    return dict(action='anneal_yaw_exploration',reason='stable survival with insufficient tracking; reduce sampling noise before changing rewards',
                exploration=dict(max_std_deg=target,anneal_iterations=40),effective_max_std_deg=effective)

def reward_window_trend(rows):
    """Compare segment medians within one run/objective; negative reward alone is irrelevant."""
    if not rows or len({(x.get('run'),x.get('revision')) for x in rows})!=1:return None
    if any(not finite_number(x.get('reward')) or not finite_number(x.get('time')) for x in rows):return None
    rows=sorted({x['iteration']:x for x in rows}.values(),key=lambda x:x['time'])
    if len(rows)<9 or rows[-1]['time']-rows[0]['time']<300:return None
    third=len(rows)//3
    a=median([x['reward'] for x in rows[:third]])
    b=median([x['reward'] for x in rows[third:2*third]])
    c=median([x['reward'] for x in rows[2*third:]])
    scale=max(abs(a),abs(b),abs(c))
    segment_noise=max(.0001,.0025*scale)
    cumulative_floor=max(.002,.10*abs(a))
    negative=(a-b>segment_noise and b-c>segment_noise and a-c>=cumulative_floor)
    flat=not negative and max(a,b,c)-min(a,b,c)<=max(.005,.05*scale)
    return dict(reward_medians=[a,b,c],sustained_negative=negative,plateau=flat,
                reward_total_decline=a-c,negative_segment_noise=segment_noise,
                negative_cumulative_threshold=cumulative_floor)

def severe_evaluation(e, turn=False, yaw=False, reference=None):
    if not e:return False
    if e.get('survival_fraction',0)<.75:return True
    if yaw:return False  # Yaw angle/lift failures enter task tuning, not an invented hard reset threshold.
    actual=e.get('mean_actual',[0,0,0]);command=e.get('mean_command',[0,0,0])
    if turn and actual[2]*command[2]<0:return True
    # Failing acceptance is not regression: an imperfect baseline must be allowed to improve.
    if not reference:return False
    def worse(key,floor,delta):
        old=reference.get(key);current=e.get(key)
        return old is not None and current is not None and current>floor and current>max(old*1.5,old+delta)
    if turn:return worse('max_turn_radius_m',.20,.10)
    return worse('max_abs_final_heading_deg',30,10) or worse('maximum_cross_track_m',.15,.05) or (
        abs(actual[2])>.2 and abs(actual[2])>max(reference.get('absolute_mean_yaw_rate',abs(actual[2]))*1.5,reference.get('absolute_mean_yaw_rate',abs(actual[2]))+.1))

def evaluation_reference(ident,run,e,references,baseline_data=None):
    item=references.setdefault(ident,{})
    if item.get('run')!=run:
        item.update(run=run,current_run_initial=dict(e))
    item.setdefault('initial',dict(e));best=item.setdefault('best',{})
    source=(baseline_data or {}).get('teachers',baseline_data or {}).get(ident,{})
    trusted=source.get('evaluation',source.get('reference',{}));proxies=set(source.get('proxy_metric_keys',[]))
    frozen=item.setdefault('frozen_baseline',{})
    for key,value in trusted.items():
        if key not in proxies and (finite_number(value) or (key=='mean_actual' and finite_vector(value))):frozen.setdefault(key,value)
    if finite_vector(frozen.get('mean_actual')):frozen.setdefault('absolute_mean_yaw_rate',abs(frozen['mean_actual'][2]))
    if source:item.update(baseline_provenance=source.get('provenance'),proxy_metric_keys=sorted(proxies))
    reference=dict(item.get('current_run_initial',item['initial']));scopes={key:'current_run_first_evaluation_fallback' for key in reference}
    for key,value in frozen.items():reference[key]=value;scopes[key]='trusted_frozen_baseline'
    for key,value in best.items():
        if key in frozen and finite_number(frozen[key]):reference[key]=min(value,frozen[key]);scopes[key]='frozen_baseline_or_cross_run_best'
        else:reference[key]=value;scopes[key]='cross_run_best_observed'
    item['reference_scopes']=scopes
    return reference

def update_evaluation_reference(ident,e,references,severe):
    if severe:return
    best=references[ident]['best']
    values={key:e[key] for key in ('max_turn_radius_m','max_abs_final_heading_deg','maximum_cross_track_m') if finite_number(e.get(key))}
    if finite_vector(e.get('mean_actual')):values['absolute_mean_yaw_rate']=abs(e['mean_actual'][2])
    for key,value in values.items():best[key]=min(best.get(key,value),value)

def observe_post_reload_anchor(ident,run,e,applied_reloads,anchors):
    """Latch the first proven post-reload held-out report, never an exact-source baseline claim."""
    token=applied_reloads.get(ident)
    if not token:return anchors.get(ident,{}).get('anchor')
    record=anchors.setdefault(ident,{})
    application={'run':str(run),'token':token}
    if record.get('application')!=application:
        try:applied_time=(run/'night_reload_applied.json').stat().st_mtime
        except OSError:return record.get('anchor')
        record.update(application=application,applied_unix=applied_time,captured=False)
    if record.get('captured'):return record.get('anchor')
    applied_time=record['applied_unix'];checkpoint=e.get('checkpoint',{})
    try:
        model=Path(checkpoint['path']).resolve();root=run.resolve()
        report_time=(run/ident/'latest_evaluation.json').stat().st_mtime
        created=e.get('created_unix');iteration=e.get('checkpoint_iteration')
        eligible=(e.get('independent') is True and e.get('num_envs',0)>=4
                  and iteration is not None and checkpoint.get('iteration')==iteration
                  and root in model.parents and model.is_file()
                  and model.stat().st_mtime>=applied_time and report_time>=applied_time
                  and finite_number(created) and created>=applied_time)
    except (OSError,KeyError,TypeError,ValueError):eligible=False
    if not eligible:return record.get('anchor')
    metrics={key:e[key] for key in ('maximum_cross_track_m','max_abs_final_heading_deg','max_turn_radius_m')
             if finite_number(e.get(key)) and e[key]>=0}
    if not metrics:return record.get('anchor')
    record.update(captured=True,anchor=dict(metrics=metrics,run=str(run),token=token,
        checkpoint_path=str(model),checkpoint_iteration=iteration,num_envs=e['num_envs'],
        applied_unix=applied_time,checkpoint_mtime_unix=model.stat().st_mtime,
        evaluation_created_unix=created,evaluation_mtime_unix=report_time,
        scope='first_independent_observation_after_reload; training_may_have_advanced; not_exact_source_checkpoint_evaluation'))
    return record['anchor']

def rollback_reference_floor(ident,reference,baseline_data,anchor,quality_references):
    """Regression may not use an unusually small historic best to reject the fallback policy itself."""
    result=dict(reference);scopes=quality_references[ident].setdefault('reference_scopes',{})
    source=(baseline_data or {}).get('teachers',baseline_data or {}).get(ident,{})
    raw=source.get('evaluation',source.get('reference',{}));floors=[]
    for key in source.get('proxy_metric_keys',[]):
        if key in ('maximum_cross_track_m','max_abs_final_heading_deg','max_turn_radius_m'):
            floors.append((key,raw.get(key),'single_environment_source_proxy_lower_bound; not_multi_environment_maximum'))
    if anchor:
        for key,value in anchor.get('metrics',{}).items():floors.append((key,value,anchor['scope']))
    evidence={}
    for key,value,scope in floors:
        if not finite_number(value) or value<0:continue
        old=result.get(key)
        if not finite_number(old) or value>=old:
            result[key]=value;scopes[key]=scope
        evidence.setdefault(key,[]).append({'value':value,'scope':scope})
    quality_references[ident]['rollback_floor_evidence']=evidence
    return result

def physical_improved(previous,current,turn=False,yaw=False):
    if not previous or not current:return False
    before=previous.get('survival_fraction',0);after=current.get('survival_fraction',0)
    if not finite_number(before) or not finite_number(after):return False
    if after>=before+.05:return True
    if after<before-.02:return False
    def error(e):
        if yaw:return e.get('angle_rmse_deg',e.get('rmse_deg'))
        actual=e.get('mean_actual');command=e.get('mean_command')
        if not finite_vector(actual) or not finite_vector(command):return None
        if turn:
            radius=e.get('max_turn_radius_m')
            return radius/.2+abs(actual[2]-command[2])/max(abs(command[2]),.1) if finite_number(radius) else None
        values=[e.get(k) for k in ('max_abs_final_heading_deg','maximum_cross_track_m','xy_rmse_m_s')]
        return values[0]/30+values[1]/.15+values[2]/max(math.hypot(*command[:2]),.02) if all(finite_number(v) for v in values) else None
    old=error(previous);new=error(current)
    return finite_number(old) and finite_number(new) and old>1e-6 and new<old*.9

def turn_failure_adjustments(e,yaw_factor):
    # Survival is handled by the caller before asking which task objective failed.
    actual=e.get('mean_actual');command=e.get('mean_command')
    if not finite_vector(actual) or not finite_vector(command) or abs(command[2])<1e-6:return {}
    target=abs(command[2]);rate=abs(actual[2])
    if actual[2]*command[2]<=0 or rate<.6*target or rate>1.4*target:return {'v2_yaw':yaw_factor}
    radius=e.get('max_turn_radius_m')
    if math.hypot(*actual[:2])>=.03 or (finite_number(radius) and radius>.20):
        return {'v2_cross_motion':1.1,'v2_velocity':1.1}
    return {}

def apply_reward_terms(ident,cfg,factors):
    """Apply only effective multiplier changes; report saturation and known inactive terms."""
    weights=cfg.setdefault('multipliers',{});changes={};blocked={}
    for key,factor in factors.items():
        if 'TURN' in ident and key=='v2_phase_symmetry':
            blocked[key]='term is forced to zero for turns';continue
        current=weights.get(key,1.)
        if not finite_number(current) or not finite_number(factor):
            blocked[key]='nonfinite multiplier/factor';continue
        before=max(.25,min(4.,current));after=adjusted_multiplier(ident,cfg,key,current,factor)
        if abs(after-before)<=1e-12:
            blocked[key]='effective multiplier is unchanged at cap/floor or rounding precision';continue
        weights[key]=after;changes[key]={'before':before,'after':after,'requested_factor':factor}
    return {'changes':changes,'blocked_terms':blocked}

def negative_reward_adjustment(ident,cfg,e,passed):
    """Every negative decision produces a real reward-function delta or an explicit revision request."""
    yaw=ident=='YAW_SINE_STANCE';turn='TURN' in ident;selected={};reason='insufficient physical failure evidence'
    survival=e.get('survival_fraction')
    if finite_number(survival):
        if yaw:
            selected={'smooth':1.2,'feet':.8} if survival<.95 else {'angle':1.2,'rate':1.15}
            reason='independent yaw survival/tracking failure'
        elif survival<.95:
            selected={'v2_contact_timing':.85,'v2_foot_local':.85,'v2_gravity_alignment':1.15}
            reason='survival priority'
        elif not passed:
            selected=turn_failure_adjustments(e,1.2) if turn else translation_failure_adjustments(e,1.2)
            reason='measured turn task failure' if turn else 'measured translation failure; no yaw_sigma change'
        else:
            contact=e.get('contact_phase_agreement');pose=e.get('pose_symmetry_rmse_m');gravity=e.get('gravity_rmse_deg');cadence=e.get('pair_liftoff_count_asymmetry');feet=e.get('foot_rmse_m_per_foot')
            if finite_number(contact) and contact<.9:selected={'v2_contact_timing':1.1};reason='task passes but contact phase fails'
            elif not turn and finite_number(pose) and pose>.01:selected={'v2_phase_symmetry':1.15};reason='task passes but measured mirror pose fails'
            elif finite_number(gravity) and gravity>5:selected={'v2_gravity_alignment':1.15};reason='task passes but posture fails'
            elif finite_number(cadence) and cadence>.10:selected={'v2_contact_timing':1.1};reason='liftoff-count asymmetry proxy fails before foot tightening; not a full per-cycle phase certificate'
            elif finite_vector(feet,4) and max(feet)>.02:selected={'v2_foot_local':1.1};reason='task/posture pass; measured foot error remains'
    proof=apply_reward_terms(ident,cfg,selected)
    fallback=None
    if (turn and 'v2_yaw' in selected and not proof['changes']
            and cfg.get('reward_profile')=='cycle_net_progress_gamma099_v1'
            and finite_number(cfg.get('multipliers',{}).get('v2_yaw',1.))
            and cfg.get('multipliers',{}).get('v2_yaw',1.)>=1.2):
        fallback={'v2_velocity':.8,'v2_cross_motion':.8}
        alternate=apply_reward_terms(ident,cfg,fallback)
        proof['changes'].update(alternate['changes']);proof['blocked_terms'].update(alternate['blocked_terms'])
        reason='yaw multiplier saturated; reduce competing instantaneous translation terms while retaining their .25 floor'
    changed=bool(proof['changes'])
    return dict(action='reward_adjustment' if changed else 'needs_manual_reward_revision',actual_reward_change=changed,
                requires_manual_reward_revision=not changed,reward_changes=proof['changes'],blocked_reward_terms=proof['blocked_terms'],
                proposed_reward_factors=selected,competing_term_fallback=fallback,
                reason=reason if changed else reason+'; no safe effective multiplier change is available')

def record_negative_window(ident,row,negative,e,streaks,reload_epoch,turn=False,yaw=False):
    # Every slope was measured WITHIN one stable objective. Never compare reward levels across objectives.
    scope=json.dumps(dict(run=row['run'],reload=reload_epoch),sort_keys=True)
    item=streaks.setdefault(ident,{})
    if item.get('scope')!=scope:item.clear();item.update(scope=scope,count=0)
    improved=physical_improved(item.get('evaluation'),e,turn,yaw)
    item['count']=(1 if improved else item.get('count',0)+1) if negative else 0
    item.update(evaluation=dict(e),epoch=row['revision'])
    return item['count'],improved

def process_identity(pid):
    try:
        root=Path('/proc')/str(pid);fields=(root/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=fields[19],state=fields[0],command=(root/'cmdline').read_bytes().decode(errors='replace'))
    except (OSError,ValueError,IndexError,TypeError):return None

def launcher_is_live(record):
    identity=process_identity(record.get('pid'))
    return bool(identity and identity['state']!='Z' and identity['start_ticks']==record.get('start_ticks'))

def ack_ready(request, applied_token, applied_time, checkpoint_time, model_time, checkpoint_exists):
    return bool(request and request.get('token') and request.get('token')==applied_token and applied_time is not None and checkpoint_exists
                and checkpoint_time is not None and model_time is not None
                and checkpoint_time>=applied_time and model_time>=applied_time)

def acknowledge_rollbacks(control, bank, yaw_bank):
    acknowledged=[]
    for ident,settings in list(control.get('teachers',{}).items())+[('YAW_SINE_STANCE',control.get('yaw',{}))]:
        request=settings.get('reload')
        if not request:continue
        yaw=ident=='YAW_SINE_STANCE';run=yaw_bank if yaw else bank
        applied_file=run/('reload_receipt.json' if yaw else 'night_reload_applied.json')
        applied=read(applied_file);token=applied.get('token') if yaw else applied.get(ident)
        cp_file=run/'checkpoint.json' if yaw else run/ident/'checkpoint.json';cp=read(cp_file)
        try:
            model=Path(cp['path']).resolve();root=run.resolve()
            ready=ack_ready(request,token,applied_file.stat().st_mtime,cp_file.stat().st_mtime,
                model.stat().st_mtime,model.is_file() and root in model.parents)
        except (OSError,ValueError,KeyError,TypeError):ready=False
        if ready:
            acknowledged.append(dict(teacher=ident,token=token,checkpoint=str(model),checkpoint_iteration=cp.get('iteration')))
            del settings['reload']
    return acknowledged

def recovery_action(record, run, now, trainer_alive=False, launcher_alive=False):
    if trainer_alive:return 'healthy'
    if launcher_alive:return 'wait_launcher'
    if record.get('attempts',0)>=3:return 'exhausted'
    if now<record.get('retry_after',0):return 'cooldown'
    return 'launch'

def find_external_launcher(kind):
    names=['launch_yaw.py'] if kind=='yaw' else ['recover_bank.py','launch_selected.py','launch_audited.py','launch_bank.py','launch_straightfix.py']
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        identity=process_identity(proc.name)
        if identity and identity['state']!='Z' and any(str(D/name) in identity['command'] for name in names):return identity
    return None

def maintain_recovery(kind, proc, terminal, records, now):
    record=records.setdefault(kind,{})
    if now>=END:record['status']='deadline_reached';return
    if proc['alive']:
        # Do not reset the retry budget for a trainer that only survives startup.
        if record.get('verified_run')!=proc['run'] or record.get('status') not in ('healthy','verifying'):
            record.update(status='verifying',verified_run=proc['run'],verified_at=now)
        if now-record.get('verified_at',now)>=300:record.update(status='healthy',attempts=0)
        record.pop('dead_since',None);record.pop('dead_run',None)
        return
    if terminal not in ('failed','completed','stopped_requested','stopped_at_deadline'):return
    if launcher_is_live(record):return
    external=find_external_launcher(kind)
    if external:
        record.update(pid=external['pid'],start_ticks=external['start_ticks'],status='waiting_external_launcher',run=proc['run'])
        return
    receipt=read(P/('deployment/reference_v2_20261028/launch_receipt.json' if kind=='bank' else 'deployment/yaw_sine_20261029/launch_receipt.json'))
    candidate=receipt.get('bank' if kind=='bank' else 'run')
    if candidate and (candidate!=proc['run'] or receipt.get('pid')!=proc.get('pid')) and alive(receipt.get('pid'),candidate):
        record.update(status='waiting_run_publication',candidate_run=candidate,candidate_pid=receipt.get('pid'));return
    if record.get('dead_run')!=proc['run']:
        record.update(dead_run=proc['run'],dead_since=now)
    if now-record['dead_since']<90:
        record['status']='confirming_terminal';return
    result=read(Path(record['result_path'])) if record.get('result_path') else {}
    if record.get('status') in ('launching','waiting_external_launcher'):
        record.update(status='launcher_exited',last_result=result or {'exit_code':None,'reason':'launcher exited without result receipt'},retry_after=now+90)
    action=recovery_action(record,proc['run'],now)
    if action!='launch':
        record['status']='retry_exhausted' if action=='exhausted' else record.get('status','cooldown');return
    script=D/('recover_bank.py' if kind=='bank' else 'launch_yaw.py')
    if not script.is_file():record.update(status='missing_launcher',retry_after=now+90);return
    attempt=record.get('attempts',0)+1;result_path=D/('recovery_%s_%d_%d.json'%(kind,int(now),attempt))
    wrapper=('import subprocess,json,time,sys;from pathlib import Path;'
             'r=subprocess.run([sys.executable,sys.argv[1]]);'
             'Path(sys.argv[2]).write_text(json.dumps({"exit_code":r.returncode,"ended_unix":time.time()}));sys.exit(r.returncode)')
    try:
        with (D/(kind+'_auto_recovery.log')).open('a') as log:
            child=subprocess.Popen([sys.executable,'-c',wrapper,str(script),str(result_path)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except OSError as exc:
        record.update(status='launch_failed',attempts=attempt,last_result={'error':str(exc)},retry_after=now+90);return
    identity=process_identity(child.pid)
    record.update(status='launching',pid=child.pid,start_ticks=identity['start_ticks'] if identity else None,
                  run=proc['run'],attempts=attempt,started=now,result_path=str(result_path),retry_after=now+90)

def epoch_for_sample(ident,settings,amplitude,epochs,reload_epochs,history,bad_windows,physical_regressions,applied_reload=None,run=None):
    token=settings.get('reload',{}).get('token')
    latch=reload_epochs.get(ident,{})
    if not isinstance(latch,dict) or latch.get('run')!=run:latch={'run':run}
    if token:latch['requested']=token  # Keep this latch when the acknowledged request is removed.
    if applied_reload:latch['applied']=applied_reload
    reload_epochs[ident]=latch
    epoch=objective_epoch(settings,amplitude,latch)
    if epochs.get(ident)!=epoch:
        history[ident]=[];bad_windows[ident]=0;physical_regressions[ident]=0;epochs[ident]=epoch
    return epoch

def main():
    saved=read(D/'monitor_memory.json')
    history=saved.get('history',{});bad_windows=saved.get('bad_windows',{});last_eval=saved.get('last_eval',{});physical_regressions=saved.get('physical_regressions',{});last_action=saved.get('last_action',{});reviews=[];last_review=read(D/'monitor_status.json').get('last_review_unix',0);epochs=saved.get('epochs',{});reload_epochs=saved.get('reload_epochs',{});recovery_records=saved.get('recovery_records',{});quality_references=saved.get('quality_references',{});negative_streaks=saved.get('negative_streaks',{});rollback_anchors=saved.get('rollback_anchors',{})
    baseline=read(D/'baseline_sources.json')
    while time.time()<END:
        now=time.time();pointer=read(P/'results/teacher_bank/current.json');bank=Path(pointer.get('bank','/invalid'))
        state=read(bank/'teacher_bank.json');samples={};processes={'bank':dict(run=str(bank),pid=state.get('pid'),alive=alive(state.get('pid'),bank))}
        control=read(D/'control.json');control.setdefault('teachers',{});applied_reloads=read(bank/'night_reload_applied.json')
        for ident in baseline:
            s=read(bank/ident/'status.json');m=s.get('metrics',{});e=read(bank/ident/'latest_evaluation.json')
            reward=m.get('mean_reward_per_control_step')
            if isinstance(reward,(float,int)) and processes['bank']['alive']:
                row=dict(time=now,iteration=s.get('iteration'),reward=reward,revision=epoch_for_sample(ident,control['teachers'].get(ident,{}),None,epochs,reload_epochs,history,bad_windows,physical_regressions,applied_reloads.get(ident),str(bank)),run=str(bank),metrics=m,evaluation=e)
                history.setdefault(ident,[]).append(row);history[ident]=history[ident][-90:];samples[ident]=row
        yr=read(P/'deployment/yaw_sine_20261029/launch_receipt.json');yb=Path(yr.get('run','/invalid'));ys=read(yb/'status.json');ye=read_yaw_evaluation(yb)
        processes['yaw']=dict(run=str(yb),pid=ys.get('pid'),alive=alive(ys.get('pid'),yb),status=ys.get('status'))
        if isinstance(ys.get('mean_reward_per_step'),(float,int)) and processes['yaw']['alive']:
            row=dict(time=now,iteration=ys.get('iteration'),reward=ys['mean_reward_per_step'],run=str(yb),revision=epoch_for_sample('YAW_SINE_STANCE',control.get('yaw',{}),ys.get('amplitude_deg',30),epochs,reload_epochs,history,bad_windows,physical_regressions,read(yb/'reload_receipt.json').get('token'),str(yb)),metrics=ys,evaluation=ye)
            history.setdefault('YAW_SINE_STANCE',[]).append(row);history['YAW_SINE_STANCE']=history['YAW_SINE_STANCE'][-90:];samples['YAW_SINE_STANCE']=row
        # Check each new independent evaluation, not just every reward-review window.
        control=read(D/'control.json');control.setdefault('teachers',{})
        acknowledged=acknowledge_rollbacks(control,bank,yb)
        if acknowledged:
            write(D/'control.json',control)
            with (D/'reload_acknowledgments.jsonl').open('a') as f:f.write(json.dumps(dict(time=now,acknowledged=acknowledged))+'\n')
        physical_baselines=read(D/'baseline_physical.json')
        urgent=[]
        for ident,row in samples.items():
            e=row['evaluation'];evaluation_id=str(row['run'])+':'+str(e.get('checkpoint_iteration',e.get('iteration')))
            if not e or evaluation_id==last_eval.get(ident):continue
            last_eval[ident]=evaluation_id;turn='TURN' in ident;yaw=ident=='YAW_SINE_STANCE'
            actual=e.get('mean_actual',[0,0,0]);command=e.get('mean_command',[0,0,0])
            reference=evaluation_reference(ident,row['run'],e,quality_references,physical_baselines)
            if not yaw:
                anchor=observe_post_reload_anchor(ident,bank,e,applied_reloads,rollback_anchors)
                reference=rollback_reference_floor(ident,reference,physical_baselines,anchor,quality_references)
            severe=severe_evaluation(e,turn=turn,yaw=yaw,reference=reference)
            update_evaluation_reference(ident,e,quality_references,severe)
            physical_regressions[ident]=physical_regressions.get(ident,0)+1 if severe else 0
            if physical_regressions[ident]>=2 and now-last_action.get(ident,0)>600:
                checkpoint=read(D/'yaw_baseline.json') if yaw else baseline[ident]['checkpoint']
                if checkpoint:
                    cfg=control.setdefault('yaw',{}) if yaw else control['teachers'].setdefault(ident,{})
                    if cfg.get('reload'):continue
                    cfg['reload']=dict(checkpoint,token=str(int(now))+'_'+ident);last_action[ident]=now;physical_regressions[ident]=0
                    urgent.append(dict(teacher=ident,action='rollback_to_baseline',reason='two consecutive independent evaluations show physical regression',evaluation=evaluation_id))
        if urgent:
            control['revision']=control.get('revision',0)+1;write(D/'control.json',control)
            with (D/'urgent_actions.jsonl').open('a') as f:f.write(json.dumps(dict(time=now,actions=urgent),ensure_ascii=False)+'\n')
        # Persist launcher identity/results and bounded retries across monitor restarts.
        for key,proc in processes.items():
            terminal=state.get('status') if key=='bank' else ys.get('status')
            maintain_recovery(key,proc,terminal,recovery_records,now)
        if now-last_review>=900:
            control=read(D/'control.json');control.setdefault('teachers',{});decisions=[]
            for ident,row in samples.items():
                rows=[x for x in history[ident] if x['run']==row['run'] and x['revision']==row['revision'] and now-x['time']<=900]
                unique={x['iteration']:x for x in rows};rows=list(unique.values())
                e=row['evaluation'];turn='TURN' in ident;yaw=ident=='YAW_SINE_STANCE';passed=bool(e.get('accepted')) if yaw else basic(e,turn)
                decision=dict(teacher=ident,iteration=row['iteration'],evaluation_iteration=e.get('checkpoint_iteration',e.get('iteration')),basic_pass=passed,action='observe',reason='collecting comparable samples')
                if yaw:rows=yaw_comparable_rows(rows,row['metrics'])
                trend=reward_window_trend(rows)
                if trend is not None:
                    negative=trend['sustained_negative'];flat=trend['plateau']
                    decision.update(trend)
                    bad_windows[ident]=bad_windows.get(ident,0)+1 if negative else 0
                    negative_count,physical_gain=record_negative_window(ident,row,negative,e,negative_streaks,reload_epochs.get(ident),turn,yaw)
                    decision.update(negative_trend_windows=negative_count,physical_improved=physical_gain)
                    cfg=control.setdefault('yaw',{}) if yaw else control['teachers'].setdefault(ident,{})
                    if cfg.get('reload') and not negative:
                        decision.update(action='await_reload',reason='existing rollback is awaiting application/checkpoint acknowledgment');decisions.append(decision);continue
                    weights=cfg.setdefault('multipliers',{})
                    flat_changes={};flat_blocked={}
                    def adjust(key,factor):
                        proof=apply_reward_terms(ident,cfg,{key:factor});flat_changes.update(proof['changes']);flat_blocked.update(proof['blocked_terms'])
                    if not cfg.get('reload') and not yaw and (physical_regressions.get(ident,0)>=2 or (negative_count>=2 and not passed)):
                        cp=baseline[ident]['checkpoint'];cfg['reload']=dict(cp,token=str(int(now))+'_'+ident)
                        decision.update(action='rollback_to_baseline',reason='repeated physical regression or two sustained-negative review windows');physical_regressions[ident]=0;bad_windows[ident]=0
                    if not cfg.get('reload') and yaw and (physical_regressions.get(ident,0)>=2 or (negative_count>=2 and not passed)):
                        cp=read(D/'yaw_baseline.json')
                        if cp:
                            cfg['reload']=dict(cp,token=str(int(now))+'_yaw')
                            decision.update(action='rollback_yaw_baseline',reason='repeated yaw-task regression');physical_regressions[ident]=0;bad_windows[ident]=0
                    if yaw and cfg.get('reload') and not negative:
                        decision['multipliers']=dict(weights);decisions.append(decision);continue
                    if yaw and not passed and flat and e.get('survival_fraction',0)>=.95:
                        exploration_choice=yaw_exploration_decision(cfg,row['metrics'],e)
                        if exploration_choice['action']!='ready_for_angle_tuning':
                            if 'exploration' in exploration_choice:cfg['exploration']=exploration_choice['exploration']
                            decision.update(exploration_choice);decision['multipliers']=dict(weights)
                            decisions.append(decision);continue
                    if negative:
                        prior_action=decision['action'];proof=negative_reward_adjustment(ident,cfg,e,passed)
                        decision.update(proof);decision['action']=prior_action+'+'+proof['action'];decision['rollback_requested']=bool(cfg.get('reload'))
                    elif flat:
                        if not passed:
                            if yaw:
                                if e.get('survival_fraction',0)<.95:
                                    decision.update(action='retain_survival_priority',reason='yaw survival has not passed; do not raise angle reward');decisions.append(decision);continue
                                adjust('angle',1.2)
                            elif e.get('survival_fraction',0)<.95:adjust('v2_contact_timing',.9);adjust('v2_gravity_alignment',1.1)
                            elif turn:
                                selected=turn_failure_adjustments(e,1.15);decision['turn_tuning_terms']=list(selected)
                                for key,factor in selected.items():adjust(key,factor)
                            else:
                                selected=translation_failure_adjustments(e,1.15);decision['translation_tuning_terms']=list(selected)
                                for key,factor in selected.items():adjust(key,factor)
                            decision.update(action='tune_basic',reason='plateau without survival/direction acceptance')
                        elif not yaw:
                            # Symmetry first; foot tightening only after contact-phase and body-posture checks.
                            pose=e.get('pose_symmetry_rmse_m')
                            pose_ok=turn or (pose is not None and pose<=.010)
                            cycle_ok=e.get('pair_liftoff_count_asymmetry',1)<=.10
                            if not pose_ok or not cycle_ok or e.get('contact_phase_agreement',0)<.9 or e.get('gravity_rmse_deg',99)>5:
                                adjust('v2_phase_symmetry',1.15);adjust('v2_contact_timing',1.1);decision.update(action='tune_symmetry',reason='basic pass; prioritize pose and periodic symmetry')
                            elif max(e.get('foot_rmse_m_per_foot',[1]))>.02:
                                adjust('v2_foot_local',1.1);decision.update(action='tune_feet',reason='basic/posture/timing gates pass; tighten foot tracking')
                            else:decision.update(action='retain_passing',reason='physical gates passing; do not change rewards solely to raise return')
                    if flat:
                        decision.update(actual_reward_change=bool(flat_changes),reward_changes=flat_changes,blocked_reward_terms=flat_blocked)
                        if decision['action'].startswith('tune_') and not flat_changes:
                            decision.update(action='tuning_saturated',reason='No effective reward multiplier changed; preserve current settings')
                    decision['multipliers']=dict(weights)
                decisions.append(decision)
            control['revision']=control.get('revision',0)+1;control['last_review_unix']=now;write(D/'control.json',control)
            review=dict(time=now,local_time=datetime.datetime.fromtimestamp(now).isoformat(),processes=processes,decisions=decisions)
            with (D/'reviews.jsonl').open('a') as f:f.write(json.dumps(review,ensure_ascii=False)+'\n')
            reviews.append(review);last_review=now
        write(D/'monitor_status.json',dict(status='monitoring',deadline_unix=END,updated_unix=now,last_review_unix=last_review,next_review_unix=last_review+900,processes=processes,recovery_records=recovery_records,last_review=reviews[-1] if reviews else None))
        write(D/'monitor_memory.json',dict(history=history,bad_windows=bad_windows,last_eval=last_eval,physical_regressions=physical_regressions,last_action=last_action,epochs=epochs,reload_epochs=reload_epochs,recovery_records=recovery_records,quality_references=quality_references,negative_streaks=negative_streaks,rollback_anchors=rollback_anchors))
        time.sleep(30)
    write(D/'monitor_status.json',dict(status='deadline_reached',deadline_unix=END,updated_unix=time.time(),reviews=len(reviews)))


if __name__=='__main__':
    main()
