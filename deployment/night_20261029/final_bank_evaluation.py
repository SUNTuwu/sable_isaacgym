"""Isolated 32-environment, 24-second final evaluation; never optimizes a policy.

Run only after the source trainer has exited. --plan performs filesystem checks
without importing Torch/Isaac Gym or creating output. All producer files and the
global current-bank pointer remain unchanged. The frozen source core supplies
the actual controller, rewards, boundaries and existing evaluation metrics.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

IDS=('01_FORWARD_POLICY','04_TURN_LEFT','05_TURN_RIGHT','06_BACKWARD',
     '02_STRAFE_LEFT','03_STRAFE_RIGHT','09_BACKWARD_WALK','10_FORWARD_TROT')
HORIZON=24.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path,root):
    path=Path(path).resolve();root=Path(root).resolve()
    return path==root or root in path.parents


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.%s.tmp'%os.getpid())
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def evaluation_argv(command,source,output):
    """Retain the source launch's actual objective flags, not CLI defaults."""
    if len(command)<2 or Path(command[1]).name!='teacher_bank_core.py':
        raise ValueError('Unrecognized source launch command')
    removed_values={'--bank','--reference','--resume-bank','--deadline','--duration','--envs-per-teacher'}
    removed_flags={'--smoke','--allow-objective-change'}
    result=[];index=2
    while index<len(command):
        arg=command[index]
        if arg in removed_values:
            if index+1>=len(command):raise ValueError('Missing option value: '+arg)
            index+=2;continue
        if arg in removed_flags:index+=1;continue
        if any(arg.startswith(name+'=') for name in removed_values|removed_flags):
            raise ValueError('Unexpected equals-form launch option')
        result.append(arg);index+=1
    return ['final_bank_evaluation_core']+result+['--bank',str(output),'--reference',str(source/'reference'),
            '--resume-bank',str(source),'--envs-per-teacher','0','--duration','0']


class EvaluationOnly(ast.NodeTransformer):
    """Fail closed if a future core no longer matches these audited edits."""
    def __init__(self,source):
        self.source=source;self.functions=[]
        self.counts=dict(training_loop=0,duration_assert=0,horizon_label=0,horizon_steps=0,terminal_trace=0)
    def visit_FunctionDef(self,node):
        self.functions.append(node.name);node=self.generic_visit(node);self.functions.pop();return node
    def visit_While(self,node):
        node=self.generic_visit(node)
        text=''.join((ast.get_source_segment(self.source,node.test) or '').split())
        if self.functions==['main'] and text.startswith('time.time()<deadline'):
            self.counts['training_loop']+=1;node.test=ast.Constant(False)
        return node
    def visit_Assert(self,node):
        text=''.join((ast.get_source_segment(self.source,node.test) or '').split())
        if self.functions==['main'] and text=='deadline>start':
            self.counts['duration_assert']+=1;return ast.copy_location(ast.Pass(),node)
        return self.generic_visit(node)
    def visit_Dict(self,node):
        node=self.generic_visit(node)
        if self.functions==['main','finish_eval']:
            for index,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='horizon_s':
                    if not isinstance(node.values[index],ast.Constant) or node.values[index].value!=16.:
                        raise ValueError('Unknown original evaluation horizon')
                    node.values[index]=ast.Constant(HORIZON);self.counts['horizon_label']+=1
        return node
    def visit_If(self,node):
        node=self.generic_visit(node)
        if self.functions==['main','collect_eval']:
            text=''.join((ast.get_source_segment(self.source,node.test) or '').split())
            if text in ("es['step']>=800",'es["step"]>=800'):
                node.test=ast.parse("es['step']>=env.max_episode_length",mode='eval').body
                self.counts['horizon_steps']+=1
            elif text in ("es['step']%5==0",'es["step"]%5==0'):
                node.test=ast.parse("es['step']%5==0 or es['step']+1>=env.max_episode_length",mode='eval').body
                self.counts['terminal_trace']+=1
        return node


def transform_core(source,filename='<frozen-core>'):
    transformer=EvaluationOnly(source);tree=transformer.visit(ast.parse(source,filename))
    if any(value!=1 for value in transformer.counts.values()):
        raise ValueError('Unsupported frozen core structure: '+str(transformer.counts))
    ast.fix_missing_locations(tree)
    return compile(tree,filename,'exec'),transformer.counts


def source_process_alive(source,bank):
    pid=bank.get('pid')
    if not isinstance(pid,int):return False
    proc=Path('/proc')/str(pid)
    if proc.exists():
        try:
            if proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':return False
            return str(source) in proc.joinpath('cmdline').read_bytes().decode(errors='replace')
        except OSError:return False
    return False


def pin_checkpoint_selection(manifest):
    """Freeze the already-verified selection; never reread mutable receipts."""
    source=Path(manifest['source_bank']).resolve();selection={}
    for selected in manifest['selected_checkpoints']:
        ident=selected['teacher_id'];cp=Path(selected['path']).resolve()
        if ident not in IDS or ident in selection or not inside(cp,source/ident):
            raise ValueError('Invalid frozen teacher selection: '+ident)
        if not cp.is_file() or sha(cp)!=selected['sha256']:
            raise ValueError('Frozen checkpoint changed before evaluation: '+ident)
        selection[ident]=dict(source_teacher_id=ident,source_bank=str(source),
            checkpoint={key:selected[key] for key in ('path','sha256','iteration')},
            reason='Exact checkpoint pinned by final evaluation manifest')
    if set(selection)!=set(IDS):raise ValueError('Frozen selection must contain all eight teachers')
    path=Path(manifest['output'])/'teacher_sources.json'
    atomic_json(path,selection)
    return dict(teacher_sources_path=str(path),teacher_sources_sha256=sha(path),
                checkpoint_selection_mode='manifest_pinned_teacher_sources')


def verify_loaded_selection(manifest,verification):
    expected={item['teacher_id']:item for item in manifest['selected_checkpoints']}
    loaded=verification.get('sources',[])
    if len(loaded)!=len(IDS) or {item.get('teacher_id') for item in loaded}!=set(IDS):
        raise RuntimeError('Runtime did not load the exact eight frozen teachers')
    for item in loaded:
        target=expected[item['teacher_id']]
        if (Path(item.get('path','')).resolve()!=Path(target['path']).resolve()
                or item.get('sha256')!=target['sha256'] or item.get('iteration')!=target['iteration']):
            raise RuntimeError('Runtime checkpoint differs from frozen selection: '+item['teacher_id'])


def prepare(source,output,project,control):
    source=Path(source).resolve();output=Path(output).resolve();project=Path(project).resolve()
    if source.parent!=project/'runs' or not source.name.startswith('teacher_bank_fullref_'):
        raise ValueError('Source must be a teacher bank in this project/runs')
    if inside(output,source) or inside(source,output):raise ValueError('Output overlaps source bank')
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be new or empty')
    bank=read_json(source/'teacher_bank.json');launch=read_json(source/'launch_receipt.json')
    if bank.get('reward_cleanup_profile')!='reference_local_v2':raise ValueError('This final evaluator is audited for reference_local_v2')
    core=source/'runtime/teacher_bank_core.py';text=core.read_text(encoding='utf-8');_,patches=transform_core(text,str(core))
    selected=[]
    for ident in IDS:
        receipt=read_json(source/ident/'checkpoint.json');cp=Path(receipt['path']).resolve()
        if not inside(cp,source/ident) or not cp.is_file() or sha(cp)!=receipt['sha256']:
            raise ValueError('Invalid selected checkpoint: '+ident)
        selected.append(dict(teacher_id=ident,path=str(cp),sha256=receipt['sha256'],iteration=receipt['iteration'],
                             source_receipt_sha256=sha(source/ident/'checkpoint.json'),
                             reference_sha256=sha(source/'reference'/(ident+'.npz'))))
    if len(set(value['iteration'] for value in selected))!=1:raise ValueError('Checkpoint publication incomplete: mixed iterations')
    raw_control=Path(control).read_bytes();frozen_control=json.loads(raw_control)
    for settings in frozen_control.get('teachers',{}).values():settings.pop('reload',None)
    manifest=dict(schema='standalone-final-bank-evaluation/v2',status='prepared',source_bank=str(source),
                  output=str(output),project=str(project),mode='evaluation_only',optimization_steps=0,
                  training_envs=0,total_envs=32,envs_per_teacher=4,horizon_s=HORIZON,evaluation_seed=947,
                  selected_checkpoints=selected,source_bank_metadata_sha256=sha(source/'teacher_bank.json'),
                  frozen_core_sha256=sha(core),ast_edits=patches,global_current_pointer_write_allowed=False,
                  control_source=str(Path(control).resolve()),control_sha256=hashlib.sha256(raw_control).hexdigest(),
                  control_snapshot=frozen_control,control_reload_requests_ignored=True,
                  source_process_alive=source_process_alive(source,bank),source_status=bank.get('status'),
                  source_deadline=bank.get('deadline'),argv=evaluation_argv(launch['command'],source,output),
                  first_episode_only=True,natural_terminal_reward_included=True,
                  metrics_scope='4 fresh deterministic environments per teacher, complete 24-second natural episode or first physical fall',
                  runtime_python_sha256={f.name:sha(f) for f in (source/'runtime').glob('*.py')},
                  created_unix=time.time())
    return manifest,text


def run_evaluation(manifest,source_text):
    source=Path(manifest['source_bank']);output=Path(manifest['output']);project=Path(manifest['project'])
    if manifest['source_process_alive']:raise RuntimeError('Source trainer still running; final evaluation refused')
    if manifest['source_status'] not in ('completed','stopped_at_deadline','stopped_requested','failed','smoke_complete'):
        raise RuntimeError('Source has not published a terminal status')
    if manifest.get('source_deadline') and time.time()<datetime.datetime.fromisoformat(manifest['source_deadline']).timestamp():
        raise RuntimeError('Requested final evaluation is before the training deadline')
    output.mkdir(parents=True,exist_ok=True);manifest=dict(manifest,status='initializing')
    manifest.update(pin_checkpoint_selection(manifest))
    atomic_json(output/'final_evaluation_manifest.json',manifest)
    sys.dont_write_bytecode=True
    runtime=source/'runtime';sys.path.insert(0,str(runtime));os.environ['SABLE_GYM_ROOT']=str(project)
    # Frozen core imports Isaac before Torch, as required by its runtime.
    core=types.ModuleType('final_bank_evaluation_frozen_core');core.__file__=str(runtime/'teacher_bank_core.py')
    code,_=transform_core(source_text,core.__file__);exec(code,core.__dict__)
    torch=core.torch
    original_atomic=core.atomic;original_create=core.create_env;original_save=torch.save
    original_adam_step=torch.optim.Adam.step;original_argv=sys.argv[:]
    global_pointer=(project/'results/teacher_bank/current.json').resolve()
    pointer_before=global_pointer.read_bytes() if global_pointer.exists() else None
    suppressed=[];context={}
    def guarded_atomic(path,value):
        resolved=Path(path).resolve()
        if resolved==global_pointer:suppressed.append(str(resolved));return
        if not inside(resolved,output):raise RuntimeError('Core attempted external write: '+str(resolved))
        if resolved==(output/'teacher_sources.json').resolve():
            raise RuntimeError('Core attempted to overwrite the frozen checkpoint selection')
        if resolved==(output/'resume_verification.json').resolve():verify_loaded_selection(manifest,value)
        return original_atomic(path,value)
    def guarded_save(value,path,*args,**kwargs):
        if not inside(path,output):raise RuntimeError('Core attempted external checkpoint write')
        return original_save(value,path,*args,**kwargs)
    def forbidden_update(*args,**kwargs):raise RuntimeError('Optimizer/update forbidden during final evaluation')
    def create_env(cfg):
        if cfg.env.num_envs!=32 or cfg.teacher_bank['train_per']!=0 or cfg.env.episode_length_s!=HORIZON:
            raise RuntimeError('Unexpected final evaluation scene dimensions')
        env=original_create(cfg)
        env.night_control=copy.deepcopy(manifest['control_snapshot'])
        if env.training_env_count!=0 or len(env.eval_ids)!=32:raise RuntimeError('Training environments leaked into final evaluation')
        context.update(env=env,step_count=0,alive=torch.ones(32,device=env.device,dtype=torch.bool),
                       samples=torch.zeros(32,device=env.device),terms={},terminal=[None]*32)
        original_step=env.step
        def record_step(action):
            result=original_step(action);state=env.pre_reset_teacher_state;active=context['alive'].float()
            context['samples']+=active;context['step_count']+=1
            for name,value in env.teacher_terms.items():
                if name not in context['terms']:context['terms'][name]=torch.zeros(32,device=env.device)
                context['terms'][name]+=value.detach()*env.dt*active
            terminal=state['fall']|env.time_out_buf
            for index in torch.nonzero(context['alive']&terminal,as_tuple=False).flatten().tolist():
                context['terminal'][index]=dict(time_s=context['step_count']*env.dt,
                    physical_fall=bool(state['fall'][index]),natural_timeout=bool(env.time_out_buf[index]),
                    diagnostics={name:value[index].item() for name,value in state['fall_diagnostics'].items()})
            context['alive']&=~terminal
            return result
        env.step=record_step
        return env
    core.atomic=guarded_atomic;core.create_env=create_env;core.update=forbidden_update
    torch.save=guarded_save;torch.optim.Adam.step=forbidden_update
    try:
        sys.argv=manifest['argv'];core.main()
        if sha(output/'teacher_sources.json')!=manifest['teacher_sources_sha256']:
            raise RuntimeError('Frozen checkpoint selection changed during evaluation')
        verify_loaded_selection(manifest,read_json(output/'resume_verification.json'))
        env=context['env']
        if context['step_count']!=env.max_episode_length or any(item is None for item in context['terminal']):
            raise RuntimeError('Full first-episode terminal coverage was not achieved')
        for i,selected in enumerate(manifest['selected_checkpoints']):
            ident=selected['teacher_id'];report_path=output/ident/'latest_evaluation.json';report=read_json(report_path)
            if report['horizon_s']!=HORIZON or report['evaluation_seed']!=947:raise RuntimeError('Wrong evaluation protocol')
            copied=read_json(output/ident/'checkpoint.json')
            original_cp=torch.load(selected['path'],map_location='cpu');evaluated_cp=torch.load(copied['path'],map_location='cpu')
            if not core.state_equal(original_cp['model_state_dict'],evaluated_cp['model_state_dict']):
                raise RuntimeError('Policy weights changed during evaluation: '+ident)
            if not core.state_equal(original_cp['optimizer_state_dict'],evaluated_cp['optimizer_state_dict']):
                raise RuntimeError('Optimizer state changed during evaluation: '+ident)
            terminal=context['terminal'][i*4:(i+1)*4]
            failed=[bool(item['diagnostics'].get('task_boundary_failure',False)) for item in terminal]
            report.update(mode='final_evaluation_only',optimization_steps=0,horizon_s=HORIZON,
                acceptance_scope='Final simulation evaluation: four held-out environments, 24 seconds, first episode only; not hardware validation',
                reward_return_scope='Undiscounted first-episode environment rewards; excludes PPO timeout value bootstrap; compare physical metrics across reward revisions',
                executed_simulation_s=context['step_count']*env.dt,source_checkpoint=selected,
                policy_and_optimizer_unchanged=True,terminal_episodes=terminal,
                task_boundary_pass_fraction=1-sum(failed)/4.,
                natural_timeout_fraction=sum(item['natural_timeout'] for item in terminal)/4.,
                first_episode_samples=context['samples'][i*4:(i+1)*4].cpu().tolist(),
                reward_returns_per_environment={name:value[i*4:(i+1)*4].cpu().tolist() for name,value in context['terms'].items()},
                accepted=bool(report.get('accepted',False) and not any(failed) and all(item['natural_timeout'] for item in terminal)))
            atomic_json(report_path,report)
            atomic_json(output/ident/'evaluations'/('final_24s_%06d.json'%selected['iteration']),report)
            video_path=output/ident/'video_receipt.json';video=read_json(video_path)
            video.update(mode='final_evaluation_only',horizon_s=HORIZON,source_checkpoint=selected,
                         optimization_steps=0,source='Fresh 4-env-per-teacher deterministic 24-second evaluation')
            atomic_json(video_path,video)
            if sha(selected['path'])!=selected['sha256']:raise RuntimeError('Source checkpoint changed during evaluation')
        if (global_pointer.read_bytes() if global_pointer.exists() else None)!=pointer_before:
            raise RuntimeError('Global current pointer changed concurrently; final evaluation did not write it')
        manifest.update(status='completed',completed_unix=time.time(),policy_and_optimizer_unchanged=True,
                        actual_simulation_steps=context['step_count'],suppressed_pointer_writes=suppressed)
        atomic_json(output/'final_evaluation_manifest.json',manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),updated_unix=time.time())
        atomic_json(output/'final_evaluation_manifest.json',manifest);raise
    finally:
        torch.save=original_save;torch.optim.Adam.step=original_adam_step;sys.argv=original_argv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--project',type=Path,default=Path(os.environ['SABLE_GYM_ROOT']) if 'SABLE_GYM_ROOT' in os.environ else None)
    parser.add_argument('--control',type=Path)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();project=args.project or args.source_bank.resolve().parent.parent
    control=args.control or project/'deployment/night_20261029/control.json'
    manifest,text=prepare(args.source_bank,args.output,project,control)
    if args.plan:print(json.dumps(manifest,ensure_ascii=False,indent=2));return
    print(json.dumps(run_evaluation(manifest,text),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
