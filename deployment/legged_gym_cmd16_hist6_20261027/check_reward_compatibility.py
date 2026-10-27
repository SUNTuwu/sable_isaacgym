"""Audit the tested yaw-cost-only change before preserving PPO learning state.

This proves compatible learning/control structure, not successful locomotion.
Independent physical evaluation is required separately.
"""
import argparse
import ast
from pathlib import Path
from run_io import atomic_json,read_json,sha256

NONTRAINING={'train_ppo.py','launch_ppo.py','watch_migration.py','watch_frontier.py',
    'export_policy.py','select_regression_checkpoint.py','queue_continuation.py'}
REWARD_METHODS={'_yaw_error_rad_s','_reward_yaw_rate_error','_reward_tracking_ang_vel'}


def core_files(directory):
    return {p.name:p for p in directory.glob('*.py')if p.name not in NONTRAINING
        and not p.name.startswith(('evaluate_','evaluation_','probe_','check_'))}


def expression_is(node, text):
    return ast.dump(node,include_attributes=False)==ast.dump(ast.parse(text,mode='eval').body,include_attributes=False)


class Contract(ast.NodeTransformer):
    def __init__(self, kind):self.kind=kind

    def visit_Import(self,node):
        if self.kind=='train' and len(node.names)==1 and node.names[0].name=='math':return None
        return node

    def visit_ImportFrom(self,node):
        if self.kind=='environment' and node.module=='motion_rewards':
            assert [(x.name,x.asname)for x in node.names]==[('yaw_error_cost',None),('angular_tracking_reward',None)]
            return None
        return node

    def visit_ClassDef(self,node):
        if self.kind=='environment' and node.name=='MigrationRobot':
            node.body=[x for x in node.body if not isinstance(x,ast.FunctionDef)or x.name not in REWARD_METHODS]
        return self.generic_visit(node)

    def visit_keyword(self,node):
        if self.kind in ('config','train') and node.arg=='yaw_error_scale_rad_s':return None
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if self.kind=='train' and isinstance(node.value,ast.Call):
            call=node.value
            if expression_is(call.func,'p.add_argument')and call.args and isinstance(call.args[0],ast.Constant)and call.args[0].value=='--yaw-error-scale':
                return None
        return self.generic_visit(node)

    def visit_If(self,node):
        if self.kind=='train':
            if expression_is(node.test,'a.yaw_error_scale is not None'):return None
            if expression_is(node.test,'a.restore_optimizer'):
                # Only the reviewed same-stage guard plus the unchanged loader
                # may be here. The new guard permits actor-frozen critic warmup.
                assert not node.orelse and len(node.body)==2 and isinstance(node.body[0],ast.If)
                assert ast.dump(node.body[1],include_attributes=False)==ast.dump(
                    ast.parse('restore_optimizers(algorithm,parent)').body[0],include_attributes=False)
                node.body=[node.body[1]]
        return self.generic_visit(node)


def contract(path,kind):
    return ast.dump(Contract(kind).visit(ast.parse(path.read_text())),include_attributes=False)


def audit(parent,source):
    parent=parent.resolve();source=source.resolve();old=parent/'runtime';failures=[]
    previous,current=core_files(old),core_files(source)
    exceptions={'migration_env.py','migration_config.py','motion_rewards.py'}
    if set(previous)-exceptions!=set(current)-exceptions:failures.append('training_module_set_changed')
    checked=[]
    for name in sorted((set(previous)&set(current))-exceptions):
        if sha256(previous[name])!=sha256(current[name]):failures.append('training_code_changed:'+name)
        checked.append(name)
    for name,kind in [('migration_env.py','environment'),('migration_config.py','config'),('train_ppo.py','train')]:
        try:
            if contract(old/name,kind)!=contract(source/name,kind):failures.append('nonreward_contract_changed:'+name)
        except (AssertionError,SyntaxError,AttributeError):failures.append('unrecognized_contract:'+name)
    old_assets={str(p.relative_to(old/'assets')):sha256(p)for p in(old/'assets').rglob('*')if p.is_file()}
    new_assets={str(p.relative_to(source/'assets')):sha256(p)for p in(source/'assets').rglob('*')if p.is_file()}
    if old_assets!=new_assets:failures.append('physical_assets_changed')
    if not(source/'motion_rewards.py').is_file():failures.append('missing_yaw_cost_helper')
    return dict(schema='sable-yaw-reward-optimizer-compatibility/1',passed=not failures,failures=failures,
        parent_run=str(parent),parent_runtime_manifest_sha256=sha256(parent/'runtime_manifest.json'),
        source_runtime=str(source),source_files={str(p.relative_to(source)):sha256(p)for p in source.rglob('*')
            if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc'},
        identical_core_modules=checked,identical_asset_files=len(old_assets),
        reviewed_changes=['Physical yaw error cost normalization; positive Gaussian remains separately defined',
            'Explicit yaw-cost CLI/config/metrics; restored main actor/std may remain frozen during critic warmup'],
        scope='Structural optimizer compatibility for the tested yaw-cost change only. No claim of new locomotion capability or full task acceptance.')


def verify_preservation_evidence(path,parent,source,alpha,yaw_scale):
    from evaluate_frontier import assess_frontier
    evidence=read_json(path);structural=evidence['structural']
    assert evidence['schema']=='sable-tested-yaw-learning-state/1'
    assert evidence['training_alpha']==alpha and evidence['yaw_error_scale_rad_s']==yaw_scale
    assert structural['parent_run']==str(parent.resolve())
    assert structural['parent_runtime_manifest_sha256']==sha256(parent/'runtime_manifest.json')
    current=audit(parent,source)
    assert current['passed'] and structural['passed']
    assert current['source_files']==structural['source_files'],'Prepared source changed after review'
    for item in evidence['sources'].values():assert sha256(item['path'])==item['sha256']
    sources=evidence['sources'];report=read_json(sources['independent']['path'])
    config=read_json(sources['PPO_config']['path'])
    assert config['stage']=='S3' and config['optimizer_restore']
    assert config['yaw_error_scale_rad_s']==yaw_scale and config['command_frontier']['training_alpha']==alpha
    assert report['evaluation_yaw_error_scale_rad_s']==yaw_scale
    assert sha256(report['checkpoint'])==report['checkpoint_sha256']
    assert report['survival_fraction']>=.9 and report['reference_execution_calls']==0
    assert sources['baseline']['sha256']==sha256(parent/'baseline_metrics.json')
    gate=assess_frontier(report['frontier_curriculum'],report,read_json(parent/'baseline_metrics.json'),alpha)
    # A training preflight may still lack yaw response; it may not have lost
    # old-domain behavior, current-range gait/survival, or numerical structure.
    allowed={'turn_left:axis2_insufficient_response','turn_right:axis2_insufficient_response'}
    assert set(gate['failures'])<=allowed,gate['failures']
    tested=evidence['tested_training_files']
    assert set(tested)==set(core_files(source))|{'train_ppo.py'},'Incomplete tested training source'
    for name,digest in tested.items():assert sha256(source/name)==digest
    return dict(passed=True,source=str(path.resolve()),sha256=sha256(path),
        paired_independent_checkpoint_sha256=report['checkpoint_sha256'],
        independent_survival=report['survival_fraction'],remaining_frontier_failures=gate['failures'],
        reason='Identical policy/optimizer/observations/control/assets; audited yaw-cost-only change with successful restored-state physical preflight',
        task_acceptance=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--source-runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.parent_run,a.source_runtime);atomic_json(a.output,result)
    print({'passed':result['passed'],'failures':result['failures'],'identical_core_modules':len(result['identical_core_modules'])})
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
