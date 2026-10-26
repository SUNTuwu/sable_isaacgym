#!/usr/bin/env bash
set -euo pipefail
source /root/gpufree-data/projects/sable-isaacgym/deployment/runtime.sh
cd "$SABLE_GYM_ROOT"
name="${1:-stand5_20261025_v8_guided}"
phase="${4:-guided}"
strength="${5:-0.25}"
course_mode="${6:-validated}"
mirror="${7:-false}"
[[ "$mirror" =~ ^(true|false)$ ]] || exit 2
mirror_args=()
if test "$mirror" = true; then
  test "$phase" = guided || exit 2
  mirror_args=(--mirror-policy)
fi
[[ "$course_mode" =~ ^(validated|experimental)$ ]] || exit 2
[[ "$phase" =~ ^(balance|walk|gait|lift|guided)$ ]] || exit 2
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
run="$SABLE_GYM_ROOT/runs/$name"
if test -e "$run"; then echo 'Run directory already exists; refusing overwrite'; exit 2; fi
resume_args=()
if test -n "${2:-}"; then
  [[ "$2" == "$SABLE_GYM_ROOT"/runs/*/checkpoints/model_*.pt ]] && test -f "$2" || exit 2
  resume_args=(--resume "$2")
fi
if test -n "${3:-}"; then
  [[ "$3" == "$SABLE_GYM_ROOT"/runs/*/checkpoints/model_*.pt ]] && test -f "$3" || exit 2
  resume_args+=(--critic-checkpoint "$3")
fi
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); checks=[json.loads((p/f).read_text()) for f in ["fixed.json","standing.json"]]; assert all(c["passed"] for c in checks); assert checks[1]["seconds"]>=16' deployment/legged_gym/verification
python -c 'import json,hashlib; from pathlib import Path; p=Path("deployment/legged_gym"); r=json.loads((p/"verification/timeout_fixed.json").read_text()); assert r["passed"] and r["env_sha256"]==hashlib.sha256((p/"sable_env.py").read_bytes()).hexdigest()'
if test "$phase" = gait || test "$phase" = lift || test "$phase" = guided; then
  python -c 'import json,hashlib,sys; from pathlib import Path; p=Path("deployment/legged_gym"); r=json.loads((p/"verification/gait_signals.json").read_text()); assert r["passed"] and r["period_s"]==(.8 if sys.argv[1] in ["lift","guided"] else .5); assert r.get("guided",False)==(sys.argv[1]=="guided"); assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in r["source_sha256"].items())' "$phase"
fi
if test "$phase" = guided; then
  python -c 'import json,sys; from pathlib import Path; p=Path("deployment/legged_gym/verification"); expected=sys.argv[1]=="true"; assert json.loads((p/"guided_transfer.json").read_text()).get("mirror_policy",False)==expected; assert json.loads((p/"deadline_validation.json").read_text()).get("mirror_policy",False)==expected' "$mirror"
  python -c 'import json,hashlib,sys; from pathlib import Path; p=Path("deployment/legged_gym"); strength=float(sys.argv[1]); assert 0<=strength<=1; r=json.loads((p/"verification/guided_transfer.json").read_text()); assert r["reference_strength"]==strength; assert all(v for k,v in r["checks"].items() if k!="parent_survival_gate"); assert sys.argv[3]=="experimental" or (r["passed"] and all(r["checks"].values())); assert hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()==r["parent_sha256"]; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in r["source_sha256"].items()); g=json.loads((p/"verification/gait_signals.json").read_text()); assert g["reference_strength"]==strength; d=json.loads((p/"verification/deadline_validation.json").read_text()); assert d["passed"] and d["phase"]=="guided" and d["reference_strength"]==strength and d["supervisor_sha256"]==hashlib.sha256((p/"supervise.py").read_bytes()).hexdigest()' "$strength" "${2:-}" "$course_mode"
fi
if test "$phase" = lift; then
  python -c 'import json,hashlib; from pathlib import Path; p=Path("deployment/legged_gym"); r=json.loads((p/"verification/lift_transfer.json").read_text()); assert r["passed"] and all(r["checks"].values()); assert r["train_sha256"]==hashlib.sha256((p/"train.py").read_bytes()).hexdigest(); assert r["evaluator_sha256"]==hashlib.sha256((p/"evaluate.py").read_bytes()).hexdigest(); d=json.loads((p/"verification/deadline_validation.json").read_text()); assert d["passed"] and d["supervisor_sha256"]==hashlib.sha256((p/"supervise.py").read_bytes()).hexdigest()'
fi
mkdir -p "$run/runtime" "$run/checkpoints" "$run/evaluations"
cp deployment/legged_gym/*.py "$run/runtime/"
cp -a deployment/legged_gym/assets "$run/runtime/"
cp -a deployment/legged_gym/verification "$run/preflight"
python -c 'import json,sys,time; from pathlib import Path; run=Path(sys.argv[1]); r=json.loads((run/"preflight/guided_transfer.json").read_text()) if sys.argv[3]=="guided" else {}; (run/"stage_intent.json").write_text(json.dumps({"mode":sys.argv[2],"phase":sys.argv[3],"reference_strength":float(sys.argv[4]),"initial_performance_gate_passed":r.get("checks",{}).get("parent_survival_gate"),"initial_performance":r.get("parent_with_reference_summary"),"accepted_locomotion":False,"scope":"Remote simulation learning curriculum; structural checks remain mandatory. Initial performance failure is retained and is not gait acceptance.","created_unix":time.time()},indent=2))' "$run" "$course_mode" "$phase" "$strength"
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"stage_intent.json"; d=json.loads(p.read_text()); d["mirror_policy"]=sys.argv[2]=="true"; p.write_text(json.dumps(d,indent=2))' "$run" "$mirror"
printf '{"run_dir":"%s"}\n' "$run" > runs/current.json
nohup python -u "$run/runtime/supervise.py" --run "$run" --phase "$phase" --reference-strength "$strength" "${mirror_args[@]}" "${resume_args[@]}" > "$run/supervisor.log" 2>&1 < /dev/null &
echo $! > "$run/supervisor.pid"
echo "Started supervisor PID $! in $run"
