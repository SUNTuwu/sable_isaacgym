# cmd16 / history6 migration

The authoritative interface is `sable_cmd16_hist6_v1`. Commands use physical SI
units before named normalization. The deployment input is six current-to-past
51D proprioceptive frames,16 current commands and4 current clocks (326 values).
The history encoder predicts ROOT velocity and a16D learned latent; the action
network consumes90 values and produces15 complete joint-target offsets.
The92D training critic is separate. The opposed right hip receives no action or
motor torque. `spine_pitch` remains frozen during initial distillation.

The physical asset and motor envelope are unchanged. `migration_env.py` adds
the new interface, continuous phase integration, command measurements and the
explicit teacher attachment. Independent execution deliberately works with a
missing reference-table path. `teacher_adapter.py` mixes complete targets at
each of the40 physics substeps; it never adds a reference to a complete student
target. The student's target is held for one20ms control interval. Its previous
target observation records the actual last clamped substep target.

The encoder uses same-time velocity supervision plus reset-masked next-clean-
proprioception prediction for the latent. This is an explicit predictive
auxiliary objective, not a claim to reproduce HIMLoco's prototype loss.
No teacher tensors are silently loaded into incompatible student layers;
`parameter_mapping.json` lists the fully new initialization and skipped teacher
keys. Strict mirror averaging recomputes the encoder on mirrored raw histories.

Use `source deployment/runtime.sh` first. Useful entrypoints:

- `python -m unittest discover -s deployment/legged_gym_cmd16_hist6_20261027/tests -v`
- `prepare_teacher.py --output results/cmd16_hist6/s0`: wait for real training exit and verify the final archive.
- `launch_s1.py`: freeze a new runtime, collect complete teacher targets, and run deadline-bounded BC.
- `check_migration_env.py --teacher <teacher_manifest.json> --output <report.json> [--no-teacher]`:2s structural probe; not stability acceptance.
- `evaluate_migration.py --teacher <teacher_manifest.json> --checkpoint <student.pt> --output <report.json> --suite S2`: independent closed-loop evaluation.

Full completion requires teacher-free old-domain regression, subsequent
translation/yaw command learning and measured touchdown-frequency transitions.
Frozen runs and collected datasets remain immutable. Code in this development
directory may continue to improve while those runs execute.
