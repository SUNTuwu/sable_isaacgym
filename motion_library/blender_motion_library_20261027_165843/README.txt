Sable authored motion library / 2026-10-27

Entry points
- motion_catalog.json: all ten motions, coordinate and joint conventions, provenance.
- motions/<id>.npz + .json: fresh 50Hz exports from ONE live-scene snapshot.
- editor/sable_motion_edit.blend: matching baked editable scene; sparse toe IK controls preserved.
- editor/scripts/sable_motion_editor.py: optional live IK bootstrap in Blender's Text Editor.
- quality_report.json: pose/velocity seams, ground penetration, limits and IK error.
- asset_compatibility.json: physical/authored asset hashes match remote frozen and development assets.
- remote_bottleneck_diagnosis.json: historical evidence and the actual command-coverage bottleneck.
- teacher_bank_plan.json: independent teachers FROM SCRATCH, then a fresh distilled student.
- FILES_SHA256.json: integrity inventory; run verify_bundle.py with existing NumPy Python.

Important data contract
The 15-joint animation order differs from the policy action order. Use joint_names,
policy_joint_names and policy_from_motion_indices. Explicit policy-order arrays are supplied.
Only the opposed right hip coordinate is passive. Foot order is FL,FR,HL,HR.
Positions are meters, angles radians, quaternion is wxyz, +X forward/+Y left/+Z up.
Each file has51samples over1second, inclusive of both endpoints. The24frame UI gaps
are excluded. Derivatives are finite differences at50Hz, not physics-substep measurements.
Apply cycle_transform_world when repeating a translating/turning reference, after seam repair.
Toe targets mean the distal capsule tip. Ground contact is a different point.
No contact-force labels, torques or validated teacher action labels are supplied.

Status and current user direction
These are kinematic references requiring physical preflight, NOT ready-to-deploy policies.
Do not initialize new teachers or student from the latest S3 weights. Each specialist
starts fresh; accepted physical teacher rollouts later supervise a new conditional student.
No training was started and the existing TRAINING_PAUSED marker was left intact.
Eight primary motion teachers are planned, with two original side-pace comparison motions.
Teacher plan resource values are unbenchmarked starting points, not a measured capacity.

Known cleanup needs
- Forward/backward walk still have about1.94deg pose seam and strong velocity seams.
- Several references penetrate the ground up to about2.5mm; right pace has0.74mm IK residual.
- Turn references contain about100deg per cycle, so a small-turn task needs a coherent variant.
- All ten clips need velocity/seam and dynamic checks; no existing-contact labels are assumed.
Full numerical details are in quality_report.json. Preserve these originals when creating
processed teacher references; write a separate processing recipe and output version.

The editor/README.md is the authoring history and contains paths to older local evidence.
The package-root catalog, quality report and NPZ files are authoritative for THIS delivery.
