"""Build, check or restore private training artifacts, without overwrites or training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from prepare_resume import BASELINE, verify_baseline

BANK = 'runs/teacher_bank_fullref_referencev2_straightcross_20261029_080810'
YAW = 'runs/yaw_sine_night_annealed_20261029_061927_168251'
YAW_BASELINE = 'runs/yaw_sine_night_20261029_041931'
MANIFEST = 'TRANSFER_MANIFEST.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_files(project):
    verify_baseline(project / BASELINE)
    selected = set(p for p in (project / BASELINE).rglob('*') if p.is_file())
    # Include latest separate yaw training and its protected older fallback.
    for name in (YAW, YAW_BASELINE):
        directory = project / name
        receipt = json.loads((directory / 'checkpoint.json').read_text())
        checkpoint = Path(receipt['path'])
        if checkpoint.parent.resolve() != directory.resolve():
            raise ValueError('Yaw checkpoint is outside its own run')
        selected.add(checkpoint)
        for filename in ('checkpoint.json', 'config.json', 'status.json', 'evaluation.json',
                         'corrected_evaluation.json', 'initial_poses.json', 'launch_receipt.json',
                         'completion.json', 'resume.json'):
            if (directory / filename).is_file():
                selected.add(directory / filename)
        selected.update(p for p in (directory / 'runtime').rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    for name in ('teacher_bank.json', 'launch_receipt.json', 'resume_verification.json'):
        selected.add(project / BANK / name)
    for name in ('control.json', 'yaw_baseline.json', 'new_baseline_receipt.json',
                 'final_acceptance_summary.json', 'final_evaluation_pointer.json'):
        selected.add(project / 'deployment/night_20261029' / name)
    return sorted(selected)


def build(project, archive):
    if archive.exists():
        raise FileExistsError('Refuse to overwrite a transfer archive')
    files = select_files(project)
    records = {str(p.relative_to(project)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in files}
    manifest = dict(schema='sable-private-transfer/v1', baseline=BASELINE,
                    bank_iteration=2084, yaw_latest_iteration=1274, files=records,
                    excludes='SDK, Python environment, old PPO runs and bulk replay videos')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.partial')
    with tarfile.open(temporary, 'w:gz') as handle:
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(data)
        handle.addfile(info, io.BytesIO(data))
        for path in files:
            handle.add(path, arcname=str(path.relative_to(project)), recursive=False)
    inspect_archive(temporary)
    temporary.replace(archive)
    checksum = sha(archive)
    archive.with_name(archive.name + '.sha256').write_text(checksum + '  ' + archive.name + '\n')
    print(json.dumps(dict(archive=str(archive), sha256=checksum, bytes=archive.stat().st_size,
                          files=len(records)), indent=2))


def inspect_archive(archive):
    with tarfile.open(archive, 'r:gz') as handle:
        members = handle.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive paths')
        for member in members:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member: ' + member.name)
        with handle.extractfile(MANIFEST) as stream:
            manifest = json.load(stream)
        if set(names) != set(manifest['files']) | {MANIFEST}:
            raise ValueError('Archive contents differ from manifest')
        for name, expected in manifest['files'].items():
            with handle.extractfile(name) as stream:
                content = stream.read()
            if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
                raise ValueError('Corrupt archive member: ' + name)
    return manifest


def restore(project, archive):
    manifest = inspect_archive(archive)
    # Full preflight: existing Git-managed files may only match byte-for-byte.
    for name, expected in manifest['files'].items():
        target = project / name
        if project not in target.resolve().parents:
            raise ValueError('Destination path escapes project: ' + name)
        if target.exists() and (not target.is_file() or sha(target) != expected['sha256']):
            raise FileExistsError('Existing file differs; refusing overwrite: ' + name)
    restored = 0
    with tarfile.open(archive, 'r:gz') as handle:
        for name in manifest['files']:
            target = project / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(name) as source, target.open('xb') as destination:
                destination.write(source.read())
            restored += 1
    verify_baseline(project / BASELINE)
    print(json.dumps(dict(passed=True, verified_files=len(manifest['files']), restored_files=restored)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify', 'restore'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.project.resolve(), args.archive.resolve())
    elif args.action == 'restore':
        restore(args.project.resolve(), args.archive.resolve())
    else:
        manifest = inspect_archive(args.archive)
        print(json.dumps(dict(passed=True, verified_files=len(manifest['files']))))


if __name__ == '__main__':
    main()
