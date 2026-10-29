"""Reject large or private training artifacts in the staged Git payload."""
from pathlib import PurePosixPath
import subprocess

LIMIT = 1024 * 1024  # Per newly added/modified file, not existing Git history.
BLOCKED = ('.pt', '.pth', '.onnx', '.ckpt', '.safetensors', '.blend', '.mp4',
           '.avi', '.zip', '.tar.gz', '.tar', '.pem', '.key')


def main():
    names = subprocess.check_output(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z']).decode().split('\0')
    names = list(filter(None, names))
    if any('\n' in name for name in names):
        raise SystemExit('Newline-containing Git paths require manual review')
    objects = subprocess.run(
        ['git', 'cat-file', '--batch-check=%(objecttype) %(objectsize)'],
        input=''.join(':' + name + '\n' for name in names),
        text=True, capture_output=True, check=True).stdout.splitlines()
    if len(objects) != len(names):
        raise SystemExit('Could not inspect every staged object')
    failures = []
    total = 0
    count = 0
    largest = (0, '')
    for name, metadata in zip(names, objects):
        path = PurePosixPath(name)
        if (name.endswith(BLOCKED) or path.parts[0] in
                ('runs', 'logs', 'transfer', '待删除', '.ssh', '.aws', '.codex', '.agents')
                or path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example'
                or name.startswith('vendor/isaacgym/')):
            failures.append('Excluded artifact: ' + name)
            continue
        kind, size_text = metadata.split()
        if kind == 'commit':  # A pinned submodule pointer carries no payload.
            continue
        size = int(size_text)
        total += size
        count += 1
        largest = max(largest, (size, name))
        if size > LIMIT:
            failures.append('File exceeds 1 MiB: ' + name)
    if failures:
        raise SystemExit('\n'.join(failures))
    print('PASS: %d staged files, %d bytes, largest %d bytes (%s)' %
          (count, total, largest[0], largest[1]))


if __name__ == '__main__':
    main()
