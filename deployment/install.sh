#!/usr/bin/env bash
set -Eeuo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$task_dir/runtime.sh"
test "$(findmnt -n -o TARGET -T /root/gpufree-data)" = /root/gpufree-data
test "$(readlink -f "$task_dir")" = "$SABLE_GYM_ROOT/deployment"
for path in "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$MAMBA_ROOT_PREFIX" "$SABLE_GYM_ROOT/vendor" "$SABLE_GYM_ROOT/logs" "$SABLE_GYM_ROOT/verification"; do
    mkdir -p "$path"
done
exec 9>"$SABLE_GYM_ROOT/install.lock"
flock -n 9 || { printf '%s\n' 'Another installation is running'; exit 1; }
trap 'status=$?; printf "exit_code=%s\n" "$status" > "$SABLE_GYM_ROOT/logs/install.exit"' EXIT
printf '%s\n' 'STAGE: validate and extract official Isaac Gym Preview 4 archive'
python3 - <<'PY'
from pathlib import Path
import hashlib,json,tarfile,shutil,subprocess
root=Path('/root/gpufree-data/projects/sable-isaacgym')
cache=Path('/root/gpufree-data/cache/sable-isaacgym')
cached=Path('/root/gpufree-data/cache/sable-legged-gym/IsaacGym_Preview_4_Package.tar.gz.partial')
archive=cached if cached.is_file() else cache/'IsaacGym_Preview_4_Package.tar.gz'
if not archive.is_file():
    subprocess.run(['curl','-fSL','--retry','3','--connect-timeout','20','--max-time','600','https://developer.nvidia.com/isaac-gym-preview-4','-o',str(archive)],check=True)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).parts[0]=='isaacgym' for m in members)
    destination=root/'vendor'
    if not (destination/'isaacgym/python/setup.py').exists():
        tar.extractall(destination,filter='data')
    size=sum(m.size for m in members)
record={'official_source':'https://developer.nvidia.com/isaac-gym-preview-4','archive_path':str(archive),
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,
        'uncompressed_bytes':size,'members':len(members),'version':'1.0.preview4'}
(root/'verification/package_source.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
PY
printf '%s\n' 'STAGE: create isolated Python 3.8 environment on the data disk'
task_mamba=/root/gpufree-data/toolchains/micromamba-sable/bin/micromamba
if [ ! -x "$task_mamba" ]; then
    curl -fsSL --retry 3 --connect-timeout 20 --max-time 180 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$TMPDIR/micromamba.tar.bz2"
    tar -xjf "$TMPDIR/micromamba.tar.bz2" -C "$MAMBA_ROOT_PREFIX" bin/micromamba
fi
if [ ! -x "$SABLE_GYM_ENV/bin/python" ]; then
    "$task_mamba" --no-rc create -y -p "$SABLE_GYM_ENV" --override-channels -c conda-forge python=3.8 pip=24.3.1 'setuptools<66' wheel
fi
python --version
printf '%s\n' 'STAGE: install CUDA-enabled PyTorch and Isaac Gym dependencies'
python "$task_dir/download_torch.py"
python -m pip install --index-url https://pypi.org/simple /root/gpufree-data/cache/sable-isaacgym/wheels/torch-2.1.2+cu118-cp38-cp38-linux_x86_64.whl
python -m pip install --index-url https://pypi.org/simple 'numpy==1.23.5' 'scipy==1.10.1' 'PyYAML==6.0.2' 'Pillow==10.4.0' 'imageio==2.35.1' 'ninja==1.11.1.3' requests
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu118 'torchvision==0.16.2+cu118'
python -m pip install --no-deps --no-build-isolation -e "$SABLE_GYM_ROOT/vendor/isaacgym/python"
python -m pip check
python -m pip freeze > "$SABLE_GYM_ROOT/verification/requirements-lock.txt"
printf '%s\n' 'STAGE: verify imports, tensor bridge and one CPU physics environment'
cd "$SABLE_GYM_ROOT"
python "$task_dir/verify.py" --device cpu
printf '%s\n' 'INSTALLATION_COMPLETE_GPU_VALIDATION_PENDING'
