2026-10-29 最新训练与服务器迁移
================================

本次仅整理、校验、提交与准备迁移，没有开始新训练，也没有删除文件。

当前继续训练的起点是 `baselines/user_accepted_20261029_120250`，来自
`runs/teacher_bank_fullref_referencev2_straightcross_20261029_080810` 的第 2084 轮。
八个模型均包含策略、critic 与 Adam 优化器状态。保护基线的 267 个文件均已校验；
八份基线模型与最终评估使用的模型逐张量一致，优化器状态也一致。

它们是 94 维观测、15 维输出的参考动作加残差 teacher，腰 pitch 固定在参考站姿，
14 个关节参与探索。它们不是已经验收完成的 cmd16/history6 无参考 student。
`results/cmd16_hist6/current_migration.json` 与旧 launcher 中的 PID、日期和 deadline
是历史记录；不要用它们判断当前运行状态或直接重新开训。

最终独立评估是每策略 4 个环境、每环境 24 秒，共 32 个环境，全部存活。
八个策略的严格 tracking `accepted` 仍为 false。用户对基本行走的接受与严格验收
分开记录。独立 yaw stance 最新是 1274，16 个环境存活，角度 RMSE 21.05°，
尚未通过验收，也不在用户本次基本行走接受范围内。详细数值见
[training_snapshot.json](training_snapshot.json)、
[final_acceptance_summary.json](final_acceptance_summary.json) 和
[yaw_final_evaluation.json](yaw_final_evaluation.json)。

需要搬走的内容
----------------

| 内容 | 获取方式 | 用途 |
| --- | --- | --- |
| 本仓库及固定版本 submodule | GitHub clone | 代码、配置、物理资产、参考数组、结果摘要、迁移工具 |
| `transfer/sable_training_20261029.tar.gz` 与 `.sha256` | 私下传输，约 31.1 MB / 29.7 MiB | 完整保护基线、8 个模型及 Adam、最新 yaw1274 和旧保护 yaw28、各自运行代码与证据 |
| `vendor/isaacgym/` | 从已有服务器私下复制，或按 NVIDIA 许可安装 Preview 4 | NVIDIA SDK；不进入 GitHub |
| `/root/gpufree-data/envs/sable-isaacgym-py38/` | 同路径复制现有环境，或重建匹配环境 | Python 3.8.20、Torch 2.1.2+cu118、NumPy 1.23.5 等 |
| Blender 编辑源 | 需要继续模型/动作编辑时另行复制 | 新的大 `.blend` 已排除在 Git；当前训练使用导出的资产 |

迁移包 SHA256：

```text
30096e40527ec7a8545dba2679ee8aef0659ffe53556fcaf0ba8a3b823d76b3a
```

包中有 628 个文件，每个文件的尺寸和 SHA256 位于包内 `TRANSFER_MANIFEST.json`。
`restore` 会逐项验证，已有同内容文件直接跳过，遇到不同内容会拒绝覆盖。
旧的大规模 runs、视频、缓存、旧 transport zip 不需要为基本续训全量复制。
如要分析旧实验，应另外保存对应历史目录；本次没有移除它们。

在目标服务器准备
----------------

保留 `/root/gpufree-data/projects/sable-isaacgym` 与现有环境路径，避免旧配置和
SDK editable install 的绝对路径失效。新服务器需有可用 NVIDIA GPU/驱动。

```bash
mkdir -p /root/gpufree-data/projects
git clone --recurse-submodules git@github.com:SUNTuwu/sable_isaacgym.git \
  /root/gpufree-data/projects/sable-isaacgym
cd /root/gpufree-data/projects/sable-isaacgym
mkdir -p transfer
```

用现有 SSH 方式将迁移包和 checksum 文件放入目标仓库的 `transfer/`。同时将 SDK
复制到目标 `vendor/isaacgym/`，将 Python 环境复制到相同绝对路径。跨不同 Linux
发行版时，复制环境可能需要按目标系统重建；以以下目标机检查为准。
`verification/requirements-lock.txt` 记录了版本，但含原服务器 wheel 本地路径，
不能直接当作通用网络安装命令。需要重建时参考 `deployment/install.sh` 中的版本，
先准备好官方 Preview 4 SDK；不要改全局驱动或系统 Python。

```bash
cd transfer
sha256sum -c sable_training_20261029.tar.gz.sha256
cd ..
python3 deployment/migration_20261029/transfer_artifacts.py restore \
  --archive transfer/sable_training_20261029.tar.gz
python3 deployment/migration_20261029/prepare_resume.py --verify-only

source deployment/runtime.sh
python -m pip check
python deployment/verify.py --device gpu
python deployment/verify_articulation.py
```

当前迁移检查只做了 CPU 文件/模型状态校验；目标机 GPU 物理检查与一次实际 PPO
续接仍需在该服务器执行。SDK 必须先于 Torch 导入，现有物理验证脚本遵守此顺序。
缓存会在 `/root/gpufree-data` 下重建。SDK、环境与 GitHub 权限凭据独立迁移，
凭据不要放进仓库或迁移包。

准备与启动新的基本行走训练
--------------------------

以下准备命令不启动训练。选择未使用的新名称；已有目录会拒绝覆盖。
默认每 teacher 1024 个训练环境，共 8192 个训练环境，加 32 个独立评估环境；
显存不足时可以显式减小 `--envs-per-teacher`，但吞吐和学习条件会相应变化。

```bash
python deployment/migration_20261029/prepare_resume.py \
  --name teacher_bank_fullref_newserver_round01 --envs-per-teacher 1024
```

它从保护基线复制出独立 source2084 权重、一个新运行目录、冻结 runtime、参考数组、
控制设置与 `start.sh`。保持模型、Adam、探索标准差、参考强度、物理资产及奖励配置。
默认时长读取 `results/legged_gym/training_preferences.json`，当前为 20 分钟。
180 秒独立评估与可视化轨迹由冻结 teacher core 生成，包含实际 checkpoint 来源；
这些是运动学 JSON 回放轨迹，不是额外编码的 MP4。

只有在目标服务器明确开始新一轮时执行：

```bash
bash runs/teacher_bank_fullref_newserver_round01/start.sh
```

启动时才生成新的 deadline，启动并核对实际 watchdog 进程。保存
`resume_verification.json` 中策略/Adam 的逐项校验。正常到时后检查最终评估与
checkpoint 匹配，并生成和校验本轮 `.tar.gz`。提前停止或失败不会标记正常完成。
不修改原始 2084 基线或旧 run。新机器的首轮实际续接结果以日志为准。

需要网页时，在目标服务器另一个终端启动现有 dashboard，明确指定新 bank：

```bash
source deployment/runtime.sh
python deployment/teacher_bank_dashboard_20261027_1720/server.py \
  --bank runs/teacher_bank_fullref_newserver_round01 --port 11111
```

该页面跟随本轮实际 checkpoint、独立评估和轨迹，不运行旧夜间 monitor 的自动调参。

本入口只续八个基本动作 teacher，不自动开启独立 yaw 训练、蒸馏或完整 student
迁移流程。yaw 最新模型、原代码和旧保护模型已放入私有包，保留后续独立改进起点。
原夜间 monitor/launcher 含旧截止时间、PID 与恢复策略，不应直接重新执行。

本地复核和重新打包
------------------

```bash
python3 deployment/migration_20261029/prepare_resume.py --verify-only
python3 deployment/migration_20261029/transfer_artifacts.py verify \
  --archive transfer/sable_training_20261029.tar.gz
python3 deployment/migration_20261029/check_git_payload.py
```

如需另建包，给 `build --archive` 一个新的文件名；工具拒绝覆盖已有包。
本次验证记录在 [validation.json](validation.json) 和
[checkpoint_state_audit.json](checkpoint_state_audit.json)。

待批准的清理
------------

当前仅建议清理可重建缓存：15 个开发代码 `__pycache__` 目录约 4.0 MB，
dashboard telemetry cache 约 6.3 MB。完整路径见
[cleanup_candidates.json](cleanup_candidates.json)。尚未获得删除同意，实际删除数量为零。
旧冻结运行、保护基线、Blender 源和 `待删除/` 不在此次建议中。
