# 远程 Isaac Gym 安装

2026-10-29 更新：最新训练、模型包及换服务器的操作以
[迁移说明](migration_20261029/README.md)为准。以下为 10 月 25 日的安装验收记录，
其中实例地址、进度描述和截止时间均为历史信息。

当前 GPU 实例：`ssh -p 30030 root@183.147.142.40`，使用仓库外 Windows 凭据助手做密码认证。原实例端口为 30491。

2026-10-25（香港时间）已通过无界面 GPU 物理验收。新实例的数据盘包含原安装环境，原样复用了 Isaac Gym Preview 4（包版本 `1.0rc4`）、Python 3.8.20、PyTorch 2.1.2+cu118、torchvision 0.16.2+cu118、NumPy 1.23.5 及已编译的 `gymtorch`；无需重装。GPU 为 RTX 4090 24 GB，驱动为 580.126.09，`pip check` 通过。

实测结果：

- GPU PhysX 和 CUDA 张量管线均启用，状态位于 `cuda:0`。
- 球体重力落地：240 步 / 2 s，半径 0.05 m，最终中心高度约 0.052 m，稳定接触地面。
- SDK 自带 ANYmal：单环境、固定基座、无重力，12 个关节通过 CUDA 力矩接口运动，4000 步 / 4 s；正弦目标跟踪 RMSE 约 0.011 rad，indexed reset 误差为 0。
- 上述测试未运行训练，也不代表用户机器人模型或行走能力已经验收；图形窗口和摄像头渲染未测试。

所有安装及运行数据位于挂载的数据盘 `/root/gpufree-data`：

- 项目与 SDK：`projects/sable-isaacgym/vendor/isaacgym`
- 独立 Python 3.8 环境：`envs/sable-isaacgym-py38`
- 安装脚本：`projects/sable-isaacgym/deployment`
- 包缓存和编译缓存：`cache/sable-isaacgym`
- 临时文件：`tmp/sable-isaacgym`
- 日志与验收记录：`projects/sable-isaacgym/logs`、`verification`

进入环境：

```bash
source /root/gpufree-data/projects/sable-isaacgym/deployment/runtime.sh
cd /root/gpufree-data/projects/sable-isaacgym
```

单环境 CPU 物理检查，不包含训练：

```bash
python deployment/verify.py --device cpu
```

用同一环境复现 GPU PhysX 和 CUDA 张量管线检查：

```bash
python deployment/verify.py --device gpu
python deployment/verify_articulation.py
```

GPU 验收记录见 `verification/smoke_gpu.json`、`articulation_gpu.json`、`gpu_host_preflight.json` 和 `installation_status.json`；这些记录也已同步至本地本目录下的 `verification/`。远端详细日志为 `logs/gpu_validation.log` 和 `logs/articulation_gpu.log`。

保留了两项诊断证据：旧实例无 GPU 模式的 `libcuda.so.1` 占位文件曾阻止插件加载，原 `smoke_cpu.json` 是当时的失败记录；补充四足测试最初使用 8.33 ms 显式 PD 更新，膝关节振荡，记录为 `articulation_gpu_initial_probe.json` 和 `articulation_gpu_coarse_pd_diagnostic.json`。保持增益和验收阈值不变，将显式反馈更新改为 1 ms 后通过；物理子步本身不会重算外部 PD 力矩。当前记录不将这些历史失败当作成功。

最初选择 Python 3.8 是因为 NVIDIA Preview 4 的 Python 绑定要求 `<3.9`，旧 MuJoCo 环境使用 Python 3.12。此次直接复用了已创建的 Isaac Gym 独立环境，没有修改 MuJoCo 环境或系统 GPU 驱动。Isaac Gym 应在 PyTorch 之前导入。所有固定版本见 `install.sh`，实际环境清单见 `verification/requirements-lock.txt`。

当前服务器为 Ubuntu 24.04；NVIDIA 公布的原始支持系统是 Ubuntu 18.04 / 20.04，所以运行可用性以实际检查结果为准。此步骤只安装 Isaac Gym，还没有将新 Blender 模型转换为可训练的物理机器人。

官方参考：[Isaac Gym 下载与环境要求](https://developer.nvidia.com/isaac-gym/download)、[Micromamba 手动安装](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)。
