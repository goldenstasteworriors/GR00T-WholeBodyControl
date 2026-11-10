# SONIC v1.1 板载控制与 Inspire 数据采集 SOP

适用：G1 29 自由度＋Inspire 六电机双手，机器人运行 SONIC v1.1 C++ 推理和相机服务，PC 运行 PICO、手桥接及数据导出。沿用本项目数据格式：`observation.state` / `action.wbc` 为 41 维（身体29＋左手6＋右手6），保留 motion token、teleop、图像及 modality 元数据。不能换成 Dex1 的31维导出器。

以下代码块按行执行；行末 `\` 表示命令继续到下一行。首次安装使用新目录；已有环境不要重复执行会重建 venv 的安装脚本。

## 1. 一次性准备

**机器人系统前提**：按 NVIDIA 官方教程准备机载系统；AGX Orin 使用 JetPack 6＋TensorRT 10.7，Thor 使用 JetPack 7.1＋TensorRT 10.13。先核对实际型号、L4T 和 TensorRT，不在本 SOP 中升级 CUDA、驱动或刷机。不同平台的 TensorRT engine 不直接混用，在目标机生成。

机器人和 PC 均取得我们的固定代码版本（需要仓库读取权限；SSH key 配置由各自帐号完成）：

```bash
git clone git@github.com:goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 4ad2687808b44e3ef126b998fc3fc0972121f6e4
git submodule update --init --recursive
git lfs pull
```

推荐机器人放到 `/home/unitree/VLA/GR00T-WholeBodyControl`；PC 可以自行选目录。后文 PC 命令均从项目根目录执行。已有部署可复用，但必须确认启动器指向的机器人目录和代码版本一致。

**机器人：C++ 依赖和编译。** 依赖安装依据官方教程及 `gear_sonic_deploy/scripts/install_deps.sh`；先核对已有依赖，该脚本含系统依赖/CUDA 检测安装逻辑，不在已有机器人上不加检查地重跑。系统基础未满足时先完成平台适配。依赖齐全后：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
just build
test -x target/release/g1_deploy_onnx_ref
```

**机器人：相机与 v1.1 模型。** 在新克隆目录中执行；RealSense 的 Python 包必须能够导入并枚举设备，ARM64 无可用 wheel 时按 RealSense 官方方法构建匹配 SDK，不能直接宣称 pip 已完成相机适配。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
uv pip install --python .venv_camera/bin/python pyrealsense2 huggingface_hub
.venv_camera/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
.venv_camera/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GEAR-SONIC', allow_patterns=['sonic_v1_1/model_encoder.onnx','sonic_v1_1/model_decoder.onnx','sonic_v1_1/observation_config.yaml'], local_dir='gear_sonic_deploy/policy')"
.venv_camera/bin/python download_from_hf.py
```

最后一条是当前固定代码的兼容办法：下载它支持的默认 planner 等资源；实际控制仍显式选 v1.1 的 encoder、decoder、observation_config 三件套。当前下载脚本不支持 `--sonic-v1-1`，不能照最新版文档原样调用。记录模型 SHA256，交付时连模型文件一并提供；下载后将本次 HF revision 记录到交付清单。

```bash
sha256sum gear_sonic_deploy/policy/sonic_v1_1/* gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

**PC：安装 PICO 和采集环境**（安装前已有 uv；脚本会重建对应环境，只用于首次安装）：

```bash
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
uv pip install --python .venv_teleop/bin/python flask
```

按官方 [PICO 配置教程](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)完成头显端及 PC Service 配对。安装脚本不等于头显已连接。PC 配置到机器人 SSH 免密登录，并确认到机器人的有线网卡、PC 有线地址及两手 IP；下面所有示例值需按本机替换。左右手物理归属须分别核实，不能只依据 IP 尾号。

## 2. 调整并记录灵巧手开合

调节时先退出采集，不同时运行多个手命令来源。你们旧机器人曾使用以下工具；它是现存机器人上的外部工具，新装机器不保证存在：

机器人终端：

```bash
cd /home/unitree/VLA/locodemo
python ctrl_hand.py
```

PC 另开终端，建立隧道并保持运行：

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.164
```

PC 浏览器地址（不是直接访问 PC 上不存在的服务）：

```text
http://localhost:8080/
```

如果接收方没有 `ctrl_hand.py`，使用仓库内替代工具。在 PC 项目根目录开两个终端；将 `enp7s0` 换成实际机器人有线网卡：

终端一，直接透传网页六电机目标：

```bash
.venv_teleop/bin/python decoupled_wbc/scripts/inspire_modbus_hand.py --mode dds --network enp7s0 --left-ip 192.168.123.210 --right-ip 192.168.123.211 --dds-pose-mode passthrough
```

终端二，网页端口保持8080：

```bash
.venv_teleop/bin/python decoupled_wbc/scripts/inspire_hand_web.py --network enp7s0 --host 127.0.0.1 --port 8080
```

同样打开上面的本地网页，无需 SSH 隧道。逐个调整电机并发送，分别记录“张开”和“抓握”的6个值。顺序固定为：小指、无名指、中指、食指、拇指弯曲、拇指旋转。本项目值域0–1，通常0闭合、1张开；若旧工具显示0–1000，则除以1000，若显示角度单位则不能直接套用。

将结果写入 **PC 采集仓库**的以下文件；网页调节不会自动更新该文件：

```text
gear_sonic/config/data_collection/inspire_hand_pose.json
```

格式示例（数值须替换为实测结果）：

```json
{
  "open": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
  "grasp": [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
}
```

当前配置对双手共用一套开合姿态。左右分别记录的值若不同，现有配置不能表达两套独立姿态，须先扩展配置和桥接，不能误以为已生效。记录任务名称、日期、手的物理侧别和六电机值。完成后 Ctrl+C 退出网页工具及调试桥接，再进入采集。

## 3. 每次采集

**机器人终端：启动相机**（已有同端口相机服务时复用，不重复启动）。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

**PC：先看图，按 q 退出。**

```bash
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```

**PC：启动全身采集。** 修改任务提示、dataset-name、PC 有线 IP、手桥接网卡及机器人机载网卡。示例由启动器统一启动手桥接，勿再手动启动第二份；模型推理在机器人，导出在 PC。

```bash
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --deploy-onboard --deploy-onboard-host 192.168.123.164 \
  --deploy-onboard-repo-root /home/unitree/VLA/GR00T-WholeBodyControl \
  --deploy-onboard-interface eth0 --offboard-zmq-host 192.168.123.222 \
  --deploy-checkpoint policy/sonic_v1_1/model \
  --deploy-obs-config policy/sonic_v1_1/observation_config.yaml \
  --deploy-planner planner/target_vel/V2/planner_sonic.onnx \
  --inspire-hand-bridge --inspire-hand-network enp7s0 \
  --inspire-left-ip 192.168.123.210 --inspire-right-ip 192.168.123.211 \
  --inspire-hand-pose-config gear_sonic/config/data_collection/inspire_hand_pose.json \
  --task-prompt "pick up the box" --dataset-name pick_box_v11_20261110_01 \
  --enable-sonic-publish --no-pico-vis-smpl --no-pico-vis-vr3pt \
  --profile-timing --save-profile-logs
```

这条命令会请求真实机器人控制，须现场确认机器人已准备好，再在 deploy pane 按 y。发布开关与部署确认缺一不可。首先确认头显数据、身体状态、手反馈及图像正常，再按 manager 提示校准、进入控制。可视化默认关闭以减少额外负载；需要预览时可改回两个 `--pico-vis-*` 参数。

将焦点放到数据导出 pane：按 c 开始录制，再按 c 结束并保存；录制中按 x 丢弃当前段。每批使用新 dataset-name，不默认启用 `--overwrite-existing-dataset`。停止时先按控制界面提示停止身体控制，再退出进程，勿把 tmux 断开视为停止机器人。

## 4. 保存与交付验收

PC 输出位于项目 `outputs/<dataset-name>/`。检查 meta、data、videos：`observation.state`、`action.wbc` 各41维，身体/左手/右手顺序正确，motion token及teleop字段存在，图像可解码，episode帧数与视频一致。缺少状态或图像时不算采集验收通过；记录50Hz不代表相机有50张独立曝光。

交付代码固定提交、上面三个模型文件与 planner、手姿态 JSON、双方网卡/IP表、依赖版本清单及本 SOP。虚拟环境目录不能视为可移植安装包；保留环境定义与原始权重，在目标平台重新构建。无需安装训练环境或 Isaac Lab。

依据：本项目 `launch_data_collection.py`、`run_data_exporter.py`、`features_sonic_vla.py`、`inspire_modbus_hand.py`、`inspire_hand_web.py`；[官方机载安装](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)、[官方模型下载](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)。

核查记录（2026-11-10）：上述CLI、构建入口和数据维度已按本地源码核对；机器人SSH连接超时，未现场验证旧 `ctrl_hand.py`、新机全新安装或真实控制。本机找到历史机载代码工作区，未找到可确认版本的完整机器人系统/环境归档；因此本 SOP 不将其当作已验证镜像。接收方第一次安装仍须完成一次现场验收。
