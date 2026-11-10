# SONIC v1.1 板载控制与 Inspire 数据采集 SOP

适用：G1 29 自由度＋Inspire 六电机双手，机器人运行 SONIC v1.1 C++ 推理和相机服务，PC 运行 PICO、手桥接及数据导出。沿用本项目数据格式：`observation.state` / `action.wbc` 为 41 维（身体29＋左手6＋右手6），保留 motion token、teleop、图像及 modality 元数据。不能换成 Dex1 的31维导出器。

以下代码块按行执行；行末 `\` 表示命令继续到下一行。首次安装使用新目录；已有环境不要重复执行会重建 venv 的安装脚本。

## 官方配置与本项目补充的边界

**基础环境、SONIC v1.1 模型和观测配置按官方说明配置即可**，不需要为我们的数据格式重新训练模型或修改其网络。先依据下列官方教程完成基础安装及遥操作连通，再使用本文的手部配置与采集入口：

| 部分 | 配置依据 |
| --- | --- |
| 机载平台、TensorRT、C++ 依赖与编译 | [官方 Deployment 安装](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) |
| v1.1 encoder、decoder、observation_config | [官方模型下载](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)，三件套保持配套，不手改模型输入维度 |
| PICO 安装、配对和全身遥操作 | [官方 PICO 配置](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)及[全身遥操作](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html) |
| Inspire 开合调节、DDS/Modbus 桥接 | 本文第2节及本项目代码 |
| 41维 state/action、手反馈及采集元数据 | 本项目导出器与 features 配置；不能直接用官方原版导出器替代 |

本文不是官方文档的逐字复刻。另有三项适配需要保留：①我们的启动器通过SSH在机器人运行推理，官方数据采集示例默认推理在PC；②我们使用RealSense，官方采集教程主要验证OAK相机；③固定代码版本的下载参数、发布开关和启动参数与最新官方代码不同。**可以按官方文档安装基础依赖，但不要混用最新版官方启动命令与本项目旧版本参数；最终采集使用第3节入口。**

如果选择最新官方仓库完成基础验证，官方 v1.1 下载参数为 `--sonic-v1-1`；本文固定版本采用下文的兼容下载方式。仅完成官方安装不会自动获得我们的灵巧手映射和保存格式。

## 1. 一次性准备

**机器人系统前提**：按 NVIDIA 官方教程准备机载系统；AGX Orin 使用 JetPack 6＋TensorRT 10.7，Thor 使用 JetPack 7.1＋TensorRT 10.13。先核对实际型号、L4T 和 TensorRT，不在本 SOP 中升级 CUDA、驱动或刷机。不同平台的 TensorRT engine 不直接混用，在目标机生成。

机器人先检查系统与已安装组件（只读）：

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core libnvinfer10 libnvinfer-dev
nvcc --version
```

若使用 TensorRT TAR 安装而非系统包，`dpkg-query` 查不到它是正常的；检查实际安装目录的 `include/NvInferVersion.h`。已有符合要求的安装直接复用。新机器人若尚未准备 JetPack/CUDA，需由设备维护者先完成基础系统准备；本文从平台已就绪开始，不包含刷机或驱动更换。

机器人及PC准备基础工具（Ubuntu）：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux
git lfs install
```

PC与机器人均需uv；若尚未安装，在对应机器执行官方uv安装器，再加载PATH：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

机器人在 `/home/unitree/VLA` 下克隆，PC在自己选定的项目父目录下克隆。以下只在机器人执行：

```bash
mkdir -p /home/unitree/VLA
cd /home/unitree/VLA
```

机器人和 PC 均取得我们的固定代码版本（需要仓库读取权限；SSH key 配置由各自帐号完成）：

```bash
git clone git@github.com:goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 4ad2687808b44e3ef126b998fc3fc0972121f6e4
git submodule update --init --recursive
git lfs pull
```

推荐机器人放到 `/home/unitree/VLA/GR00T-WholeBodyControl`；PC 可以自行选目录。后文 PC 命令均从项目根目录执行。已有部署可复用，但必须确认启动器指向的机器人目录和代码版本一致。

**机器人：TensorRT 路径、C++ 依赖和编译。** 官方原生安装依次运行依赖脚本、加载环境、编译。若 TensorRT TAR 已解压到自己的 `~/TensorRT`，先执行下面两行；实际安装在别处时替换路径。使用JetPack系统TensorRT时不要把它指向一个不存在的目录，应沿用系统安装与脚本检测。

```bash
export TensorRT_ROOT="$HOME/TensorRT"
test -f "$TensorRT_ROOT/include/NvInferVersion.h"
```

确认前面的CUDA检查已通过，再执行官方依赖安装步骤。此脚本在发现CUDA缺失时会尝试安装，基础CUDA未就绪时应停止并交由设备维护者处理；不要让它替换现有CUDA环境。源码位置为 `gear_sonic_deploy/scripts/install_deps.sh`。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
bash scripts/install_deps.sh
```

依赖齐全后编译；官方ROS2为可选依赖，本SOP采用ZMQ和Unitree DDS，不需要为了SONIC控制额外安装ROS2：

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

**PC：安装 PICO 和采集环境。** PC在项目根目录执行（脚本会重建对应环境，只用于首次安装；固定版本的Python由脚本选择，不能拿最新官网的Python版本说明强制覆盖）：

```bash
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
uv pip install --python .venv_teleop/bin/python flask
```

**PC：安装并启动XRoboToolkit PC Service。** 以下下载地址来自官方PICO教程，适用于x86_64 Ubuntu 22.04；文件保存在项目自己的目录中。Ubuntu 24.04将文件名内的 `22.04` 两处改为 `24.04`，不要装错系统版本。

```bash
mkdir -p external_dependencies/downloads
wget -P external_dependencies/downloads https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
sudo apt-get install ./external_dependencies/downloads/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

安装后从桌面应用菜单打开XRoboToolkit PC Service并保持运行；具体启动入口随安装包版本变化，以已安装应用为准。PC Service装在PC，本SOP不要求在机器人再装一份。

**PICO：一次性配对与校准。** 按[官方PICO教程](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)完成以下操作：

1. 准备PICO 4/4 Pro、双手柄、双脚踝追踪器；追踪器指示灯朝上且不被衣物遮挡。在头显Developer设置中按官方要求关闭Safeguard。
2. Motion Tracker应用中清除旧配对，长按追踪器按钮约6秒进入红蓝闪烁配对模式，配对两只追踪器；按Calibrate，先站直、手柄垂于两侧，再低头让头显识别脚踝追踪器。
3. 在头显浏览器进入[XRoboToolkit PICO发布页](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)，安装官方教程指定的 `XRoboToolkit-PICO-1.1.1.apk`，从应用库Unknown栏目打开。
4. PC与头显接入同一Wi-Fi；在头显应用的PC Service中填写PC的 **Wi-Fi IPv4地址**，连接后状态须为WORKING。勾选Head、Controller，Data/Control选Send，Pico Motion Tracker选Full body。

PC用以下命令查看地址和机器人路由；头显填写Wi-Fi地址，机载启动器填写机器人有线网卡地址，两者不要混淆：

```bash
ip -br -4 addr
ip -4 route get 192.168.123.164
ssh unitree@192.168.123.164 hostname
```

PC配置到机器人SSH免密登录（将PC已有公钥安装到机器人即可，不复制私钥）。网络对应：机器人连接PC的TCP 5556获取PICO输入，PC连接机器人的5557获取反馈、5555获取图像；启用防火墙时放通这三条路径。官方跨主机参数分别是 `--zmq-host` 和 `--zmq_feedback_host`，本项目启动器负责生成它们。确认实际网卡及两手IP，下面示例值按本机替换；左右手物理归属须分别核实。

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

首次控制顺序：站直、双脚并拢、大臂贴体、小臂向前弯曲90度、掌心相对；按A+B+X+Y启动策略并校准，等待启动稳定，再将手臂与机器人当前姿态对齐，按A+X进入POSE全身遥操作。再按A+X返回PLANNER模式，按A+B+X+Y停止策略；C++控制终端的O也是官方停止入口。PLANNER不是停机状态，需确认已停止再退出。

将焦点放到数据导出 pane：按 c 开始录制，再按 c 结束并保存；录制中按 x 丢弃当前段。每批使用新 dataset-name，不默认启用 `--overwrite-existing-dataset`。停止时先按控制界面提示停止身体控制，再退出进程，勿把 tmux 断开视为停止机器人。

## 4. 保存与交付验收

PC 输出位于项目 `outputs/<dataset-name>/`。检查 meta、data、videos：`observation.state`、`action.wbc` 各41维，身体/左手/右手顺序正确，motion token及teleop字段存在，图像可解码，episode帧数与视频一致。缺少状态或图像时不算采集验收通过；记录50Hz不代表相机有50张独立曝光。

交付代码固定提交、上面三个模型文件与 planner、手姿态 JSON、双方网卡/IP表、依赖版本清单及本 SOP。虚拟环境目录不能视为可移植安装包；保留环境定义与原始权重，在目标平台重新构建。无需安装训练环境或 Isaac Lab。

依据：本项目 `launch_data_collection.py`、`run_data_exporter.py`、`features_sonic_vla.py`、`inspire_modbus_hand.py`、`inspire_hand_web.py`；[官方机载安装](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)、[官方模型下载](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)。

核查记录（2026-11-10）：上述CLI、构建入口和数据维度已按本地源码核对；机器人SSH连接超时，未现场验证旧 `ctrl_hand.py`、新机全新安装或真实控制。本机找到历史机载代码工作区，未找到可确认版本的完整机器人系统/环境归档；因此本 SOP 不将其当作已验证镜像。接收方第一次安装仍须完成一次现场验收。
