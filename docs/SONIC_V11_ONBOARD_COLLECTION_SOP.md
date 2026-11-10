# SONIC v1.1 机载推理与 Inspire 数据采集 SOP

**按第0～10步依次操作，不需要另外执行官方教程。** 首次安装完成第0～8步；以后每天从第9步开始。官方资料只用于说明来源，放在文末。

部署分工：机器人运行SONIC v1.1推理和RealSense相机；PC运行PICO、Inspire手桥接和数据导出。数据沿用本项目41维格式（身体29＋左手6＋右手6）。

代码块逐行执行，某行失败先处理该错误，再继续；行末 `\` 表示命令在下一行继续。以下步骤用于新目录，已有环境不要重复安装。

## 0. 确认设备和填写参数

准备：G1＋机载Jetson、Inspire双手、RealSense、Ubuntu x86_64 PC、PICO 4/4 Pro、双手柄、双脚踝追踪器；双方需要联网下载依赖及本项目Git仓库读取权限。

先记录下表。下文命令使用“示例值”；不同的地方在运行前全部替换，左右手IP必须按物理侧别核实。

| 项目 | 示例值 |
| --- | --- |
| 机器人SSH用户 / IP | unitree / 192.168.123.164 |
| 机器人连接运动控制网络的网卡 | eth0 |
| PC连接机器人的网卡 / IPv4 | enp7s0 / 192.168.123.222 |
| PC Wi-Fi IPv4（填入PICO） | 用下面命令查看 |
| 左手 / 右手IP | 192.168.123.210 / 192.168.123.211 |
| 机器人项目目录 | /home/unitree/VLA/GR00T-WholeBodyControl |
| PC项目目录 | $HOME/sonic_v11/GR00T-WholeBodyControl |

PC查看地址及有线路由：

```bash
ip -br -4 addr
ip -4 route get 192.168.123.164
```

**完成标准：**已确认有线通信参数和PC Wi-Fi地址；PICO连接PC的Wi-Fi地址，机载程序连接PC的有线地址。

## 1. 机器人：检查系统基础

在机器人终端执行：

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core libnvinfer10 libnvinfer-dev
nvcc --version
```

支持的基础：AGX Orin使用JetPack 6＋TensorRT 10.7，AGX Thor使用JetPack 7.1＋TensorRT 10.13。TAR安装的TensorRT可能查不到系统包，以其安装目录的 `include/NvInferVersion.h` 为准。

**完成标准：**基础系统、CUDA及TensorRT已安装且版本匹配。本SOP从可用的机载系统开始，不包含刷机、CUDA或驱动更换；若缺失，先由设备维护者准备基础系统，再继续。不能拿不匹配的TensorRT engine直接运行。

## 2. 机器人和PC：安装工具，取得同一份代码

先分别在机器人和PC各执行一次：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux nano openssh-client
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

机器人执行：

```bash
mkdir -p /home/unitree/VLA
cd /home/unitree/VLA
git clone git@github.com:goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 4ad2687808b44e3ef126b998fc3fc0972121f6e4
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

PC执行：

```bash
mkdir -p "$HOME/sonic_v11"
cd "$HOME/sonic_v11"
git clone git@github.com:goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git checkout 4ad2687808b44e3ef126b998fc3fc0972121f6e4
git submodule update --init --recursive
git lfs pull
git rev-parse HEAD
```

**完成标准：**两端提交号均为上述固定提交，依赖仓库及LFS文件下载成功。若clone提示权限不足，由仓库管理员给接收方帐号授予读取权限并配置其SSH公钥后重试。

## 3. 机器人：安装C++依赖并编译推理程序

若TensorRT是TAR安装，先设置真实安装目录。下面示例仅适用于解压在机器人 `~/TensorRT` 的情况；JetPack系统包安装则跳过这两行，由环境脚本检测。

```bash
export TensorRT_ROOT="$HOME/TensorRT"
test -f "$TensorRT_ROOT/include/NvInferVersion.h"
```

在第1步已确认完整CUDA环境的机器人上执行：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
test -x target/release/g1_deploy_onnx_ref && echo "部署程序已生成"
```

依赖脚本在发现CUDA组件缺失时会尝试安装；基础组件缺失的机器不要继续跑此步骤，回到第1步处理。本流程使用ZMQ及Unitree DDS，无需额外配置ROS2。

**完成标准：**编译成功，显示“部署程序已生成”。本步不启动机器人控制。

## 4. 机器人：安装相机环境，下载v1.1模型

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

脚本询问是否安装开机相机服务时，**输入n**；本文统一在第9步手动启动，避免重复服务。继续：

```bash
uv pip install --python .venv_camera/bin/python pyrealsense2 huggingface_hub
.venv_camera/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
.venv_camera/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GEAR-SONIC', allow_patterns=['sonic_v1_1/model_encoder.onnx','sonic_v1_1/model_decoder.onnx','sonic_v1_1/observation_config.yaml'], local_dir='gear_sonic_deploy/policy')"
.venv_camera/bin/python download_from_hf.py
sha256sum gear_sonic_deploy/policy/sonic_v1_1/* gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

最后一个下载命令补齐planner等资源，实际控制在第9步显式选v1.1。模型encoder、decoder与观测配置三件套保持配套，不改网络和输入维度。将下载版本及SHA256保存到交付记录。

**完成标准：**打印出真实相机序列号，模型及planner存在且校验可读。若ARM64无法安装/导入pyrealsense2，需先完成匹配的RealSense SDK适配；不能跳过相机检查继续采集。

## 5. PC：安装PICO和数据采集环境

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
uv pip install --python .venv_teleop/bin/python flask
.venv_teleop/bin/python -c "import unitree_sdk2py, flask; print('手控制依赖正常')"
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py --help
```

**完成标准：**手控制依赖导入正常，启动器能显示帮助。安装脚本会重建对应venv，只在首次安装时执行。

## 6. PC和PICO：连接头显

**PC执行：**以下命令自动选择Ubuntu 22.04或24.04的x86_64安装包；其他PC平台不使用此命令。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
. /etc/os-release
case "$VERSION_ID" in 22.04|24.04) ;; *) echo "本步骤仅支持Ubuntu 22.04/24.04"; exit 1 ;; esac
mkdir -p external_dependencies/downloads
wget -P external_dependencies/downloads "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
sudo apt-get install "./external_dependencies/downloads/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
```

从PC桌面应用菜单打开XRoboToolkit PC Service，保持运行。

**头显依次操作：**

1. 双脚踝追踪器指示灯朝上，不被衣物遮挡；在Developer设置关闭Safeguard。
2. Motion Tracker应用清除旧配对，长按两只追踪器按钮约6秒进入红蓝闪烁配对，完成配对。
3. 按Calibrate：先站直、手柄垂于两侧，再低头让头显识别脚踝追踪器。
4. 头显浏览器打开[APK下载地址](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)，安装后从应用库Unknown打开。
5. PICO和PC接入同一Wi-Fi。在头显应用PC Service处填第0步记录的PC Wi-Fi地址；勾选Head、Controller，Data/Control选Send，Motion Tracker选Full body。

**完成标准：**头显连接状态为WORKING，手柄和全身追踪正常。

## 7. PC：检查到机器人的连接

```bash
ssh unitree@192.168.123.164 hostname
```

确保PC可以免密SSH登录，供启动器使用。尚未配置时，先准备PC自己的SSH公钥（若已有则复用），将公钥安装到机器人；不要复制私钥。PC已有默认公钥时可执行：

```bash
ssh-copy-id unitree@192.168.123.164
ssh -o BatchMode=yes unitree@192.168.123.164 hostname
```

**完成标准：**第二条命令直接显示机器人主机名，不询问登录密码。启用防火墙时还需允许：机器人→PC TCP5556（PICO输入），PC→机器人TCP5557（状态反馈）、5555（相机）。

## 8. PC：调灵巧手，并保存开合配置

调节前退出采集。本步骤统一使用仓库自带网页，无需在机器人安装额外手控程序。

**PC终端A**：将网卡和左右手IP替换为第0步的值，启动透传桥接并保持运行。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_teleop/bin/python decoupled_wbc/scripts/inspire_modbus_hand.py --mode dds --network enp7s0 --left-ip 192.168.123.210 --right-ip 192.168.123.211 --dds-pose-mode passthrough
```

**PC终端B**：启动网页并保持运行。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_teleop/bin/python decoupled_wbc/scripts/inspire_hand_web.py --network enp7s0 --host 127.0.0.1 --port 8080
```

**PC浏览器打开：**

```text
http://localhost:8080/
```

逐个调整并发送，记录张开、抓握各6个数值。顺序：**小指、无名指、中指、食指、拇指弯曲、拇指旋转**；值域0～1，通常0闭合、1张开。

打开PC项目中的配置文件：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
nano gear_sonic/config/data_collection/inspire_hand_pose.json
```

用实测值替换下面的示例，保存退出（nano：Ctrl+O、回车、Ctrl+X）：

```json
{
  "open": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
  "grasp": [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
}
```

**网页不会自动保存该JSON。** 当前配置对双手共用一套开合目标；左右若必须用不同目标，需要先扩展配置和桥接后再交付。记录任务、日期和调节值。

**完成标准：**网页发送后真实手响应正常，JSON已保存。完成后在终端A、B分别Ctrl+C退出，防止与采集手桥接冲突。

## 9. 每次使用：按顺序启动相机、检查图像、采集

**9.1 机器人终端：**接好相机并启动，保持运行。已有同端口相机服务则复用，不重复启动。

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

**9.2 PC终端：**先打开PC Service并连接PICO，再检查图像；确认画面连续更新后按q退出查看器。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```

**9.3 PC终端：**替换第0步参数、任务提示和新的dataset-name后执行。启动器会启动机器人推理及PC手桥接，不再另外开手桥接。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
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
  --profile-timing
```

现场确认机器人已准备好后，在deploy窗口按y允许部署。确认相机、头显、身体状态和手反馈正常，然后依次：

1. 站直、双脚并拢、大臂贴体、小臂向前弯曲90度、掌心相对。
2. A+B+X+Y启动策略并校准；等待稳定，对齐手臂，A+X进入POSE遥操作。
3. 焦点放到数据导出pane：c开始录制，再按c结束保存；录制中x丢弃当前段。
4. A+X返回PLANNER；A+B+X+Y停止策略，C++终端O也是停止入口。PLANNER不等于停机，确认停止后再退出程序；断开tmux不代表停机。

每批使用新dataset-name，不加覆盖已有数据的参数。

## 10. PC：检查数据后交付

打开项目输出目录（将dataset-name替换为本次名称）：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
ls outputs/pick_box_v11_20261110_01
```

**完成标准：**meta、data、videos均存在；视频可播放，帧数与episode对应；state/action为41维且顺序是身体29、左手6、右手6，motion token、teleop和modality信息保留。50Hz记录不代表相机有50张独立曝光。

交付代码固定提交、v1.1三件套及planner、模型版本/SHA256、手姿态JSON、网络参数表、依赖版本和本SOP。首次安装必须实际录制并检查一条短episode后，才算整套验收完成。

## 附录：已有机器人上的旧手控工具

仅为已使用旧工具的操作者保留，不属于首次安装的必做步骤；第8步已提供完整替代流程。旧工具存在时，可在机器人启动：

```bash
cd /home/unitree/VLA/locodemo
python ctrl_hand.py
```

PC转发8080端口后访问同一个本地网页：

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.164
```

不要同时启动第8步网页。旧工具若显示0～1000，除以1000后写回第8步JSON；若为物理角度单位则不能直接套用。旧工具也不保证自动保存到我们的配置。

## 参考与验证范围

官方资料仅作来源参考，无需再跟随其安装流程：[机载部署](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)、[模型](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)、[PICO配置](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)、[遥操作](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)。

2026-11-10：命令参数、数据维度及Shell/JSON语法按本项目源码核对；机器人连接超时，尚未完成干净环境安装和真机验收。适用范围是第1步已具备匹配基础系统的设备；未找到可确认版本的完整机器人环境归档，不将现有代码备份称为已验证镜像。
