# SONIC v1.1 机载推理＋强脑 Revo2 手采集 SOP

**GitHub 无法访问时：**先使用 [本机下载包与 PC→机器人安装说明](../tools/github_bundle/README_ZH.md)。它替代本文第 2～7 步的下载和安装入口，包含旧 GLIBC 的 RealSense 源码编译路径；不要再混用本文原来的在线安装脚本。网络、调手、采集和验收仍按本文对应步骤，并使用下载包说明里的强脑服务路径。APT/PyPI 和 HF Mirror 仍需联网。

按第0～10步顺序操作。首次安装完成全部步骤；以后从第9步开始。官方链接只作来源参考，不需要另外跟随另一套教程。

机器人运行SONIC v1.1、相机和**宇树官方BrainCo串口→DDS服务**；PC运行PICO和数据导出。本版只适配官方服务支持的强脑Revo2，不使用Inspire的手IP、TCP6000桥接或ctlhand.py，也不使用Dex3手接口。

数据仍为LeRobot格式，state/action各41维：身体29＋左手6＋右手6。强脑手为**0张开、1闭合**，与旧Inspire配置不能混用。

代码块逐行执行，末尾反斜杠表示下一行继续。机器人终端指本机终端或已SSH登录的终端；PC终端是电脑自己的终端。每次新开终端先执行相应的cd。某步失败先处理错误，再继续。

## 0. 查清参数并记录

| 参数 | 本文示例 | 如何确认 |
| --- | --- | --- |
| 机器人SSH帐号/IP | unitree / 192.168.123.164 | 0.1 |
| 机器人运动控制/DDS网卡 | eth0 | 0.2 |
| PC有线IPv4 | 192.168.123.222 | 0.3 |
| PC Wi-Fi IPv4 | 由本机查询 | 0.3 |
| 机器人项目目录 | /home/unitree/VLA/GR00T-WholeBodyControl | 第2步创建；已有目录用pwd -P确认 |
| PC项目目录 | $HOME/sonic_v11/GR00T-WholeBodyControl | 第2步创建 |

后文所有示例地址、帐号和路径都替换为此表的实际值。**强脑手没有本SOP需要填写的左右手网络IP。**

### 0.1 机器人：查帐号和机载IP

在机器人本地终端或已有SSH连接中执行：

```bash
whoami
hostname
ip -br -4 addr
```

whoami是SSH帐号；hostname只是机器名称。记录PC能够连接的网卡IPv4，去掉/24等掩码后缀。不能登录且不知道IP时，先从机器人本地终端查询，或向设备管理员取得机载地址。

### 0.2 机器人：确认DDS网卡及手部接线

```bash
ip -br -4 addr
ip -br link
ls -l /dev/serial/by-id/ 2>/dev/null
ls /dev/ttyUSB* /dev/ttyHAND* /dev/ttyUN* 2>/dev/null
```

运动控制网卡以这台机器人原有可用启动命令/接线说明为准，不一定等于SSH网卡。若只知道运动控制器IP：

```bash
read -r -p "已确认的运动控制器IPv4地址: " CONTROLLER_IP
ip -4 route get "$CONTROLLER_IP"
```

结果dev后的网卡作为候选，后面必须收到实际身体状态才算验证。PC与机器人若分别运行DDS程序，需要配置正确的有线DDS网络；本SOP把手DDS收发集中在机器人上。

每只Revo2通过独立USB串口连接机器人；官方服务扫描ttyUSB/ttyHAND/ttyUN，按设备SKU和手ID区分左右，不按USB编号大小猜。官方当前实现的左手ID是0x7e，右手0x7f，波特率460800；这里不要求操作者修改固件ID。第6步检查官方服务日志中两侧绑定的端口、SKU、序列号；第8步再以实际响应确认侧别。

### 0.3 PC：查有线和Wi-Fi地址

接好机器人网线：

```bash
read -r -p "机器人机载IPv4地址: " ROBOT_IP
ip -br -4 addr
ip -4 route get "$ROBOT_IP"
```

例如结果为“dev enp7s0 src 192.168.123.222”，则src后的地址填入第9步的offboard-zmq-host。若走了VPN或Wi-Fi，先修正有线连接。

查Wi-Fi：

```bash
for netdev in /sys/class/net/*; do
  if [ -d "$netdev/wireless" ]; then
    ip -br -4 addr show dev "${netdev##*/}"
  fi
done
```

把Wi-Fi的IPv4填入PICO的PC Service；它不是机器人的地址。没有输出时，在Ubuntu“设置→Wi-Fi→当前网络→详情”查看。

**完成标准：**帐号、地址、DDS网卡和两端目录已记录；手接到机器人USB串口。此步只查信息，不发送动作。

## 1. 机器人：核对系统基础

```bash
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core libnvinfer10 libnvinfer-dev
nvcc --version
```

机载基础：AGX Orin使用JetPack6＋TensorRT10.7，AGX Thor使用JetPack7.1＋TensorRT10.13。TAR安装TensorRT可能没有系统包记录，以安装目录include/NvInferVersion.h为准。

**完成标准：**CUDA/JetPack/TensorRT已匹配安装。本SOP不刷机、不更换驱动/CUDA；基础系统缺失时先由设备维护者准备好再继续。Revo3或其他手型不按Revo2流程试运行。

## 2. 机器人和PC：安装工具，下载本交付版

先在机器人和PC各执行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl wget tmux nano openssh-client cmake build-essential
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

机器人执行：

```bash
mkdir -p /home/unitree/VLA
cd /home/unitree/VLA
git clone --branch feat/brainco-hand-onboard https://github.com/goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git submodule update --init --recursive
git lfs pull
git branch --show-current
git rev-parse HEAD
```

PC执行：

```bash
mkdir -p "$HOME/sonic_v11"
cd "$HOME/sonic_v11"
git clone --branch feat/brainco-hand-onboard https://github.com/goldenstasteworriors/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git submodule update --init --recursive
git lfs pull
git branch --show-current
git rev-parse HEAD
```

**完成标准：**两端分支均为 feat/brainco-hand-onboard，并记录相同的提交号。不要覆盖已有同名目录，另选新父目录并同步改后续路径。旧 tag brainco-onboard-v1-20261110 保留作初始版本回滚；本文安装入口使用包含后续修复的强脑分支。

## 3. 机器人：编译SONIC推理

先查TensorRT位置：

```bash
printf '%s\n' "${TensorRT_ROOT:-未设置TensorRT_ROOT}"
dpkg-query -L libnvinfer-dev 2>/dev/null | grep '/NvInferVersion.h$'
```

TAR安装且忘记目录时：

```bash
find "$HOME" /opt /usr/local -maxdepth 4 -name NvInferVersion.h -print 2>/dev/null
```

例如找到/home/unitree/TensorRT/include/NvInferVersion.h，根目录就是/home/unitree/TensorRT。仅TAR安装时设置实际根目录（系统包安装跳过）：

```bash
export TensorRT_ROOT="$HOME/TensorRT"
test -f "$TensorRT_ROOT/include/NvInferVersion.h"
```

第1步基础完整后执行；依赖脚本会在CUDA缺失时尝试安装，基础缺失不要继续运行：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy
bash scripts/install_deps.sh
source scripts/setup_env.sh
just build
test -x target/release/g1_deploy_onnx_ref && echo "推理程序已生成"
ldd target/release/g1_deploy_onnx_ref
```

**完成标准：**编译成功，ldd无not found。本步不启动控制；新版本必须重新编译，不能继续使用以前只支持Inspire的旧二进制。

旧版 JetPack / Ubuntu 20.04 的默认 GCC 9 不满足本项目 C++20 要求。使用下载包的 `robot-build` 步骤时，更新后的安装器会在项目 `.runtime/sonic-toolchain` 中准备 GCC 10，不替换系统编译器；强脑分支已对 TensorRT 8.5 的数据类型差异作条件编译。只安装相机和手服务不会生成 SONIC 推理程序，必须完成本步并看到“推理程序已生成”，再启动采集。

已有旧下载包时，先更新强脑分支代码，再用项目中的新安装器复用原包。以下在机器人逐行执行；第二行末尾反斜杠表示与下一行连成一条命令，下载包目录按实际位置替换：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
SONIC_BUNDLE_ROOT=/home/unitree/VLA/sonic-brainco-github-bundle-20261111 \
  bash tools/github_bundle/install.sh robot-build "$PWD"
```

## 4. 机器人：安装相机环境及v1.1模型

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash install_scripts/install_camera_server.sh
```

询问安装开机相机服务时输入n，本文在第9步手动启动。继续：

```bash
uv pip install --python .venv_camera/bin/python pyrealsense2 huggingface_hub
.venv_camera/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
.venv_camera/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GEAR-SONIC', allow_patterns=['sonic_v1_1/model_encoder.onnx','sonic_v1_1/model_decoder.onnx','sonic_v1_1/observation_config.yaml'], local_dir='gear_sonic_deploy/policy')"
.venv_camera/bin/python download_from_hf.py
sha256sum gear_sonic_deploy/policy/sonic_v1_1/* gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

最后一条下载命令补齐planner；控制仍显式选v1.1三件套。保存模型版本与SHA256，不混用encoder、decoder、观测配置。

**完成标准：**真实相机序列号非空，模型与planner下载成功。pyrealsense2在目标ARM64平台无可用安装时需先完成RealSense SDK适配，不能跳过出图验收。相机安装脚本会重建环境，只在首次安装运行。

### 4.1 当前 TensorRT 8.5：生成兼容 planner

当前旧版机载环境需将官方 planner 的 LayerNormalization 展开、折叠固定形状计算，并展开五维 Einsum 的省略号，解决 TensorRT 8.5 的导入错误。以下在 **PC** 逐行执行；已有 `.venv_model_tools` 时跳过创建环境的那一行。原始模型保留，生成新文件，控制策略仍使用原 SONIC v1.1 encoder/decoder。

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
mkdir -p gear_sonic_deploy/planner/target_vel/V2
scp unitree@192.168.123.164:/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx gear_sonic_deploy/planner/target_vel/V2/
uv venv .venv_model_tools --python 3.10
uv pip install --python .venv_model_tools/bin/python numpy==1.26.4 onnx==1.17.0 onnxruntime==1.17.3
.venv_model_tools/bin/python tools/model_compat/prepare_planner_trt85.py gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx gear_sonic_deploy/planner/target_vel/V2/planner_sonic_trt85.onnx
scp gear_sonic_deploy/planner/target_vel/V2/planner_sonic_trt85.onnx unitree@192.168.123.164:/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/
```

第9步的启动命令使用此兼容文件。TensorRT 10 环境可使用官方原始 planner，并将第9步的 planner 文件名改回 `planner_sonic.onnx`。首次加载会生成本机 TensorRT 缓存，需等待构建完成，不能在此过程中反复重启。

## 5. PC：安装采集环境及PICO

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
bash install_scripts/install_pico.sh
bash install_scripts/install_data_collection.sh
.venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --help
.venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py --help
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py --help
```

安装脚本只首次执行，会重建venv。网络中断后即使 `.venv_teleop` 目录已经存在，也不代表安装完成；必须让以上三个帮助命令全部成功。PICO 环境还必须包含编译好的 `xrobotoolkit_sdk`，不能只创建空虚拟环境。PC安装XRoboToolkit PC Service（x86_64 Ubuntu22.04或24.04）：

```bash
. /etc/os-release
case "$VERSION_ID" in 22.04|24.04) ;; *) echo "仅适用Ubuntu22.04/24.04"; exit 1 ;; esac
mkdir -p external_dependencies/downloads
wget -P external_dependencies/downloads "https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
sudo apt-get install "./external_dependencies/downloads/XRoboToolkit_PC_Service_1.0.0_ubuntu_${VERSION_ID}_amd64.deb"
```

打开桌面应用XRoboToolkit PC Service并保持运行。头显按顺序：

1. 两只脚踝追踪器指示灯朝上，避免衣物遮挡；Developer设置关闭Safeguard。
2. Motion Tracker清除旧配对，长按追踪器按钮约6秒到红蓝闪烁后配对。
3. 按Calibrate：先站直，手柄垂于两侧，再低头让头显识别脚踝追踪器。
4. 安装[XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk)，从Unknown应用库打开。
5. 头显与PC连同一Wi-Fi，PC Service填第0步的PC Wi-Fi IPv4；勾选Head、Controller，Data/Control选Send，Motion Tracker选Full body。

**完成标准：**三个帮助命令正常，头显状态WORKING，全身和手柄追踪正常。

## 6. 机器人：安装并启动宇树官方强脑服务

本步所有命令均在机器人执行，不启动其他手厂商的串口服务。先安装官方服务需要的系统依赖及Unitree SDK2；这里安装的是通信库，不更换GPU/CUDA：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
sudo apt-get install -y libspdlog-dev libfmt-dev libboost-program-options-dev libyaml-cpp-dev
cmake -S gear_sonic_deploy/thirdparty/unitree_sdk2 -B .runtime/brainco-sdk-build -DBUILD_EXAMPLES=OFF
cmake --build .runtime/brainco-sdk-build -j 4
sudo cmake --install .runtime/brainco-sdk-build
sudo ldconfig
git clone https://github.com/unitreerobotics/brainco_hand_service.git external_dependencies/brainco_hand_service
git -C external_dependencies/brainco_hand_service checkout d71996b6999edb2f838a3dca3d9621429a2ef966
cmake -S external_dependencies/brainco_hand_service -B .runtime/brainco-service-build
cmake --build .runtime/brainco-service-build --target brainco_hand_server -j 4
ldd external_dependencies/brainco_hand_service/bin/brainco_hand_server
```

确认ldd无not found。若已有官方brainco_hand.service，先查看状态，已有正常服务时复用，不重复开串口：

```bash
systemctl status brainco_hand.service --no-pager
```

没有服务时在机器人终端A手动启动（替换DDS网卡），并保持运行：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl/external_dependencies/brainco_hand_service
sudo ./bin/brainco_hand_server --network_interface eth0
```

本版本实际参数名是network_interface；按源码使用，不混用其他版本命令。官方服务启动时默认q=0（张开），需在允许张手的现场条件下启动。本SOP不运行官方循环握拳测试程序，也不自动配置开机启动。

**完成标准：**日志分别出现left/right绑定端口和设备SKU/序列号，两个worker正常运行。只找到一只手、无串口或打不开设备时先处理接线/设备适配，不把缺失手当作零值继续采集。

## 7. PC：配置SSH；机器人安装手部网页环境

PC先确认SSH（替换帐号/IP）：

```bash
ssh unitree@192.168.123.164 hostname
ls "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" 2>/dev/null
```

没有公钥时执行ssh-keygen；已有则跳过，不覆盖已有密钥：

```bash
ssh-keygen -t ed25519
```

将公钥装到机器人：

```bash
ssh-copy-id unitree@192.168.123.164
ssh -o BatchMode=yes unitree@192.168.123.164 hostname
```

密钥有口令而BatchMode失败时，PC加载密钥后重试（文件名用自己的实际密钥）：

```bash
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519"
ssh -o BatchMode=yes unitree@192.168.123.164 hostname
```

后续从这个终端启动采集，使tmux继承agent。每次重新登录PC后检查BatchMode。启用防火墙时放通机器人→PC5556、PC→机器人5557和5555。

机器人安装项目内的独立手控环境：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash tools/brainco_hand_web/install_env.sh
bash tools/brainco_hand_web/run.sh --network eth0 --check-seconds 5
```

替换eth0为官方服务使用的DDS网卡。check-seconds只读状态，不发命令；运行时第6步服务必须保持运行。

安装脚本保留已有虚拟环境，自动在项目内补齐CycloneDDS 0.10.2开发头文件及Python绑定，不更换系统DDS。已有离线源码时可把压缩包路径作为安装脚本参数；具体见tools/brainco_hand_web/README.md。启动脚本自动设置库路径，不必手输环境变量。

**完成标准：**PC免密检查成功；机器人打印左右各6个q、speed、current_a及age_s。q/speed为归一化量；current_a是电流A，不是力矩。检查超时则回第6步检查服务和DDS网卡。

## 8. 调节左右手，保存并同步配置

**8.1 PC上传当前配置**到机器人，避免重复调节时加载旧配置。首次使用也是执行同一条命令：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
scp gear_sonic/config/data_collection/brainco_hand_pose.json unitree@192.168.123.164:/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic/config/data_collection/brainco_hand_pose.json
```

**8.2 机器人终端B启动调节网页**，第6步官方服务仍运行；此时不要启动SONIC采集：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
bash tools/brainco_hand_web/run.sh --network eth0 --enable-commands
```

**8.3 在采集PC本地另开终端转发并保持运行：**PC端使用18080，避免与已有HMI的8080冲突；机器人网页仍监听127.0.0.1:8080。以下命令在PC执行，不能在机器人SSH终端执行。

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:127.0.0.1:8080 unitree@192.168.123.164
```

命令保持前台运行、不返回提示符是正常现象。若出现Address already in use或转发失败，先处理报错，不能把浏览器已有页面当作连接成功。无需停止已有HMI或点击其控制按钮。

PC浏览器新开标签页，输入完整地址：

```text
http://127.0.0.1:18080/
```

**页面验收：**标题为“强脑 Revo2 双手调节”，显示左右手各六个滑块及真实反馈。若出现/HMI/tele-operation、启动VR、开始控制等界面，说明打开了其他应用，不能继续当作本工具操作。若页面连接失败，检查机器人第8.2步网页进程是否仍运行。

分别选择左右手，轻微调整一根手指，点击“发送此手目标”，通过实际响应确认左右。官方服务已按SKU分侧；若响应侧别不符，检查设备/官方服务绑定，不通过交换任意数组来掩盖问题。

网页每侧顺序固定为：**拇指、拇指副指、食指、中指、无名指、小指**。值域0～1，0张开、1闭合；无需除以1000，也不要套用Inspire的反向数值。speed默认1.0，遵循官方建议。

调到合适张开目标后点“保存为此手张开”，调到抓握目标后点“保存为此手抓握”，左右分别保存。拖动、加载目标不发命令；“发送”只控制一侧，“保存”只写配置。反馈超过0.25秒或未收到反馈时网页拒绝发命令。

**8.4 PC把保存结果下载回来：**

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
scp unitree@192.168.123.164:/home/unitree/VLA/GR00T-WholeBodyControl/gear_sonic/config/data_collection/brainco_hand_pose.json gear_sonic/config/data_collection/brainco_hand_pose.json
.venv_data_collection/bin/python -m json.tool gear_sonic/config/data_collection/brainco_hand_pose.json
```

确认hand_type是brainco_revo2，joint_names顺序未改，left/right各有open/grasp六个0～1目标，speed在(0,1]。配置示例：

```json
{
  "hand_type": "brainco_revo2",
  "joint_names": ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"],
  "speed": 1.0,
  "left": {"open": [0, 0, 0, 0, 0, 0], "grasp": [1, 1, 1, 1, 1, 1]},
  "right": {"open": [0, 0, 0, 0, 0, 0], "grasp": [1, 1, 1, 1, 1, 1]}
}
```

示例为全开/全闭范围，实际目标以调节结果为准。记录任务、日期和配置。完成后Ctrl+C关闭网页进程与PC隧道，**官方手服务保持运行**。采集时以PC配置为准，启动器将它上传为机器人.runtime/brainco_hand_pose.json后由C++读取；修改配置须停止并重新启动采集，不会热更新。

**完成标准：**左右实际响应与网页侧别相符，PC配置已同步，网页已退出，两个官方手worker仍运行。

## 9. 每次采集：官方手服务→相机→PICO→采集

先确保第6步官方手服务运行、第7步免密成功。调手网页与官方test_brainco_hand_server测试程序必须退出，避免多个命令来源。

**9.1 机器人新终端启动相机，保持运行：**

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

只运行一份相机服务。若系统有多个 RealSense 设备条目或默认枚举遇到异常设备，可指定目标相机的 **SDK 序列号**；从相机启动日志的 `Serial number` 行确认，不用 `/dev/video` 编号或 USB 路径代替。将下一条命令中的文字替换为实际序列号，再整行执行，代替上一条启动命令：

```bash
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --ego-view-device-id "替换为SDK序列号" --port 5555
```

启动日志会打印目标相机的 USB type：`3.x` 为 USB 3，`2.x` 为 USB 2。看到画面后至少观察 3 分钟，确认持续更新且没有连续取帧超时。若卡住，先看机器人端日志：本机取帧也超时应排查相机/USB 链路，指定序列号不能修复线缆、供电或设备通信故障。更换接口后重新确认速度，不能只根据接口颜色判断。

**9.2 PC打开PC Service并连接PICO，然后检查画面；连续更新后按q退出：**

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python gear_sonic/scripts/run_camera_viewer.py --camera-host 192.168.123.164 --camera-port 5555
```

**9.3 PC启动全身采集**，替换第0步参数、任务提示与新的dataset-name。帐号不是unitree时额外加deploy-onboard-user实际帐号；不需要手部IP，也不启动Inspire桥接：

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
.venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py \
  --hand-type brainco \
  --brainco-hand-pose-config gear_sonic/config/data_collection/brainco_hand_pose.json \
  --brainco-max-state-age-s 0.25 \
  --camera-host 192.168.123.164 \
  --deploy-onboard --deploy-onboard-host 192.168.123.164 \
  --deploy-onboard-repo-root /home/unitree/VLA/GR00T-WholeBodyControl \
  --deploy-onboard-interface eth0 --offboard-zmq-host 192.168.123.222 \
  --deploy-checkpoint policy/sonic_v1_1/model \
  --deploy-obs-config policy/sonic_v1_1/observation_config.yaml \
  --deploy-planner planner/target_vel/V2/planner_sonic_trt85.onnx \
  --task-prompt "pick up the box" --dataset-name pick_box_brainco_20261110_01 \
  --enable-sonic-publish --no-pico-vis-smpl --no-pico-vis-vr3pt \
  --profile-timing
```

启动器先验证并上传手配置，失败会停止；成功后才启动控制。它使用当前PC配置的左右独立开合目标，PICO开合意图选择open/grasp。C++记录实际成功发布的六电机目标，而不是导出端重新猜测开合。

tmux的data_collection窗口：左上C++推理，左下PICO，右上数据导出，右下相机。鼠标点pane切换输入；Ctrl+B松开再按方向键也可切换。各pane不能有Traceback或持续连接失败。只显示“All components launched”不算验收通过。

现场确认机器人已准备好，再在C++ pane按y，接着：

1. 站直、双脚并拢、大臂贴体、小臂向前弯90度、掌心相对。
2. A+B+X+Y启动策略并校准；等待稳定、对齐手臂，A+X进入POSE遥操作。
3. 确认两手反馈正常，在导出pane按c开始，短暂操作，再按c结束保存；录制中x丢弃。
4. A+X回PLANNER，A+B+X+Y停止策略；C++终端O也可停止。PLANNER不是停机状态，确认已停止再退出进程。

停止身体控制/退出程序不会保证手自动松开；官方服务可保留最后目标。需要松手时，在身体控制结束、无其他手命令来源后，用第8步网页明确发送张开目标。

单侧反馈超过0.25秒、机器人状态流陈旧或手命令尚未成功发布时，导出器暂停添加帧并打印原因，不伪造缺失手状态。它不会因此自动改变身体控制策略。每批使用新dataset-name，不开启覆盖已有数据。

终端断开后PC重新进入（不重新启动控制）：

```bash
tmux attach -t sonic_data_collection
```

## 10. PC检查保存结果

```bash
cd "$HOME/sonic_v11/GR00T-WholeBodyControl"
ls outputs/pick_box_brainco_20261110_01
.venv_data_collection/bin/python -m json.tool outputs/pick_box_brainco_20261110_01/meta/info.json
```

至少录制一条短episode验收：meta/data/videos存在、视频可播放、帧数对应；info中的schema_compatibility为g1_brainco_revo2_41dof（位于script_config），state/action各41维，末12维按左手6＋右手6和官方motor顺序命名。

额外字段：observation.brainco_hand_speed为12维归一化速度；observation.brainco_hand_current为12维电流A；observation.brainco_state_age_s为左右两侧反馈接收年龄。tau_est在官方服务里是电流，**不是力矩**。年龄是机器人收到DDS后的单调时钟间隔，不是硬件采样/曝光时间。EEF保留左右wrist_yaw_link坐标，不表示强脑指尖。50Hz记录也不代表相机有50次独立曝光。

交付本版本代码、官方服务版本、v1.1三件套/planner及校验值、手配置、网络表、环境记录和本SOP。不要把新强脑数据与旧Inspire数据当作相同手部语义直接混合。

## 参考与验证范围

- [用户指定的宇树强脑手文档](https://support.unitree.com/home/zh/G1_developer/brainco_hand)。
- [宇树官方服务与配置](https://github.com/unitreerobotics/brainco_hand_service/tree/d71996b6999edb2f838a3dca3d9621429a2ef966)、[实际读写字段源码](https://github.com/unitreerobotics/brainco_hand_service/blob/d71996b6999edb2f838a3dca3d9621429a2ef966/main.cpp)、[六电机命令示例](https://github.com/unitreerobotics/brainco_hand_service/blob/d71996b6999edb2f838a3dca3d9621429a2ef966/test/test_brainco_hand_server.cpp)。
- [SONIC机载安装](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)、[模型下载](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)、[PICO配置](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)。

2026-11-10：宇树文档站正文抓取为空、浏览器访问被拦截，因此接口以同属宇树官方的上述固定服务源码和示例交叉核验。已完成x86_64主程序构建、隔离回环DDS、网页配置保存、41维导出及启动命令验证；未连接真实机器人，Jetson全新安装、串口识别和实际手动作仍需现场验收。具体结果见BRAINCO_HAND_VALIDATION.md。
