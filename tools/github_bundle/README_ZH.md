# SONIC＋强脑：GitHub 下载包使用说明

日期：2026-11-11。适用于 x86_64 Ubuntu 22.04/24.04 采集 PC 和 ARM64 Jetson 机器人。

本包替代原 SOP 中的 GitHub 下载、依赖安装入口；APT、PyPI、HF Mirror 仍需联网。这不是系统镜像。不会刷机或升级 CUDA、驱动、GLIBC。所有命令按行依次执行，任一步失败就停止该组；不要把本说明与原安装脚本混着运行。

## 1. 包中有哪些东西

- 强脑分支部署源码快照，含已经还原的 Git LFS 资产、Unitree SDK2、Python SDK 和 XRoboToolkit Python SDK；不需要 git clone、git submodule update 或 git lfs pull。排除与本流程无关的 motionbricks/out 训练权重。
- 宇树 BrainCo 服务固定提交 d71996b6999edb2f838a3dca3d9621429a2ef966，含 ARM64/x86_64 的强脑库；仍使用原 SOP 的自动识别版本，并非另一个显式 `--id/--serial` 版本。
- RealSense v2.55.1，以及它实际要求的 pybind11 v2.10.3、pybind11_json 0.2.13、nlohmann/json v3.11.3、ev-mp/libusb 固定提交。
- GoogleTest v1.14.0、cppzmq v4.10.0、ARM64 ONNX Runtime 1.16.3、固定提交的 LeRobot。
- Ubuntu 22.04/24.04 的 XRoboToolkit PC Service deb、PICO v1.1.1 APK。
- ARM64/x86_64 的 uv 与 Python 3.10，避免 uv 安装 Python 时又访问 GitHub。
- `downloads/UPSTREAM.json` 记录下载地址和 SHA256；`SOURCE.json` 记录源码版本；`SHA256SUMS` 校验包内文件。

RealSense 改了四处依赖下载定义，使用本地同版本压缩包；另将 pyrealsense2 安装目标与该版本无法编译的废弃 pybackend2 分开，关闭后者。相机功能 C++ 源码未改。SONIC 网络、数据格式、强脑 DDS 协议未改。没有打包 CUDA/TensorRT、虚拟环境或 SONIC 模型。

## 2. 你的本机 → 对方 PC

把以下两个文件通过 U 盘、局域网或网盘发给对方，放到对方 PC 的 Downloads 目录（文件名保持不变）：

```text
sonic-brainco-github-bundle-20261111.tar.gz
sonic-brainco-github-bundle-20261111.tar.gz.sha256
```

解包约需 3.3 GiB；如果还要复制一份全新项目及编译，建议目标盘至少留 12 GiB。所有内容放在下列项目所在盘，不自动改用其他挂载盘。

对方在 PC 本地终端执行，每行单独执行。先校验，再解压；首次解压才运行 tar，不覆盖已有同名解包目录：

```bash
cd "$HOME/Downloads"
df -h .
sha256sum -c sonic-brainco-github-bundle-20261111.tar.gz.sha256
tar -xzf sonic-brainco-github-bundle-20261111.tar.gz
```

已有原 SOP 的项目时继续用它，不用复制源码覆盖。只有目标项目不存在时，才执行下面三行：

```bash
mkdir -p "$HOME/sonic_v11"
test ! -e "$HOME/sonic_v11/GR00T-WholeBodyControl"
cp -a "$HOME/Downloads/sonic-brainco-github-bundle-20261111/GR00T-WholeBodyControl" "$HOME/sonic_v11/"
```

第三行仅在第二行成功后执行。包里是源码快照，没有 .git，因此不使用 `git describe` 判定版本，以 SOURCE.json 和校验文件为准。已有项目应来自原交付标签 brainco-onboard-v1-20261110。

检查解包完整性：

```bash
cd "$HOME/Downloads/sonic-brainco-github-bundle-20261111"
bash tools/github_bundle/install.sh check "$HOME/sonic_v11/GR00T-WholeBodyControl"
```

## 3. 对方 PC → 机器人

以下仍在 PC 本地终端执行。示例机器人地址为 192.168.123.164，地址不同时替换。首次 SSH 按提示核对连接并输入机器人密码。

```bash
cd "$HOME/Downloads"
ssh unitree@192.168.123.164 'mkdir -p /home/unitree/VLA'
scp sonic-brainco-github-bundle-20261111.tar.gz sonic-brainco-github-bundle-20261111.tar.gz.sha256 unitree@192.168.123.164:/home/unitree/VLA/
ssh unitree@192.168.123.164
```

现在终端已在机器人，执行：

```bash
cd /home/unitree/VLA
df -h .
sha256sum -c sonic-brainco-github-bundle-20261111.tar.gz.sha256
tar -xzf sonic-brainco-github-bundle-20261111.tar.gz
```

如果原 SOP 项目已在 `/home/unitree/VLA/GR00T-WholeBodyControl`，保留原项目、环境、模型和手部配置。仅目标项目不存在时，执行下面两行（第一行成功才执行第二行）：

```bash
test ! -e /home/unitree/VLA/GR00T-WholeBodyControl
cp -a /home/unitree/VLA/sonic-brainco-github-bundle-20261111/GR00T-WholeBodyControl /home/unitree/VLA/
```

检查完整性：

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111
bash tools/github_bundle/install.sh check /home/unitree/VLA/GR00T-WholeBodyControl
```

后续必须保留两端的解包目录：编译、安装和部分 Python 包引用其中的源码。不要只复制脚本或在安装后移动/删除解包目录。

## 4. 机器人：替代原 SOP 第 3、4、6、7 步的安装部分

每个安装入口可单独运行，不启动身体或手部运动。已有虚拟环境会保留，按所选入口安装依赖；建议先完成当前出错的 camera。完整新装机器再运行其他入口。APT/PyPI 错误需要单独解决，不表示缺了 GitHub 包。

**4.1 相机（替代原第四步的相机安装）**

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111
bash tools/github_bundle/install.sh camera /home/unitree/VLA/GR00T-WholeBodyControl
```

该入口在机器人本地用现有 GLIBC 编译 RSUSB 和 Python 绑定，编译成功后才卸载不兼容的 pyrealsense2 wheel 并安装结果。不会安装 OAK/训练依赖。不要再执行原来的 `uv pip install pyrealsense2`，也不要重跑会删除环境的 install_camera_server.sh。

末尾需要打印实际相机序列号。若 import 成功但列表为空，检查相机是否被其他进程占用及 USB 权限。需要安装 USB 权限规则时，在没有相机采集进程运行时执行官方规则脚本，然后仅重新插拔相机 USB：

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111/sources/librealsense
bash scripts/setup_udev_rules.sh
```

随后复查相机：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
.venv_camera/bin/python -c "import pyrealsense2 as r; print([d.get_info(r.camera_info.serial_number) for d in r.context().devices])"
.venv_camera/bin/python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

最后一条保持相机服务运行；PC 按原 SOP 的 viewer 验证真实出图。枚举成功不等于帧率和出图已验证。

**4.2 SONIC 编译（替代原第三步安装和 just build）**

保持原来已可用的 JetPack/CUDA/TensorRT。若 TensorRT 是 TAR 安装，按原 SOP 查明目录，在同一个终端设置 TensorRT_ROOT 后再执行。未确认版本兼容性不能视为板载推理已验证。

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111
bash tools/github_bundle/install.sh robot-build /home/unitree/VLA/GR00T-WholeBodyControl
```

输出仍为原路径 gear_sonic_deploy/target/release/g1_deploy_onnx_ref。ldd 不能有 not found。该入口使用本地 GoogleTest 和 ONNX Runtime，无需 just，也不运行原 install_deps.sh（它还有在线下载及底层环境安装分支）。

**4.3 官方强脑服务及网页环境（替代原第六步安装和第七步 Python 环境安装）**

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111
bash tools/github_bundle/install.sh hand /home/unitree/VLA/GR00T-WholeBodyControl
```

Unitree SDK 安装到项目 .runtime/github-bundle/unitree-sdk。服务编译成功不等于实体手已识别。先按原 SOP 检查已有服务，避免重复占用串口。现场允许张手时，手动启动本包的官方服务（eth0 换成实际 DDS 网卡）：

```bash
cd /home/unitree/VLA/sonic-brainco-github-bundle-20261111/sources/brainco_hand_service
sudo ./bin/brainco_hand_server --network_interface eth0
```

必须看到左右手端口绑定成功；当前没有串口的问题仍需现场排查，本包不会改变设备接线或自动修复识别。另一个机器人终端做只读检查：

```bash
cd /home/unitree/VLA/GR00T-WholeBodyControl
export CYCLONEDDS_HOME="$PWD/.runtime/github-bundle/unitree-sdk"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
.venv_brainco_hand/bin/python -m gear_sonic.scripts.brainco_hand_web --network eth0 --check-seconds 5
```

后续网页调手终端也执行上面两条 export。左右反馈正常后，继续原 SOP 第八步调开合并同步配置；保留确认左右手、轻微动作验证和配置回传 PC 的步骤。

## 5. 对方 PC：替代原 SOP 第五步安装

在 PC 本地终端运行：

```bash
cd "$HOME/Downloads/sonic-brainco-github-bundle-20261111"
bash tools/github_bundle/install.sh pc "$HOME/sonic_v11/GR00T-WholeBodyControl"
```

使用本地同提交的 LeRobot，创建或复用 .venv_teleop 和 .venv_data_collection，不会清空已有环境。适用于原 SOP 的 PICO/XRoboToolkit 路线；不安装与这条路线无关的 CloudXR 或训练组件。

安装与 PC Ubuntu 版本匹配的本地 deb（先查看 VERSION_ID，不要安装另一个版本）：

```bash
. /etc/os-release
printf '%s\n' "$VERSION_ID"
sudo apt-get install "./downloads/xrobotoolkit-${VERSION_ID}.deb"
```

仅准备了 22.04 和 24.04 的 amd64 包；其他系统先停下核实兼容性。按原 SOP 启动 PC Service、填写 Wi-Fi 地址和配置 PICO。APK 已在本包内，无需 GitHub 下载：

```text
downloads/pico.apk
```

用原有 PICO 安装方式侧载；已经安装正确版本则跳过。

## 6. 模型及开始采集

模型仍按 HF Mirror 下载并按上一版命令 scp 到机器人。这四个文件不在 GitHub 包中：

```text
gear_sonic_deploy/policy/sonic_v1_1/model_encoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/model_decoder.onnx
gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

两端 SHA256 一致后不重复下载。保持机器人项目路径不变，继续原 SOP 第八步调手、第九步启动、第十步验收数据。第九步启动强脑服务时使用本说明 4.3 的新服务路径，其余采集参数仍从原 SOP 取实际网卡、IP、dataset-name。

## 7. 验证边界

本机交付验证：全部文件 SHA256 校验通过；16 段命令通过 bash 语法检查；SONIC CMake 配置在外网代理不可达时通过；RealSense 在同样条件下完成 x86_64 实际编译、安装、Python 导入及 context 创建；PC 依赖 dry-run 解析成功，LeRobot 确认为本地来源。测试未安装或更改本机系统驱动。

SHA256 只验证传输完整性；安装脚本通过不代表真机已验收。最终需要依次确认：相机出图、SONIC 在现有 TensorRT 下可编译加载模型、两只手有新鲜反馈、PICO 正常、调手配置回传、数据文件字段正确。接收方的 ARM64 编译、USB/DDS 与身体运动必须在现场验证。
