# 强脑手：按宇树官方显式串口版本独立测试

本包对应用户提供的官方网页命令：`brainco_hand --id ... --serial ...` 与 `example_brainco_hand left/right`。

来源是宇树官方仓库提交 `9eaadea8126825edadcc1dc30a1134b8217f7fed`，其 README 的 Setup/Test 与所贴命令一致：

https://github.com/unitreerobotics/brainco_hand_service/tree/9eaadea8126825edadcc1dc30a1134b8217f7fed

其子模块为 BrainCoTech/stark-serialport-example 提交 `4363f61147d741c208cf2470877ed01fa71fca44`，download-lib.sh 指定 SDK v0.8.1。本包已从脚本指定的厂商地址下载 ARM64 和 x86_64 SDK，无需再次访问 GitHub 或运行 download-lib.sh。服务、官方示例和 SDK 源码不修改；build.sh 仅解压匹配架构库并指定项目内 Unitree SDK 的安装位置。

此包只做官方基线测试，不替换 SONIC 项目、原服务和开合配置。旧服务未填充电流字段，且循环频率与现有版本不同；不能直接用来验收我们完整采集数据链路。

## 1. 传到对方 PC，再传机器人

把压缩包、对应 .sha256 文件和本说明发给对方，放在 PC 的 Downloads。对方 PC 执行，每行单独执行，前一条成功后再继续：

```bash
cd "$HOME/Downloads"
sha256sum -c brainco-official-explicit-test-20261111.tar.gz.sha256
scp brainco-official-explicit-test-20261111.tar.gz brainco-official-explicit-test-20261111.tar.gz.sha256 unitree@192.168.123.164:/home/unitree/VLA/
ssh unitree@192.168.123.164
```

以下在机器人执行。首次解压才执行 tar，不覆盖已有同名测试目录：

```bash
cd /home/unitree/VLA
sha256sum -c brainco-official-explicit-test-20261111.tar.gz.sha256
tar -xzf brainco-official-explicit-test-20261111.tar.gz
cd brainco-official-explicit-test-20261111
bash build.sh
```

需要 APT 网络，GitHub 和 PyPI 均不需要。不会升级 CUDA、驱动或 GLIBC；编译不会连接实体手。成功标准：ldd 没有 not found；帮助中列出 --serial、--id 和 --network_interface。

## 2. 准备实际串口

先关闭正在运行的其他手部服务/调手程序，避免重复占用。同一时刻只运行一个版本；不要同时启动 SONIC 采集。

官方 README 中的 /dev/ttyUSB0 是示例，必须替换成真实存在、已确认属于手的设备路径。在机器人执行：

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/ttyHAND* /dev/ttyUN*
```

若没有任何对应节点，可以完成前面的编译和 --help 检查，但不能把示例设备路径当作真实端口；先解决设备枚举。禁止创建假串口文件、随意链接板载 UART 或遍历机器人所有串口发送探测命令。

## 3. 官方服务测试

机器人终端 A：在允许手部张开/闭合、手中没有物品时操作。先检查示例端口是否存在，只有 test 成功后才运行下一条；如果真实路径不同，下面两条都同步替换。

```bash
cd /home/unitree/VLA/brainco-official-explicit-test-20261111/brainco_hand_service/build
test -c /dev/ttyUSB0
sudo ./brainco_hand --id 126 --serial /dev/ttyUSB0
```

这是官方 README 的左手启动命令，默认 DDS 网卡自动选择。若需要明确网卡，该版本也支持 --network_interface；网卡变更需要保证服务和测试端 DDS 互通。

预期读取到设备 SKU、序列号、固件版本，并保持运行。若提示端口打开失败/设备连接失败，保留完整日志，不启动动作示例。

## 4. 官方动作示例

前一步设备连接成功且服务保持运行后，在另一个机器人 SSH 终端 B 执行：

```bash
cd /home/unitree/VLA/brainco-official-explicit-test-20261111/brainco_hand_service/build
./example_brainco_hand left
```

示例会循环发送张开/握拳目标，观察是否是左手动作。用 Ctrl+C 停止示例，再在终端 A Ctrl+C 停止服务。停止示例不表示手自动张开，操作前保持手部周围无障碍物。

右手分别将服务 ID 改为 127、串口改为确认的右手端口，示例参数改为 right；不要仅凭 USB 编号大小判断左右手。

## 5. 测试结果发什么

发 build.sh 末尾帮助输出、实际串口路径、服务连接日志、示例是否带动正确一侧。若无串口，先发设备枚举结果；官方程序不会创建 USB 设备节点。

本包已检查原始源码与固定提交一致、SDK 架构和所需符号、包内 SHA256 与脚本语法；ARM64 真机编译、实体手连接及动作需在现场验证。
