# 分机采集启动流程（2026-09-09）

接线：PICO、左 Inspire 接机器人；RealSense、REF USB 接 ThinkPad；两机有线互通。
机器人运行原 PC Service、PICO 解析、身体/导航及左手控制。ThinkPad 运行 REF 和数据记录。

## 机器人端

新增入口所在独立 worktree 如下；不切换原目录分支，不复制或覆盖原手部配置。
入口通过 robot-repo 读取原仓库的 launcher builders、配置和模型路径。
原 ref-gripper-bridge 的混合手部映射被保留，但不启动机器人 REF 进程。
不使用 left-hand-only；原右臂和 PICO 行为保留。

终端一，下面两行依次执行（会启动 PICO PC Service）：

```bash
cd /home/unitree/data_collection/GR00T-WholeBodyControl
bash scripts/onboard_pico_services.sh start
```

终端二，先执行 cd，再执行第二行。默认只预览三个进程的完整命令，不控制硬件：

```bash
cd /home/unitree/data_collection/GR00T-WholeBodyControl-worktrees/thinkpad-split-20260909
bash scripts/run_split_collection_onboard.sh --task-prompt "move the beaker from left" --dataset-name 8_25_stir_2 --hand-task grab_beaker --pico-navigation-range 0.5
```

真正采集时，在第二行末尾加 --start（保持同一行）。
**--start 会启动身体、导航和左手控制；本次测试没有执行真实 --start。**
原程序已经运行时入口拒绝重复启动，不会自动杀进程。
进入新会话后仍按原控制程序提示操作，不应仅凭 tmux 建立就认为控制已就绪。

```bash
tmux -S /home/unitree/data_collection/GR00T-WholeBodyControl/.runtime/split_collection/tmux.sock attach -t decoupled_wbc_split_onboard
```

不再运行原综合 launch_decoupled_vla_collection_onboard.py，也不在机器人运行 composed_camera。

## ThinkPad 端

终端一，下面两行依次执行。第二行实际控制 USB REF，需要机器人 PICO 解析主题已发布：

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-receive-time-20260907
bash scripts/run_thinkpad_ref_bridge.sh --hand-task grab_beaker
```

终端二，下面两行依次执行。第二行保持整行，数据集名称必须尚不存在：

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-receive-time-20260907
PYTHONDONTWRITEBYTECODE=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=0 FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/gear_sonic/config/data_collection/thinkpad_ref_fastdds.xml" /home/data-collection/miniconda3/envs/decoupled_vla_collection/bin/python -m gear_sonic.scripts.run_thinkpad_receive_collection --task-prompt "move the beaker from left" --dataset-name 8_25_stir_2_thinkpad_01 --root-output-dir outputs/thinkpad
```

两台机器的 ROS_DOMAIN_ID 必须一致；沿用实际机器人值。当前 Fast DDS profile
指定 ThinkPad 有线 IP 192.168.123.222，IP 改变时需相应修改。
记录器直接打开本机 RealSense，不额外运行 composed_camera。

在记录器终端输入 c 后 Enter 开始/结束一个 episode；x 后 Enter 标记丢弃；
q 后 Enter 保存并退出。退出记录器不会停止机器人控制。

## 安全预检与已知限制

ThinkPad 上述 REF 命令追加 --preflight，仅检查依赖，不打开 USB。
记录器命令追加 --preflight，不打开相机、不初始化 ROS。
REF 的 --dry-run 用模拟 USB，不可把它视为实际夹爪运动验收。

按 ThinkPad 接收单调时间关联相机与状态，没有跨机对时，也没有消除网络、
相机曝光及缓冲延迟。不能声称硬件级同步。
REF 目标/ACK 的同机时间日志单独保存在 .runtime/ref_bridge，
尚未合入 LeRobot；原右侧手部字段不是 REF 实测开口。
更换机器人抓握配置后，需要同步核对 ThinkPad 手部与夹爪配置快照。

## 实现与验证

- 新入口复用原机器人 5c22bfb launcher 的控制、PICO、Inspire argv builders；
  与用户原启动参数逐项比对。
- 使用真实 tmux 隔离 socket，将 Python 执行替换为只打印参数的 shell 函数；
  验证三个窗口命令分发、环境前缀及重复会话拒绝，不加载硬件控制代码。
- 机器人仅运行相同参数的默认命令预览。ThinkPad 运行两个真实入口的 preflight。
- 未验证真实 PICO/身体/左手/REF 联控、真实相机与状态联采、同步误差及长时稳定性。
- 临时测试脚本、隔离 tmux 及自产输出测试后删除。

代码：
[机器人入口](../../../gear_sonic/scripts/launch_split_collection_onboard.py)、
[启动包装](../../../scripts/run_split_collection_onboard.sh)、
[记录说明](thinkpad_receive_collection_usage.md)、
[REF 说明](thinkpad_ref_bridge.md)。
