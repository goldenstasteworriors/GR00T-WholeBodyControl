下面是当前 09_07_load_doll_1 的完整采集流程。当前配置已经创建，今天不用再次执行 init。

  ## 一、物理连接和上电

  1. 急停放在手边，清空机器人周围人员和障碍物。
  2. ThinkPad USB 网卡连接机器人交换机。
  3. G1 和物理右手连接交换机。
  4. 右手当前地址为 192.168.123.211。
  5. ThinkPad 和 PICO 连接同一个 Wi-Fi。
  6. 相机服务器运行在：

  http://222.29.157.228:8765/stream.mjpg

  7. 给交换机、右手和机器人上电，等待 G1 完全启动。

  ## 二、打开终端并恢复网络

  cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect

  nmcli connection up '有线连接 1'

  ./collectctl.sh config

  当前配置应该显示：

  dataset              09_07_load_doll_1
  task prompt          load the doll
  collection mode      resume
  hand mode            right-only
  G1                   192.168.123.164
  left/right hand      192.168.123.210 / 192.168.123.211:6000
  hand pose config     /home/data-collection/GR00T-WholeBodyControl/tools/0902collect/inspire_hand_pose.json
  camera               http://222.29.157.228:8765/stream.mjpg
  PICO PC Service IP   222.29.157.170

  启动或检查 PC Service：

  ./collectctl.sh pc-service

  如果需要强制重启：

  ./collectctl.sh pc-service --restart

  预期：

  [PASS] RoboticsService ...
  [PASS] set PICO PC Service to 222.29.157.170, recalibrate, then press Send

  ## 三、PICO 操作

  在 PICO/XRoboToolkit 中：

  1. PC Service IP 填写 222.29.157.170，但以后以 config 实际显示为准。
  2. 选择身体跟踪设备。
  3. 重新校正。
  4. 点击 Send。
  5. 保持身体数据发送。

  ## 四、启动全部采集服务

  执行：

  ./collectctl.sh launch

 结束所有：./collectctl.sh stop --off-preview-confirmed
  launch 会自动检查：

  - G1 网络、SSH和模型
  - 右手 .211:6000
  - `inspire_hand_pose.json` 中的开合姿态和固定拇指横向目标
  - HTTP 相机及帧率
  - PICO 数据连续性
  - RoboticsService
  - 数据集名称未被占用
  - 没有旧 tmux 或重复进程

  必须看到：

  [PASS] PICO body data is live
  [PASS] all launch gates passed

  随后会出现四个 pane 和额外窗口。

  ## 五、tmux 内启动机器人

  1. 点击 pane 0。
  2. 看到：

  Proceed with REAL robot deployment? [Y/n]

  3. 确认环境安全后按 Enter。
  4. 等待：

  Init Done

  5. 在控制器上短按 A+B+X+Y，然后四键完全松开。

  预期：

  Calibration completed
  StreamMode switch: OFF -> PLANNER

  6. 按一次 A+X，然后完全松开。

  预期：

  StreamMode switch: PLANNER -> POSE

  ## 六、录制前健康检查

  按：

  Ctrl+b
  松开
  再按 d

  然后执行：

  ./collectctl.sh health

  注意：必须在完成 A+B+X+Y 和 A+X 后执行。

  只有最终出现：

  [PASS] collection is healthy; recording may start or continue

  才可以录制。

  重新进入：

  ./collectctl.sh attach

  ## 七、录制控制

  左控制器 Grip + 右控制器 A

  第一次按：开始录制。

  预期：

  [Recording] 开始记录 episode 0

  完全松开两个按键后，再按一次相同组合：停止并保存。

  必须等待：

  [Recording] 结束记录 episode 0
  [Recording] Finished saving episode

  下一次录制自动成为 episode 1、episode 2，以此类推。

  丢弃当前正在录制的片段：

  左控制器 Grip + 右控制器 B

  ## 八、每段录制后的检查

  停止录制并看到 Finished saving episode 后，按 Ctrl+b d 脱离，再执行：

  ./collectctl.sh status
  ./collectctl.sh verify-latest

  预期：

  recording=false
  [PASS] resumable dataset
  [PASS] latest episode=N

  重新进入继续录制：

  ./collectctl.sh attach
  ./collectctl.sh attach
  ## 九、正常结束全部采集

  1. 如果正在录制，先按左 Grip + 右 A。
  2. 等待 Finished saving episode。
  3. 按下并完全松开 A+B+X+Y，停止机器人策略。
  4. 按 Ctrl+b d 脱离。
  5. 执行：

  ./collectctl.sh stop
  ./collectctl.sh verify-latest

  如果策略从未启动、始终处于 OFF 预览：

  ./collectctl.sh stop --off-preview-confirmed

  检查 tmux 是否完全关闭：

  tmux ls
./collectctl.sh stop --off-preview-confirmed
  正常显示：

  no server running on /tmp/tmux-.../default

  ## 十、下次上电后续采同一数据集

  cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect

  ./collectctl.sh init \
    --dataset 09_07_load_doll_1 \
    --task 'load the doll' \
    --hand-mode right-only \
    --camera-url 'http://222.29.157.228:8765/stream.mjpg' \
    --resume

  nmcli connection up '有线连接 1'

  ./collectctl.sh config
  ./collectctl.sh pc-service --restart

  随后 PICO 重新校正并 Send，然后：

  ./collectctl.sh launch

  ## 十一、异常、掉线或中途断电

  先按物理急停，再执行：

  cd /home/data-collection/GR00T-WholeBodyControl/tools/0902collect

  ./collectctl.sh emergency-stop \
    --physical-estop-pressed \
    --delete-active

  不要直接执行：

  tmux kill-session

  否则可能破坏正在保存的 Parquet 或视频。

  完整清单保存在 GR00T-WholeBodyControl/tools/0902collect/OPERATOR_CHECKLIST.md:1。
