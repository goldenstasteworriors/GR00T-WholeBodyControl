# ThinkPad 有线集中记录方案（本机接收时间，不对时）

日期：2026-09-07。2026-09-08 更新：采集程序已实现并完成离线验证，尚未进行硬件联采。实际行为与启动步骤见 [使用说明](thinkpad_receive_collection_usage.md)。以下保留原始设计，使用说明优先；当前直接读 USB 相机，无需本机相机 ZMQ 服务。
会话标识：20260907-thinkpad-wired-receive-time；本会话第 1 次保存。

## 决策与适用边界

相机通过 USB 直接连接 ThinkPad，机器人通过有线局域网连接同一台 ThinkPad，由 ThinkPad 统一记录。按用户要求，不配置 NTP/chrony/PTP，不进行跨机时钟映射，不估计或补偿固定延迟，不依据机器人绝对时间对齐。

这个方案可以用于按接收情况组织采集数据。统一时间轴消除了比较两台电脑时钟的需求，但没有消除相机曝光、USB、机器人消息传输和软件排队的延迟差。因此称为“接收时间对齐”，不宣称测量时刻同步或毫秒级精度，也不能在实测前保证满足某项训练精度要求。

## 部署与隔离

```text
RealSense ── USB ── ThinkPad 相机采集 ──┐
                                      ├─ ThinkPad exporter ── 数据集 + 时间元数据
机器人 ── 有线局域网 ── 状态接收 ───────┘

机器人保留已有 WBC / PICO / 导航 / Inspire / ref-gripper 控制服务
ThinkPad 只承担相机读取、状态订阅、记录和可选预览
```

相机默认保持原安装位置。如果移动相机安装位置，外参与视角变化是另一项需要处理的工作。

本次使用独立 worktree，不切换主项目或已有 worktree 的分支，不复制未提交改动，不启动相机或控制程序，不占用运行中的端口，不修改共享环境或网络配置，不操作机器人服务器。后续实现也必须维持这一边界。

已确认的 ThinkPad 仓库位置：

```text
/home/data-collection/GR00T-WholeBodyControl
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
```

新方案 worktree：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-receive-time-20260907
```

新分支：

```text
docs/thinkpad-wired-receive-time-20260907
```

基于已有采集 worktree 的已提交版本 1779cc5435718fc701bdacde387ead6326feb1bb。该版本不代表机器人当前部署版本，不应从这里整体覆盖机器人或正在运行的 ThinkPad 项目。

## 统一时间轴的具体规则

全部匹配使用 ThinkPad 本次开机的 time.monotonic_ns()。启动记录器时保存一次墙上时间和单调时间锚点用于日志显示；墙上时间和机器人时间不参与匹配。不同电脑的 monotonic 值不可比较。

1. 相机线程拿到一个新的有效 RealSense 帧时，立即记录 host_receive_monotonic_ns；在图像复制、JPEG 编码和写盘之前打点。保留相机帧号。这个时间是主机取得帧的时间，不称为曝光时间。
2. 如果继续使用本机 composed_camera 服务，把该单调时间和帧号传给 exporter；exporter 不得用本机 ZMQ 接收时间覆盖它。这样本机编码和传输排队不会被误当成新拍摄时刻。
3. 机器人状态在 ThinkPad ROS 接收回调开始时记录 host_receive_monotonic_ns，再反序列化和处理。不要等 exporter 轮询时才打点。这个时间包含网络和 ROS 调度延迟，不称为底层测量时间。
4. 身体、手部反馈、动作若来自不同消息，各自记录本机接收时间。合并消息中已有的旧手部数据不能因为身体消息更新而宣称重新测量。缺少独立反馈序号时明确标记来源新鲜度未知。
5. 原始源时间戳如存在，只保留作追溯，不参与本方案对齐，也不将源 monotonic 与 ThinkPad monotonic 相减。

以上是待实现要求，不是当前代码已经具备的能力。

## 按记录情况拼接数据

保留原 50 Hz 数据集快照语义作为初始兼容方案，不改变机器人控制频率，不进行历史插值或延迟补偿。

- 每个记录 tick 在同一临界区取得截至该 tick 已接收的最新图像、状态和动作引用。
- 各路接收独立于编码和写盘；接收缓存使用有界队列，明确记录溢出丢弃数量。
- 数据行保存 tick 实际时间、选中的图像/状态序号、各自本机接收时间、年龄，以及两者接收时间差。
- 30 FPS 相机配 50 Hz 数据集允许重复图像，但重复行必须沿用原帧号与原接收时间，不能刷新图像时间伪装成新帧。50 Hz 数据行不代表 50 张独立图像。
- 尚无任一路有效数据时不开始 episode。接收时间不能证明网络中没有积压，应另外检查序号推进和消息间隔。
- 可先以 100 ms 作为图像或状态在笔记本缓存中的陈旧告警阈值（初始配置建议，非同步精度保证）。超阈值或队列溢出时结束当前有效片段并标记原因，不能无限重复最后一张图像或状态。
- 记录循环明显超时同样记录为异常；不能通过连续帧号掩盖一段真实时间缺口。
- 断流和相机重连后生成新的流标识；重新满足数据就绪条件再开始新片段。停止记录不向机器人发送动作、复位或服务停止命令。

保持原 observation.state、observation.eef_state、action.wbc 等字段顺序和手部语义。动作指令与实际关节反馈分别存储，不把指令当作测量值。PICO/手部/导航等现有字段按其实际来源保留，未知值不得伪装成真实观测。

标准 LeRobot timestamp 若仍按 frame_index / fps 生成，仅表示导出时间轴。另用逐行 JSONL 关联文件保存真实接收时间，键为 episode_index + frame_index；文件与数据集写到本 worktree 下同一数据目录。配置元数据记录 alignment_mode=host_receive_latest、clock=thinkpad_monotonic、delay_compensation=false。这些字段均为本方案拟新增，不是现有 CLI 参数。

## 现有代码依据与后续实现位置

以下相对路径均相对于本 worktree：

- [run_decoupled_vla_data_exporter.py](../../../gear_sonic/scripts/run_decoupled_vla_data_exporter.py)：_poll_state_ros 当前保留源 timestamps，仅缺失时补 time.time()；_add_data_frame 当前使用 latest_proprio_msg 和最新图像。需要新增独立接收时间与快照元数据。
- [ros_utils.py](../../../decoupled_wbc/control/utils/ros_utils.py)：ROSMsgSubscriber._callback 当前仅保存消息，get_msg 再解码。需要提供本机回调时间，避免改坏其他订阅者。
- [realsense.py](../../../gear_sonic/camera/drivers/realsense.py)：当前读帧后生成 time.time()。需要新增本机 monotonic 与真实帧号。
- [composed_camera.py](../../../gear_sonic/camera/composed_camera.py) 和 [sensor_server.py](../../../gear_sonic/camera/sensor_server.py)：需要端到端传递时间和帧号，缓存返回不能生成新时间。
- [exporter.py](../../../gear_sonic/data/exporter.py)：维护固定 FPS 与视频行数关系，另存实际时间元数据。
- [launch_decoupled_vla_collection.py](../../../gear_sonic/scripts/launch_decoupled_vla_collection.py)：现有综合 launcher 会启动控制与 teleop 等进程，不能直接在笔记本重跑整套作为“仅记录”。后续需要明确的只记录入口，不重复启动机器人控制。
- [decoupled_vla_collection.md](decoupled_vla_collection.md)：现有部署和环境参考。后续若运行代码，先核对项目环境，不在 base 安装依赖。

当前 decoupled exporter 通过 ROS 订阅机器人状态，不是仅加一个 robot-ip 参数就已完成跨机订阅。实施前需要只读核实有线 IP、ROS domain、DDS 接口/发现、实际机器人消息字段和版本，避免笔记本收到另一套控制程序的状态。只记录进程不得发布控制指令。现有录制按键/服务调用是否会改变机器人控制状态，也必须在实现时审查并隔离。

## 验证与交付边界

本次只创建方案分支、写文档并进行文档与 Git 检查；未安装依赖，未启动录制、相机、DDS 订阅或机器人程序。

后续实现的验证应依次检查：

1. 不连接硬件，用已有数据或内存模拟输入验证重复帧时间不刷新、时间年龄计算、队列溢出、断流和 episode 边界。临时测试脚本测试后删除。
2. 在可用设备与端口明确后验证只读订阅来源、帧号推进、采样率、图像尺寸和关节顺序。现在两端都有程序运行，不执行此步骤。
3. 短采集检查视频帧数与数据行数、50/30 Hz 重复图像统计、真实 tick 间隔、源接收时间差分布、告警和磁盘写入稳定性。
4. 如后续需要评估训练适用性，再做视觉运动与关节轨迹的残余错位检查。这个检查不引入对时或自动延迟补偿。若效果不满足需求，报告限制，不宣称时间对齐已获物理精度保证。

尚未验证：ThinkPad 到机器人的实际有线链路、跨机 ROS 可达性、相机可用性、机器人当前消息版本、真实两路延迟和训练效果。本方案无需预设这些已通过。
