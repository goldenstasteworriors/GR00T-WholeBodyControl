# NaVILA 导航微调采集

本入口用于 ThinkPad＋PICO 摇杆＋有线 G1。低层仍为当前 Balance/Walk ONNX，速度上限不变（平移 0.25 m/s、转弯 0.5 rad/s）。不会启动 NaVILA 推理；由人示范路线。数据保存在 ThinkPad 的本项目 outputs 下。

## 官方格式与采集内容

官方训练代码：
- https://github.com/AnjieCheng/NaVILA/blob/main/llava/data/dataset.py （LazyVLNCEDataset）
- https://github.com/AnjieCheng/NaVILA/blob/main/llava/mm_utils.py （vlnce_frame_sampling）
- https://github.com/AnjieCheng/NaVILA/blob/main/llava/data/datasets_mixture.py

一个训练样本包含整条路线的指令 `q`、截至决策时刻的头部图像序列 `frames`、下一段动作文本 `a` 和 `video_id`。官方 8 帧版本从历史采样并保留当前帧；初期不足 8 帧由官方加载器补空白帧。图像不能包含决策之后的画面。

实际采集保存：

| 信息 | 保存位置与用途 |
|---|---|
| 整条路线的自然语言指令 | meta/tasks.jsonl；描述目标、地标、转向、停止位置 |
| 头部 RGB | observation.images.ego_view 视频；NaVILA 模型输入 |
| 双腕 RGB | 同步组保留；原版 NaVILA 导出不使用腕图 |
| 请求/策略速度 | navigation.requested_velocity / policy_velocity；x前、y左、yaw逆时针，m/s、rad/s |
| 控制状态 | navigation.control_mode；0阻尼、1初始化、2就绪、3运行 |
| 相机/状态时间 | navigation.camera_time / state_time；分别为机器人相机主机时钟、ThinkPad Unix 时钟 |
| 实际关节角/关节目标 | observation.state / action.wbc；含实际 Dex1 角度 |
| IMU 基座方向 | observation.root_orientation，wxyz；用于核对转向 |
| 固定双臂配置 | meta/info.json 的 script_config.navigation_collection.arm_profile |
| 人工审核的动作与成功标记 | 独立标签 JSON，再导出 annotations.json |

策略速度是低层命令输入，不能当作实测里程计。命令积分只作为标注提示；前进距离应使用实测/场景标记核对，转角可结合 IMU 和画面核对。状态与图像使用“每个新相机组配最近收到的状态”，没有硬件同步或跨主机时钟标定保证；三相机仍为软件时间戳分组、最大组内差 30 ms。

原版动作空间为前进、左转、右转、成功停止。示范应分段做纯前进和原地转向；横移、后退、混合转弯可以录下，但标注草稿会标为 unsupported，需要排除或重新示范，不能直接改名当成纯前进。

## 一键采集

每个代码块逐行执行；命令自身均为一行。

1. 进入采集目录

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
```

2. 开启 PC Service

```bash
bash scripts/thinkpad_pico_services.sh start
```

3. 连接 PICO 并开启 Send

XRoboToolkit 填 ThinkPad 当前 Wi-Fi IPv4。可用下面命令查看（Wi-Fi 地址会变化）；PC Service 已运行时采集入口会复用。

```bash
ip -4 address show wlp0s20f3
```

4. 启动导航采集

```bash
bash scripts/collect_navila_on_thinkpad.sh "Walk through the doorway, turn left, and stop in front of the table." hallway_table
```

仅修改引号中的完整路线指令和最后的项目名。脚本自动调用机器人夹爪/相机服务、PC Service、连接预检、四窗格采集和实时预览，使用现有 default 夹爪配置。目录名自动为 `navila_hallway_table_时间戳`。未连接机器人有线网卡时立即报错，SSH 连接等待最多 8 秒，不会一直无提示卡住。

**现象和操作：**启动控制进程后，会通过现有约 2 秒初始化插值把双臂/现有初始化关节带到目标，此时机器人已经可能运动；不能把四键之前视为完全静止。初始化 ready 后，新按 A+B+X+Y 才启用站立策略；左摇杆平移、右摇杆横轴转弯。A+X 不采手臂姿态，PICO 手柄姿态、扳机和侧键不改变手臂/夹爪目标；左侧键仍参与录制组合。固定目标并不意味着实测关节角完全无误差。腰部仍由现有低层控制，不增加新腰部目标。

5. 开始整条路线录制

左侧键+A：开始；再次按：结束并保存。沿用不同开始/结束提示音。左侧键+B 丢弃本条。到达目标后保留一小段静止画面再结束，人工确认成功。

6. 急停或退出

左上 control 按 d：阻尼急停，程序保留；四键重新初始化，ready 后再次四键站立。Ctrl+\：先阻尼再退出会话。急停或退出不等于导航任务成功。不要同时启动另一个机器人控制会话。

## 直接修改双臂角度

```text
gear_sonic/config/data_collection/navila_arm_pose.json
```

`units` 固定为 `degrees`；`joints` 的 14 项直接填写角度数值。配置为用户指定的 Pitch −21.81、左右 Roll ±15.63、Yaw ±0.62、肘 −42.57、腕 Roll ∓19.41、腕 Pitch +26.08、腕 Yaw ±6.10。修改在下一次新启动/初始化加载；建议结束当前会话后再改，避免两次初始化配置不同。不要修改通用机器人默认姿态。

## 审核并导出训练集

以下路径中的时间戳需要替换为实际采集目录。先加载同一采集环境：

```bash
source scripts/thinkpad_collection_env.sh
```

生成待审核分段（输出标签文件必须不存在）：

```bash
python gear_sonic/scripts/export_navila_dataset.py outputs/navila_hallway_table_时间戳 --make-template outputs/hallway_table_labels.json
```

逐段看头部视频，修改起止帧、动作和数量；删除未采用的段，审核后才设置 `reviewed: true`。帧索引从 0 开始，`end_frame` 不包含在内。`forward` 的 amount 单位为米，left/right 为度。`command_integral_hint` 是命令积分提示，导出器不会将其当作标签使用。

```json
[
  {"episode_index": 0, "start_frame": 0, "end_frame": 150, "action": "forward", "amount": 1.0, "reviewed": true},
  {"episode_index": 0, "start_frame": 150, "end_frame": 210, "action": "left", "amount": 30, "reviewed": true},
  {"episode_index": 0, "start_frame": 240, "end_frame": 250, "action": "stop", "success": true, "reviewed": true}
]
```

以上仅为格式示例，帧号和数量必须按实际录像填写。成功 stop 的 start_frame 是已经完成任务且静止的画面；未完成路线、暂停、断连、阻尼均不能标 success。导出只接受运行阶段的样本，不自动把停止录制当成功。

```bash
python gear_sonic/scripts/export_navila_dataset.py outputs/navila_hallway_table_时间戳 --labels outputs/hallway_table_labels.json --output outputs/hallway_table_navila_sft
```

产物为 annotations.json、reviewed_segments.json 和 frames/。前进标签的 amount 以米填写，但导出动作文本会转成 `Move forward 100 cm.`，兼容现有 NaVILA bridge 的厘米解析。用官方数据注册方式设置 `dataset_type="vlnce"`、`data_path` 指向 annotations.json、`image_path` 指向导出目录，并将新数据名加入训练 mixture。沿用官方 sft_8frames.sh；训练环境和权重不属于这个采集脚本的安装范围。按完整路线/场景划分训练验证集，避免同条路线的历史帧同时进入两边。

## 帧率与标注时间

采集入口已使用单连接相机转发：机器人只向 ThinkPad 传一路三相机组，本机预览和录制从 localhost:5558 取流。采集期间使用四窗格自带预览；额外的 NaVILA 独立监视进程应正常退出。仅关闭浏览器页面不会停止后台监视进程取流。

2026-11-03 的只读对照验证：额外监视进程与测试接收端分别直连机器人时约 24 FPS；暂停额外监视的取流后，单路接收恢复到三路约 29.9–30.0 FPS。按正式转发结构同时运行本机预览、三路录像和真实身体 DDS 状态记录，约 20 秒得到三路各 600 帧，Parquet 也是 600 行，源时间戳计算约 29.95–29.98 FPS。未创建执行器指令发布器；测试后已恢复原监视进程。此项仅验收相机/身体记录链路，不代表夹爪或完整遥操已通过。

NaVILA 的 8 帧输入没有原始视频必须为 30 FPS 的要求，但丢帧可能漏掉地标、转向及动作边界。导出按实际保留的帧索引读取图像，动作分段提示使用相机源时间戳积分；不会按视频标称帧率推算位移。若实际只收到 23.6 FPS 却写成 30 FPS，播放速度约为真实速度的 1.27 倍，不能用播放器时间直接计算动作距离或转角。标注应核对源时间戳、实际帧号和实测动作量，不通过重复图像补到 30 FPS。
