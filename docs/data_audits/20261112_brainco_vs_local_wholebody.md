# 强脑新数据与本地原全身数据的兼容性检查

日期：2026-11-12。新数据为pick_box_brainco_20261112_v1。以用户此前提到的7_30_1全身流程为主线，只读比较本地真实文件；没有修改任何新旧数据或采集代码。

## 对照范围

主要旧数据位于本项目outputs下：

- [7_30_1_episode_000003](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/7_30_1_episode_000003/meta/info.json)：SONIC release，43维全身格式。
- [7_31_3_selected](/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/7_31_3_selected/meta/info.json)：SONIC release，43维全身格式。
- [Grab the pipette（副本）](</home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/Grab the pipette (副本)/meta/info.json>)：SONIC low_latency，43维全身格式。

辅助检查outputs/onboard/8_11_shake_beaker及outputs/pick_up_pipette。它们的script_config表明是解耦控制数据，采用41维Inspire格式；不把它们当作原全身SONIC数据的替代基准。

上述outputs实际位于本机项目根目录，分支worktree不包含这些原始数据。新数据位于：

```text
/home/ykj/Downloads/dataset/pick_box_brainco_20261112_v1
```

比较meta/info.json、meta/modality.json以及每组前最多3个实际Parquet文件。另外从3组旧全身数据及新数据各抽100行复算FK。不是对全部历史episode的视频和数值质量逐帧验收，也没有对不同时刻的实验要求数值相等。

## 结论

采集的核心数据内容符合原全身SONIC链路：没有漏掉原来非手部的字段，身体状态/目标关节角、腕部FK、64维motion token及人体参考等语义保留。但不能说“除手外文件布局完全没变”，原43维全身state/action的右臂切片位置改变了；旧数据读取代码若写死切片，需要更新。

## 字段对照

旧全身info.json有32个feature，新数据有35个。原有字段全部仍存在；其中28个不涉及手部的feature定义（key、shape、dtype、names及视频info）完全相同。只有以下4个既有feature定义改变：observation.state、action.wbc、teleop.left_hand_joints、teleop.right_hand_joints。新增3个feature均为强脑手诊断量。

| 项目 | 原全身SONIC数据 | 新强脑数据 | 判断 |
| --- | --- | --- | --- |
| 容器 | LeRobot v2.1，Parquet＋H.264视频 | 相同 | 一致 |
| 记录频率 | 标称50Hz | 标称50Hz | 一致；不保证真实曝光/控制采样严格50Hz |
| 主相机 | observation.images.ego_view，480×640×3 | 相同 | 格式一致；不代表相机外参、画面内容或延迟相同 |
| 身体 | 29个G1关节，角度状态和目标角 | 相同29个关节 | 移除手部名字后名称与顺序完全相同 |
| EEF | 双腕14维，位置＋wxyz四元数 | 相同 | FK实测确认一致 |
| motion token | 64维float64 | 相同 | 定义一致；模型版本兼容另见下文 |
| SMPL | joints 72维、pose 63维、body quat 4维 | 相同 | 定义一致 |
| 其他全身字段 | 根姿态、重力、C++旋转偏移、初始基座姿态、delta heading、腕关节、模式及planner/VR字段 | 全部保留 | 定义一致 |
| 手部 | 两侧各7维legacy手模型接口 | 两侧各6个强脑电机，真机归一化反馈和目标 | 明确的手部差异，不可直接混用单位和索引 |
| 手部诊断 | 无这3个feature | 速度12维、电流12维、反馈接收年龄2维 | 新增强脑专属数据 |

## 身体数组位置也受到手部布局影响

以下为Python从0开始、右端不含的切片；observation.state与action.wbc都如此：

| 分段 | 原43维全身 | 新41维强脑 |
| --- | --- | --- |
| 双腿＋腰 | 0:15 | 0:15 |
| 左臂 | 15:22 | 15:22 |
| 左手 | 22:29 | 29:35 |
| 右臂 | 29:36 | 22:29 |
| 右手 | 36:43 | 35:41 |

旧全身布局是“下肢/腰＋左臂＋左手＋右臂＋右手”，新布局是“身体29＋左手6＋右手6”。因此不能直接把旧43维的前29维当作完整身体，也不能在新数据上继续用旧29:36切片读取右臂。

两者各自的modality.json与names已经描述正确位置。移除手部后的旧身体索引为0:22与29:36拼接，新身体索引为0:29。对现有41维解耦Inspire数据，身体29＋左手6＋右手6的布局则与新数据相同，但控制链路不同。

## 数值语义验证

原全身导出器先把C++的body_q及last_action按RobotModel插入完整43维关节配置，新导出器直接拼接身体29维和两侧6电机。C++身体发布仍为MuJoCo关节顺序，状态包含默认角度偏移，action.wbc为经过action_scale和默认角度转换的目标关节角；没有换成速度、关节增量或网络归一化动作。

把上述4组真实数据各100行按各自关节名取出身体29维，再使用同一个项目G1模型重算两侧wrist_yaw_link位姿，与保存的observation.eef_state比较：

| 数据 | 最大位置绝对误差（米） | 最大旋转误差（弧度） |
| --- | --- | --- |
| 7_30_1 | 0 | 1.36e-16 |
| 7_31_3 | 0 | 1.27e-16 |
| 旧low_latency | 0 | 1.24e-16 |
| 新BrainCo | 1.39e-17 | 1.72e-16 |

这验证了抽查行中身体顺序、单位及双腕FK定义的一致性；不表示不同任务的状态数值应相同。保存的是腕部位姿，不是强脑指尖位姿。

同时比较强脑适配前的导出器源码（3885e99的父提交）和当前强脑分支：_add_cpp_state_features、_compute_target_body_orientation、_add_images_to_frame_data、_poll_state_zmq、_handle_pose_message的AST相同，支持根姿态/重力、人体朝向、图像取样和PICO消息解码未因换手而改动的判断。

## 除手部外应明确保留的区别

1. **模型配置版本**：旧数据元数据记录policy/release或policy/low_latency，新数据记录用户要求的policy/sonic_v1_1。planner文件路径也不同。当前没有对新旧原始模型做SHA256比对，因此不凭目录名断言权重是否相同；也不能因为token都是64维就认定其跨checkpoint语义相同。若训练目标使用motion_token，需要管理模型版本或另行验证。
2. **部署位置**：原流程以PC推理，新流程机器人板载推理。这没有删除数据feature，但端到端延迟不能据此视作相同。
3. **本条数据质量**：新数据约1～3帧的图像滞后、10行SMPL零回退、任务目标经常不在画面内，见[本条质量报告](20261112_pick_box_brainco_v1.md)。字段一致不等于这些质量项已经验收通过。
4. **源时间信息仍缺失**：两套字段都只有固定频率生成的timestamp，没有在Parquet保存每帧相机/状态/动作原始时间戳。现有字段结构不支持严格证明曝光与动作采样同刻。

综上：可以确认采集内容仍是用户原全身SONIC数据体系；不能把旧43维训练读取代码未经调整直接用于新41维数据。使用names/modality映射、明确手部单位并管理模型版本后，身体与其他既有feature可以按相同语义处理。

源码位置：

- [features_sonic_vla.py](../../gear_sonic/data/features_sonic_vla.py)：206行旧全身定义，373行41维基础布局，470行强脑扩展。
- [run_data_exporter.py](../../gear_sonic/scripts/run_data_exporter.py)：928行状态拼接，931行动作拼接，939行起FK，994行起C++状态扩展。
- [zmq_output_handler.hpp](../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp)：315行身体状态、339行目标动作的顺序/尺度转换。

机器可读对照结果和旧新视频抽帧概览保存在本机项目.runtime/dataset_audit/pick_box_brainco_20261112_v1目录，未上传原始实验数据。
