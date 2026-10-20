# ThinkPad 无线采集流程

当前只使用此流程。
有线与机载完善已暂停。
以下均在 ThinkPad 操作。
命令按行执行。
反斜杠表示跨行续写。

## 一键启动命令

先完成头显连接与 Send。
网页夹爪控制须先停止。
现场准备好再执行。
本命令会接管真实机器人。
整段复制，按原样换行。
行末 && 表示成功后继续。
反斜杠表示参数跨行续写。

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard &&
source /home/data-collection/miniconda3/etc/profile.d/conda.sh &&
conda activate decoupled_vla_collection &&
source "$CONDA_PREFIX/setup.bash" &&
export PYTHONPATH="$PWD" &&
bash scripts/thinkpad_pico_services.sh start &&
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-preset default \
  --camera-host 192.168.123.164 \
  --record-wrist-cameras \
  --camera-sync-tolerance-ms 30 \
  --camera-max-hold-ms 250 \
  --data-exporter-frequency 30 \
  --root-output-dir /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs \
  --task-prompt "grab the red bottle" \
  --dataset-name "dex1_three_camera_$(date +%Y%m%d_%H%M%S)" \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --no-startup-t-pose \
  --profile-timing \
  --camera-viewer \
  --ros-domain-id 73 \
  --save-profile-logs
```

服务首次启动后需连接头显。
若预检失败，不会启动控制。
头显连接后重跑整段即可。
预检会检查实时 PICO 数据。
修改场景名即可切换角度。
无需启动机载PICO服务。

## 每步预期现象

| 步骤 | 预期现象 | 运动情况 |
|---|---|---|
| 1 接网线 | 机器人网络可达 | 本步骤不发动作 |
| 2 激活环境 | 显示采集环境名 | 本步骤不发动作 |
| 3 启动 PC Service | 显示服务PID或复用 | 本步骤不发动作 |
| 4 查看无线地址 | 显示当前Wi-Fi地址 | 本步骤不发动作 |
| 5 PICO 接无线网 | 头显接入同网 | 本步骤不发动作 |
| 6 打开 XRoboToolkit | WORKING且Send开启 | 本步骤不发动作 |
| 7 检查姿态 | 显示PICO OK | 本步骤不发动作 |
| 8 选择夹爪场景 | 显示场景名及四个角度 | 网页启用控制会动爪 |
| 9 完整预检 | 显示Preflight passed | 本步骤不发动作 |
| 10 确认现场 | 支撑及接管人员就绪 | 本步骤不发动作 |
| 11 正式启动 | 出现四个采集窗格 | 官方退出，手臂爪可能动 |
| 12 查看日志 | 四组件持续运行无报错 | 已在发送关节命令 |
| 13 开启身体策略 | 显示starting policy | Balance策略接管腿部 |
| 14 开启上身遥控 | 显示Teleop policy active | 手臂随手柄移动 |
| 15 选择control窗格 | 焦点切至control | 仍由当前控制运行 |
| 16 按c录制 | 显示Started recording | 不改变站立策略 |
| 17 操作手柄 | 三图与状态持续录制 | 手柄及扳机可触发动作 |
| 18 再按c结束 | 停止提示后保存完成 | 停录不停止机器人控制 |
| 19 检查文件 | 三路视频与Parquet存在 | 本步骤不发动作 |
| 20 结束控制 | 先阻尼，再关闭会话 | 不保持站立姿态 |
| 21 关闭PC Service | 显示服务已停止 | 应先确保控制已退出 |

## 三路实时预览

启动时默认开启预览。
右下窗格显示接收日志。
图像显示在独立窗口。

```text
SONIC Camera Viewer
```

三列依次为头、左腕、右腕。
预览按Q退出，采集仍运行。
只接收一份机器人图像流。
本机转发给预览和录制。
日志显示本机地址属于正常。
不修改图像或原时间戳。
时钟差值不等于网络延迟。

采集内部ROS使用域73。
机器人SDK仍使用域0。
独立ROS调试需选择域73。

## 当前夹爪配置

已保存一组，场景名如下。
存在配置不代表抓取已验收。

```text
default
released_q: [5.4, 5.4]
pressed_q: [2.5, 2.5]
```

左右各用对应扳机。
扳机大于0.5使用按下角度。
角度范围是0至5.4 rad。
0为闭合，5.4为打开。
上述数值不是开口毫米数。

## 站立控制与切换

只读查询返回官方模式 ai。
此时本项目控制尚未启动。
模式名不代表具体站立子状态。
正式启动时调用 ReleaseMode。
之后直接发送低层关节目标。
没有继续沿用官方站立服务。

策略未开启时不使用RL输出。
腿部目标取当前反馈位置。
这不等于有主动平衡能力。
手臂启动目标以约2秒插值。
名义肘角为左右负20度。
其余肩腕名义目标为0。
遥控暂停后改为保持反馈姿态。
默认不经过 T-pose。
夹爪已可能开始位置控制。

四键开启本项目身体策略。
零速附近使用 Balance 模型。
速度命令范数达到0.05后：
切换使用 Walk 模型。
摇杆归零会回到 Balance。
A+X 单独切换上身遥操作。
关闭策略不会恢复官方服务。
当前未实测模式交接与站立。
已删除1秒通信超时回退。
已删除夹爪250ms超时拒绝。
启动心跳已撤掉。
保留原初姿和2秒插值。

## 阻尼急停与退出

新启动的采集会话才生效。
选中左上control窗格。
切至英文输入，直接按此键。
不按Shift，直接按小写d。

```text
d
```

先锁住身体及夹爪目标。
身体位置增益归零。
速度目标及前馈力矩归零。
身体阻尼增益使用8。
双Dex1释放位置控制。
阻尼命令持续下发约0.5秒。
再请求官方FSM阻尼状态1。
读回确认后关闭整个会话。
确认结果保存至项目文件。
阻尼不会维持站立姿态。
不新增通信看门狗。

采集界面任意窗格也可按：

```text
Ctrl+\
```

此键给控制进程发停止信号。
不依赖Python键盘监听线程。
关闭最后界面也会阻尼退出。
最后界面detach也会退出。
异常退出同样先执行阻尼。
SIGKILL及断电无法执行清理。

独立终端可请求阻尼退出。

```bash
kill -USR1 "$(tmux show-environment -t decoupled_vla_collection DECOUPLED_WBC_CONTROL_PID | cut -d= -f2)"
```

停止确认记录：

```text
.runtime/collection_damping_stop.json
```

本版隔离验证已通过。
已测试急停键与两种退出。
已验证锁住后不再发位置目标。
真实急停已测试一次。
结果见末尾实际急停验收。

关键代码位置：

```text
decoupled_wbc/control/envs/g1/utils/state_processor.py:25
decoupled_wbc/control/envs/g1/utils/command_sender.py:63
decoupled_wbc/control/policy/g1_gear_wbc_policy.py:230
decoupled_wbc/control/envs/g1/utils/joint_safety.py:280
decoupled_wbc/control/envs/g1/g1_dex1_hand.py:80
```


## 1. 连接机器人网线

ThinkPad 有线地址：

```text
192.168.123.222
```

机器人地址：

```text
192.168.123.164
```

## 2. 激活采集环境

```bash
cd /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard
source /home/data-collection/miniconda3/etc/profile.d/conda.sh
conda activate decoupled_vla_collection
source "$CONDA_PREFIX/setup.bash"
export PYTHONPATH="$PWD"
```

## 3. 启动 PC Service

```bash
bash scripts/thinkpad_pico_services.sh start
bash scripts/thinkpad_pico_services.sh status
```

此命令不启动 DHCP。
保留服务后台运行。

## 4. 查看当前无线地址

```bash
ip -4 -brief address show dev wlp0s20f3
```

本次无线地址如下。
换网络后请重新查看。

```text
10.90.234.236
```

## 5. PICO 接入同一无线网

## 6. 打开 XRoboToolkit

服务地址填写第4步地址。
点击连接，确认 WORKING。
开启头部与双手柄追踪。
开启 Send，保持头显唤醒。

## 7. 检查实时姿态

```bash
python -m gear_sonic.scripts.check_pico_connection --timeout 15
```

必须看到以下结果。
失败时不要继续启动。
此检查不发送机器人动作。
采集中不要重复运行检查。

```text
PICO OK: headset + both controllers advancing
```

## 8. 选择夹爪场景

现有默认场景如下。
角度单位 rad，顺序左右。
角度越大，开口越大。

```text
default
released_q: [5.4, 5.4]
pressed_q: [2.5, 2.5]
```

配置目录：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/.runtime/dex1_profiles
```

需要调节时打开网页。

```text
http://127.0.0.1:8766
```

网页未运行时另开终端。
先执行第2步，再运行：

```bash
python -m gear_sonic.scripts.run_dex1_gripper_web --network enx00e04c36c4e8 --profile .runtime/dex1_profiles/default.json --port 8766
```

清空夹爪周围再启用控制。
调节左右开口，记录四值。
填写场景名并保存配置。
更换名称即可另存一组。
加载旧场景后可更新。
保存和加载不会产生动作。
采集前点击停止控制。
启动参数选择已保存场景。
采集中改文件需重启生效。

## 9. 执行完整只读预检

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-preset default \
  --camera-host 192.168.123.164 \
  --record-wrist-cameras \
  --camera-sync-tolerance-ms 30 \
  --camera-max-hold-ms 250 \
  --data-exporter-frequency 30 \
  --root-output-dir /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_three_camera \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer \
  --preflight-only
```

身体、夹爪、三图均须通过。
PICO 也须通过后再继续。
此步骤不启动真实控制。

## 10. 确认现场可安全运动

启动后将连接真实控制。
不要同时运行另一套控制。
先做好现场支撑与接管。
正式启动会退出官方模式。
未按手柄键也可能运动。

## 11. 正式启动采集

```bash
python gear_sonic/scripts/launch_decoupled_vla_collection.py \
  --interface enx00e04c36c4e8 \
  --hand-type dex1 \
  --dex1-preset default \
  --camera-host 192.168.123.164 \
  --record-wrist-cameras \
  --camera-sync-tolerance-ms 30 \
  --camera-max-hold-ms 250 \
  --data-exporter-frequency 30 \
  --root-output-dir /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs \
  --task-prompt "grab the red bottle" \
  --dataset-name pour_water_between_beakers_dex1_three_camera \
  --hand-task grab_middle_beaker \
  --no-pico-data-streamer
```

更换场景请改以下参数。

```text
--dex1-preset default
```

新任务请更换数据集名称。
默认不会覆盖已有数据。
附加数据流关闭不影响遥控。
控制启动即下发关节命令。
手臂约2秒趋向初始姿态。
夹爪可能趋向松开角度。
未启策略不等于官方站立。

## 12. 查看各窗格日志

control：身体及夹爪控制。
teleop：PICO 与遥操作。
exporter：数据录制与保存。
camera_viewer：相机预览。
Ctrl+b 后按方向键切换。

## 13. 开启身体策略

同时按 A+B+X+Y 后松开。
该组合切换身体策略开关。
日志须确认策略启动。

```text
starting policy
```

## 14. 开启上身遥操作

手柄放在合适的起始位置。
同时按 A+X 后松开。
等待上身遥操作就绪。

```text
Teleop policy active
```
再次 A+X 暂停上身遥控。
左右扳机分别控制左右爪。
大于0.5使用 pressed_q。
其余使用 released_q。

## 15. 选择 control 窗格

## 16. 按 c 开始录制

等 exporter 显示开始录制。
也可按左 Grip+A。

## 17. 操作手柄完成任务

## 18. 再按 c 结束录制

等待 exporter 保存完成。
未保存完成请勿关闭程序。
再按 c 可开始下一条。
录制中按 x 标记废弃。
也可按左 Grip+B 标记废弃。

## 19. 检查数据和三路视频

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs/pour_water_between_beakers_dex1_three_camera
```

主视角是 D435i。
左腕 JR0001，右腕 JR0002。
三图组内时间差不超过30ms。
独立同步组实测约18.7Hz。
30Hz录制可能复用整组。
此处对齐仅指时间同步。

## 20. 安全停止真实控制

现场接管后结束采集会话。
按Ctrl+反斜杠阻尼退出。
关闭最后界面也会阻尼退出。
Ctrl+b、d不再保留后台控制。

## 21. 最后停止 PC Service

确认采集控制已全部退出。

```bash
bash scripts/thinkpad_pico_services.sh stop
```

## 早期只读验证记录

2026-10-20 只读预检：
身体、双夹爪、三图通过。
PC Service 可启动与复用。
PICO 实时姿态检查已通过。
联合录制60帧并回读通过。
PICO 时间戳逐帧递增。
身体双夹爪共31维状态。
三路视频各60帧可解码。
本次扳机值始终为零。
尚未验证扳机变化及开合。
测试数据不能用于训练。
以上为早期只读测试记录。
运动模式仍未进行验收。

联合只读测试数据：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs/thinkpad_pico_readonly_test_20261020_57ee48bd
```

## 实际采集入口验收

身体控制待机持续正常。
身体及遥控消息约50Hz。
策略开启状态始终为false。
仅发送两次c测试录制。
未发送运动模式组合键。
真实状态和下发目标均31维。
删除超时处理后重新验证。
静止录制787行并回读通过。
三路视频各787帧可解码。
最大三图时间差约30ms。
最大接收缓存约152ms。
该调试数据不能用于训练。
正式采集请换数据集名称。

调试录制数据：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs/debug_thinkpad_dex1_three_camera_20261020_153446
```

Balance包含双腿及三关节腰。
腰部默认参考角均为0 rad。
动作残差会叠加到默认角。
默认关闭的是腰部上身IK。
未改参考角、初姿或模型权重。

实现位置：

```text
scripts/thinkpad_pico_services.sh
gear_sonic/scripts/check_pico_connection.py
gear_sonic/scripts/launch_decoupled_vla_collection.py
gear_sonic/scripts/run_decoupled_vla_data_exporter.py
decoupled_wbc/control/teleop/streamers/pico_streamer.py
```

## 实际急停验收

2026-10-20，仅实测一次。
机器人网线已恢复。
身体与双夹爪初始化通过。
三路相机连接通过。
PICO未连接，未开启遥控。
本次跳过头显连接预检。
未发送任何运动激活组合键。
监测策略开启状态为false。
左上control发送一次小写d。
旁路收到26帧身体阻尼命令。
29个身体关节均Kp=0、Kd=8。
阻尼后位置命令计数为0。
全部采集进程已退出。
停止流程总耗时约5.24秒。
首帧阻尼延迟未单独计时。
官方FSM接口返回3102。
3102表示请求发送失败。
未发现loco请求服务。
官方阻尼状态切换未验收通过。
未再次初始化或触发急停。
当前未运行采集控制会话。

真实测试记录位于项目下：

```text
.runtime/initialization_damping_test_20261020/lowcmd_monitor.json
.runtime/collection_damping_stop.json
```
