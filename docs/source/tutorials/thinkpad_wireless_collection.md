# ThinkPad 无线采集流程

当前只使用此流程。
有线与机载完善已暂停。
以下均在 ThinkPad 操作。
命令按行执行。
反斜杠表示跨行续写。

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
Ctrl+b、d 只退出界面。
它不会停止后台控制。

## 21. 最后停止 PC Service

确认采集控制已全部退出。

```bash
bash scripts/thinkpad_pico_services.sh stop
```

## 验证记录

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
尚未启动真实身体控制。
完整动作录制尚未验收。

联合只读测试数据：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/outputs/thinkpad_pico_readonly_test_20261020_57ee48bd
```

实现位置：

```text
scripts/thinkpad_pico_services.sh
gear_sonic/scripts/check_pico_connection.py
gear_sonic/scripts/launch_decoupled_vla_collection.py
gear_sonic/scripts/run_decoupled_vla_data_exporter.py
decoupled_wbc/control/teleop/streamers/pico_streamer.py
```
