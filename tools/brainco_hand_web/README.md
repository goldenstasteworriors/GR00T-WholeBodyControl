# 强脑 Revo2 调节工具

完整操作顺序见仓库 `docs/SONIC_V11_ONBOARD_COLLECTION_SOP.md` 第6～9步。

- 机器人先运行宇树官方 `brainco_hand_service`，本工具直接收发其官方DDS话题，不访问Inspire的TCP接口。
- 客户端：`gear_sonic/utils/data_collection/brainco_hand.py`；网页：`gear_sonic/scripts/brainco_hand_web.py`。
- 每侧6电机，顺序为thumb、thumb_aux、index、middle、ring、pinky；q为0张开/1闭合；命令dq为归一化速度，默认1.0。
- 状态tau_est在官方服务里等于电流mA/1000，本工具展示为current_a，不称作力矩。

安装在机器人项目内（已完成官方SDK2安装，且已安装uv；首次运行）：

```bash
uv venv .venv_brainco_hand --python 3.10
export CYCLONEDDS_HOME=/usr/local
uv pip install --python .venv_brainco_hand/bin/python -r tools/brainco_hand_web/requirements.txt -e external_dependencies/unitree_sdk2_python
```

在机器人项目根目录启动（eth0换为实际DDS网卡）：

```bash
.venv_brainco_hand/bin/python -m gear_sonic.scripts.brainco_hand_web --network eth0 --check-seconds 5
.venv_brainco_hand/bin/python -m gear_sonic.scripts.brainco_hand_web --network eth0 --enable-commands
```

第一条只读检查两只手状态后退出。第二条提供127.0.0.1:8080网页，PC通过SSH转发访问。没有enable-commands参数时禁止发动作；缺失/陈旧反馈也拒绝动作。初始化只订阅，不发布任何手目标。

网页“发送”只控制所选一侧；“保存”只更新机器人项目下的配置，不发动作。保存后必须按SOP将配置下载回PC，再启动采集；启动器会上传PC配置供C++读取。

```text
gear_sonic/config/data_collection/brainco_hand_pose.json
```

左右open/grasp独立保存，采用官方归一化数值，不能使用Inspire配置或0～1000原始角度。采集前关闭网页，保留官方手服务，避免两个控制来源。退出软件不会保证手自动张开。

验证结果见 `docs/BRAINCO_HAND_VALIDATION.md`；Jetson安装和真实动作尚未验收。
