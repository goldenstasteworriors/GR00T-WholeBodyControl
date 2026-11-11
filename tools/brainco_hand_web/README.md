# 强脑 Revo2 调节工具

完整操作顺序见仓库 `docs/SONIC_V11_ONBOARD_COLLECTION_SOP.md` 第6～9步。

- 机器人先运行宇树官方 `brainco_hand_service`，本工具直接收发其官方DDS话题，不访问Inspire的TCP接口。
- 客户端：`gear_sonic/utils/data_collection/brainco_hand.py`；网页：`gear_sonic/scripts/brainco_hand_web.py`。
- 每侧6电机，顺序为thumb、thumb_aux、index、middle、ring、pinky；q为0张开/1闭合；命令dq为归一化速度，默认1.0。
- 状态tau_est在官方服务里等于电流mA/1000，本工具展示为current_a，不称作力矩。

安装在机器人项目内（已安装uv、CMake、C编译器和Python 3.10）：

```bash
bash tools/brainco_hand_web/install_env.sh
```

在机器人项目根目录启动（eth0换为实际DDS网卡）：

```bash
bash tools/brainco_hand_web/run.sh --network eth0 --check-seconds 5
bash tools/brainco_hand_web/run.sh --network eth0 --enable-commands
```

第一条只读检查两只手状态后退出。第二条提供127.0.0.1:8080网页，PC通过SSH转发访问。没有enable-commands参数时禁止发动作；缺失/陈旧反馈也拒绝动作。初始化只订阅，不发布任何手目标。

安装脚本保留已有虚拟环境，在项目 `.runtime/brainco-web-native/install` 安装完整的 CycloneDDS 0.10.2 开发库，避免某些SDK2分发包缺少 `idl/retcode.h`。不改动系统DDS或手部服务。默认使用清华PyPI镜像，`BRAINCO_PYPI_INDEX` 可指定其他可用源。

已有离线源码时，安装脚本可接收压缩包路径；默认寻找 `.runtime/brainco-diagnostics/cyclonedds-0.10.2.tar.gz`，缺失时从官方codeload下载并检查SHA256。离线源需为官方0.10.2标签归档。启动脚本自动设置正确的 `CYCLONEDDS_HOME` 和库路径，不必手输；不要重建环境或把变量名写成 `CLONEDDS_HOME`。

网页“发送”只控制所选一侧；“保存”只更新机器人项目下的配置，不发动作。保存后必须按SOP将配置下载回PC，再启动采集；启动器会上传PC配置供C++读取。

```text
gear_sonic/config/data_collection/brainco_hand_pose.json
```

左右open/grasp独立保存，采用官方归一化数值，不能使用Inspire配置或0～1000原始角度。采集前关闭网页，保留官方手服务，避免两个控制来源。退出软件不会保证手自动张开。

2026-11-11已在Orin NX、Ubuntu 20.04、Python 3.10上验证项目内CycloneDDS安装、Flask导入、双手新鲜反馈、网页及状态接口HTTP 200；未通过网页发送实际动作目标。此前协议验证见 `docs/BRAINCO_HAND_VALIDATION.md`。
