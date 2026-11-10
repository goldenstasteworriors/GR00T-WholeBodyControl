# Inspire 机载网页调节工具

原文件名是 **ctlhand.py**，不是 ctrl_hand.py。从用户保存的机器人备份原样复制：

```text
/home/ykj/Downloads/data_collection/VLA/locodemo/ctlhand.py
```

原文件和交付副本 SHA256 一致：

```text
86538b20698e205aabbb29865864b56a2e42630fcf500c620d1462e490494d3f
```

机器人克隆本交付版本后，在项目根目录逐行执行（需要已安装uv；仅首次创建环境）：

```bash
uv venv .venv_hand_web --python 3.10
uv pip install --python .venv_hand_web/bin/python -r tools/inspire_hand_web/requirements.txt
.venv_hand_web/bin/python tools/inspire_hand_web/ctlhand.py --help
.venv_hand_web/bin/python tools/inspire_hand_web/ctlhand.py --left 192.168.123.210 --right 192.168.123.211 --host 127.0.0.1 --web-port 8080
```

左右IP使用实际值；这是直接Modbus控制，不需要另开DDS桥接。PC另开终端并保持运行：

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.164
```

PC浏览器打开：

```text
http://localhost:8080/
```

网页支持0～100%的六电机目标及手部状态读取。选左/右手分别调节，记录张开、抓握各6个目标百分比，除以100后写入PC采集配置；原始angle值0～1000则除以1000。不要把受负载影响的实测反馈值当成目标值。

```text
gear_sonic/config/data_collection/inspire_hand_pose.json
```

六电机顺序：小指、无名指、中指、食指、拇指弯曲、拇指旋转。网页不会自动保存上述JSON；读数中的force/current/status也不能当作角度目标。调节完成Ctrl+C退出服务及隧道，再启动采集手桥接。完整安装与数据采集顺序见仓库内SOP。

验证：Python 3.10下完成CLI、百分比转换、锁定版本pymodbus调用兼容性、模拟手对象的网页接口，以及真实本地HTTP启动测试；未向机器人/灵巧手发送请求。依赖锁定在requirements.txt，Jetson安装与真实手部动作仍需现场验证。
