# BrainCo Revo2 适配验证记录

日期：2026-11-10。分支：feat/brainco-hand-onboard。未连接、未控制真实机器人。

## 接口依据

用户指定页面：https://support.unitree.com/home/zh/G1_developer/brainco_hand 。普通抓取没有正文，独立浏览器返回拦截状态567；因此未声称成功读到该页正文。

实际交叉核验依据为宇树官方仓库：

```text
https://github.com/unitreerobotics/brainco_hand_service
commit d71996b6999edb2f838a3dca3d9621429a2ef966
```

核验README_zh-CN.md、main.cpp、include/param.h、test/test_brainco_hand_server.cpp、setup_autostart.sh及CMakeLists.txt。未修改官方服务源码。服务实际CLI为network_interface；自动启动脚本启动的是brainco_hand_server，而非循环动作测试程序。

| 项目 | 官方实现及本项目表达 |
| --- | --- |
| 左/右命令 | rt/brainco/left/cmd、rt/brainco/right/cmd，MotorCmds_，每侧6条 |
| 左/右反馈 | rt/brainco/left/state、rt/brainco/right/state，MotorStates_ |
| 顺序 | 官方README：thumb、thumb_aux、index、middle、ring、pinky |
| q | 归一化位置，0张开、1闭合；服务乘1000发送给串口SDK |
| 命令dq | 归一化速度，推荐1.0；不是弧度每秒 |
| 状态dq | 服务的status.speeds/1000，单独记录 |
| 状态tau_est | status.currents/1000，按电流A记录，不宣称为力矩 |
| 左右识别 | 串口设备SKU＋ID0x7e/0x7f，不猜USB编号、不填左右手IP |
| 新鲜度 | 机器人收到有效DDS消息后的steady_clock年龄；没有设备采样时间戳 |

41维state/action保留身体29＋左手6＋右手6布局；手部名字与数值方向改为BrainCo语义。新增速度12维、电流12维、接收年龄2维；元数据保存手型、开合配置及单位，不能与旧Inspire数据静默混用。

## 已执行的验证

1. x86_64主C++程序完成配置、编译及链接。新增yaml-cpp显式链接；只使用本机已有CUDA/TensorRT，未安装或更改驱动。

```bash
TensorRT_ROOT=/home/ykj/TensorRT cmake -S gear_sonic_deploy -B .runtime/brainco-build -DCMAKE_BUILD_TYPE=Release
cmake --build .runtime/brainco-build --target g1_deploy_onnx_ref -j 4
```

2. C++ BraincoHands使用真实Unitree SDK/IDL在隔离DDS domain221、lo网卡收发：初始化无命令，左右各6电机，速度1.0，配置目标/限幅与实际发布目标一致；速度、电流、年龄映射通过；NaN反馈不刷新年龄，-ffast-math下NaN配置也被拒绝。
3. Python客户端＋网页使用真实Unitree Python SDK在domain222、lo网卡验证左右话题、q/dq/current；原子保存只更新所选侧，不发送动作；禁止发送模式、畸形请求、NaN、陈旧反馈均被拒绝；本地HTTP首页通过。
4. 使用实际机器人模型验证wrist_yaw_link FK；强脑归一化手值没有灌入旧手模型的弧度关节。ZMQ解包后正确保留诊断字段，41维state/action和新增字段经过Parquet读回检查。
5. 实际Gr00tDataExporter保存4帧合成episode：Parquet 4行、state/action各41维、12电机电流/速度、视频解码4帧，meta/info.json维度通过。测试数据由本次创建，结束后删除。
6. 未收到反馈、单侧陈旧、无已发布目标、错误维度、NaN及PC状态流陈旧时不写录制帧。
7. 捕获一键启动器实际生成命令：先验证/上传配置，机器人参数包含hand-type和配置路径，导出端选择BrainCo；不启动Inspire桥接。生成的导出CLI执行help解析通过；此测试用模拟subprocess避免SSH、tmux和真实控制。
8. 新增网页、导出器及启动器的help和Python导入通过；SOP的Shell/JSON语法及参数对应关系单独检查。临时C++测试源码在测试后删除，不提交测试脚本。
9. 以实际网页CLI启动回环服务，在浏览器操作六电机滑块并保存，确认只更新测试配置中的指定一侧；没有真实反馈时点击发送，页面明确返回拒绝。预览使用本次创建的独立配置，未修改交付默认配置，完成后已关闭服务和专用浏览器。

最终文档检查：SOP与工具说明共36个Shell代码块通过bash -n；JSON、0～10步骤顺序、启动器参数及不含Inspire桥接参数的检查通过。

## 尚未验证

- Jetson ARM64全新安装、官方串口服务实际接手、设备SKU识别、PICO实际遥操作、手部动作及真机采集。
- 官方服务本机独立构建尝试因测试主机缺少Boost program_options开发包而停在配置阶段；SOP明确安装该系统依赖。没有为此改动本机系统或官方源码，不能把主程序构建成功表述成官方串口服务已在目标Jetson验收。
- SDK反馈没有设备级采样时间，本项目接收年龄不代表传感器延迟；速度是归一化反馈，电流不是接触力或关节力矩。状态缺失时仅暂停录制，不擅自改变身体控制策略。

所有隔离DDS测试仅使用lo网卡及专用domain，不向机器人网段发送命令。部署前需按SOP录制并检查至少一条真实短episode。
