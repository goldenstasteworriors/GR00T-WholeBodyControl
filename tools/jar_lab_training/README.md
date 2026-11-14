# G1 实验室操作：仅用 observation 训练 JAR

## 数据图像＋仿真状态视频（2026-11-14）

`simulation_video.py` 使用1小时版本每项验证NLL最低的权重，每项选第一段留出episode，回放最多15秒。
输入是三视角原始RGB因果历史、仿真双臂/夹爪16维位置、固定英文操作指令；没有框输入，也不是纯图像模型。
只有frame0用记录状态初始化。之后双臂输入取自600Hz MuJoCo积分结果，夹爪取自显式选择的理想伺服状态；
真实记录状态只供右侧对照/误差计算。rollout.npz保存每次输入、预测和积分后状态，并核对完整反馈链。
`sim_physics.py` 在内存中派生标准SONIC rev1 G1模型，固定骨盆/腿/腰，只积分命名顺序核验后的14个臂关节。
双臂PD为60/1.5，参考限速0.25rad/s；力矩按XML关节限值限制，偏置补偿限幅10Nm/腕5Nm、启动斜率10Nm/s。
PD使用MuJoCo affine actuator，使implicitfast能处理腕部速度反馈。原XML/mesh文件保持不变。

当前夹爪方案为 `--gripper-mode ideal`：原生角度0–5.4rad、0.5rad/s理想伺服，仅显示数值状态条。
没有Dex1几何、夹爪接触或额外工具惯量；隐藏标准资产的rubber-hand显示，避免把它误作Dex1。
视频并排显示真实三相机、模型驱动的仿真姿态和录制观测姿态。录制图像不随仿真动作变化，未仿真物体操作成功率。
播放为30fps，模型推理按离线时间完成，视频内单列实际推理耗时。每项首帧核对16-token生成与512-token生成前缀一致。
程序不导入机器人SDK/ROS/DDS，不创建真机控制通道。模型和仿真在本次独立uv环境中运行，只读继承JAR依赖，不修改原Conda/CUDA/驱动。

服务器生成目录，以下为一整行路径：

```text
/share/personal/yekangjie/JAR_lab_primitives_20261111/retrain_1h_20261113_A800/visualizations_20261114
```

环境、模型/资产路径和运行参数见该目录run-plan.json；每段视频旁有summary.json及rollout.npz。

## 2026-11-13：每项1小时，分别保留两种最低NLL

运行于A800-1的GPU1、2，独立目录为原实验下的 `retrain_1h_20261113_A800`。
用户已授权删除上一轮上传ModelScope的本地checkpoint。`reclaim_uploaded.py` 先逐份核对云端和本地SHA256/大小，
全部通过后才清理对应模型、导出硬链接和optimizer；云端副本及本地日志、配置和归档标记保留。
复用已校验的只读vendor/cache软链接，冻结本次代码到独立Git worktree；六项都重新从官方OLMo2初始化。
运行设置仍是GPU1、2两卡DDP、global batch16、512桶、双臂与Dex1的16维未来observation；不训走路、不读action。

`retrain.py` 先检查349段数值校验和、原episode分组隔离、图像因果性和未来目标，再交给独立supervisord。
A800-1所选两卡当前均为0MiB，本轮启动门槛为显存低于1MiB、空闲磁盘至少28GiB；不停止其它训练/推理任务。
只暂停既有已核实的所选卡保卡进程，退出时恢复；本次检查没有发现保卡进程，指定README也不存在，因此未启动额外保卡任务。
空闲后自动执行实际两卡5步、checkpoint逐tensor回读和单样本自回归检查；全部通过后才开始正式队列。
每项预算3600秒，从初始化和初始验证之后计时，包含常规评估/保存；到时的最终评估、保存和离线评估另有少量开销。
SSH或本机断线不影响这个服务器端服务。
嵌套目录会触发Linux Unix socket长度限制，因此服务使用短名s.sock；训练临时目录使用原实验下的t_<hash>，
实际位置记录在retrain-recipe.json。数据和权重仍保存在本轮独立目录/原缓存中，不使用系统临时目录。

第5步、之后每500步及结束时，在eval模式下计算完整词表teacher-forced NLL：

- 训练NLL：训练episode固定切点最多64个，均匀抽取；不是随机batch loss，也不宣称覆盖全训练集。
- 测试NLL：沿用上一轮的留出验证集，每段固定0/25/50/75%切点。它用于选权重，不是独立最终测试集。
- 各自最低分别记录在 `best_train.json` 与 `best_validation.json`，`best.json`兼容指向验证最低。
  JSON记录step、NLL和具体模型路径；相同步骤只保存一份。评估曲线位于 `train-evaluation.jsonl`、`validation.jsonl`。
- 训练期间保留最近两份模型及两种NLL最佳模型，只有最新checkpoint保留optimizer供断点续训。
  正常完成后只留下两种NLL赢家的FP32模型权重，不保留optimizer；中断退出则保留恢复状态。
  `latest.json`在完成后表示最新保留的赢家；真正结束步数以status.json/last_training_checkpoint.json为准。

下面每行都是一条完整命令，分别执行，无需手动折行。prepare仅做CPU检查，start会先prepare再启动资源等待/训练服务。
相同目录重复start不会启动第二个队列；使用status查看等待或训练状态。

```bash
ssh -- 'A800-1' '/share/personal/liyanjun/yanjun/envs/jar/bin/python /share/personal/yekangjie/JAR_lab_primitives_20261111/code/tools/jar_lab_training/retrain.py prepare --root /share/personal/yekangjie/JAR_lab_primitives_20261111/retrain_1h_20261113_A800'
ssh -- 'A800-1' '/share/personal/liyanjun/yanjun/envs/jar/bin/python /share/personal/yekangjie/JAR_lab_primitives_20261111/code/tools/jar_lab_training/retrain.py start --root /share/personal/yekangjie/JAR_lab_primitives_20261111/retrain_1h_20261113_A800'
ssh -- 'A800-1' '/share/personal/liyanjun/yanjun/envs/jar/bin/python /share/personal/yekangjie/JAR_lab_primitives_20261111/code/tools/jar_lab_training/retrain.py status --root /share/personal/yekangjie/JAR_lab_primitives_20261111/retrain_1h_20261113_A800'
```

以下记录上一轮4小时配方。

本工具复用指定服务器 JAR 的 OLMo2 模型、官方权重初始化、512 桶 codec 和 packing，不修改原 JAR。
运行时先冻结 `jar/` 的独立快照到实验目录 `vendor/`，并记录原仓库版本与文件哈希。

数据采用 `lab_primitives_train_20261110_boxed_v4`，仅选择 6 个非 WALK 操作，共 349 段。
三视角输入按 ego、left wrist、right wrist 排列，不读框。数值只读取 `observation.state`，
严格按关节名称提取原索引 15–30：左臂 7、右臂 7、左夹爪 1、右夹爪 1。
输入为时刻 t 的上述 16 维状态和不晚于 t 的 4 帧去重历史；目标为 t+1 开始的 32 帧
同样的 observation。末尾重复最后观测作为输入 padding，其标签屏蔽；没有未来观测的最后切点排除。
整个准备过程不读取任何 `action.*` 或 `teleop.*` 列。

保留数据原生 30 Hz，不套用旧 ARX 的 25 Hz 或 14D。三路视频已在处理阶段按帧号对齐。
帧龄以切片视频帧时钟为基准，叠加同一相机元组内相对于最晚采集时间的非正延迟。
录制中的绝对 wall-clock 不冒充裁剪后 episode 时间；这无法恢复未记录的总采集延迟。

每个操作独立从官方 OLMo2 权重初始化，完整源 episode 分组，seed42 留出 10% 向上取整，
归一化只使用训练 episode 的 observation；臂部用 q01/q99，夹爪用 min/max。
最终运行于 `ai2-root-4080`（实际为 A800 80GB），只用 GPU 1、2，
两卡 DDP，global batch16（每卡8），FP32 参数/AdamW、BF16 计算，全部参数训练。
20 步 warmup 后按 4 小时时间预算 cosine 降到 10% LR；不跨操作继承机器人权重。
训练预算从初始化和初始验证结束后计时，包含训练期间的常规验证/保存。
到时完成当前 step、验证并保存，最后保存与离线自回归评估可能增加少量墙钟时间。

第 1 步检查有限梯度和参数更新，第 5 步保存并逐 tensor 回读；之后每 1000 步验证/保存，
避免共享盘上的完整 Adam checkpoint I/O 挤占四小时预算；保留最近两份和验证最佳权重。
监督损失是完整词表的 NLL。每项结束用验证集每个 episode 的固定中点自回归生成 512 个 token，
报告未来 observation 的臂部/夹爪 MAE、RMSE、逐关节 MAE，并对照保持当前状态的基线。
这些结果属于离线预测误差，不是闭环操作成功率；观测预测也不等同于可直接执行的 actuator 命令。

`run_queue.py` 在项目专用 supervisord 服务中运行，依次训练 6 项，每项 14400 秒。
它使用独立队列锁，失败立即停止并记录状态；重启时从当前项最新完整 checkpoint 恢复剩余预算。
只暂停所选卡上已核实的 matmul 保卡进程，保留其显存分配，服务退出时恢复；不操作 GPU 0、3。
所选卡显存占用高于 12000MiB 时等待，不停止其他训练；先运行实际两卡 5 步、
checkpoint 完整回读及一次自由生成检查，通过后才开始正式队列。未通过的检查会阻止正式训练。
恢复验证 world size、训练配方和数据签名。不操作已有 GPU 保活进程，不更改环境、CUDA 或驱动。
日志、缓存、权重和临时目录都位于 `/share/personal/yekangjie/JAR_lab_primitives_20261111`。

## ModelScope 上传托管

`upload_service.py` 在训练服务器的独立 supervisord 中运行；本机或 SSH 断线不影响它。
仅跟随本轮固定6项训练：某项训练和离线评估均完成后，上传其保留的全部模型权重。
远端大小及SHA256核对成功后写回执；网络失败保留上传缓存并重试，全部6项上传后退出。
原训练服务独立运行。上传凭据保留在服务器任务目录的 `.private` 中，不进入导出包或 Git。

以下每行是一条完整命令，按需分别执行，无须手动换行：

```bash
ssh -- 'ai2-root-4080' '/etc/aibox/bin/python3.9 /share/personal/yekangjie/JAR_lab_primitives_20261111/code/tools/jar_lab_training/upload_service.py status'
ssh -- 'ai2-root-4080' '/etc/aibox/bin/python3.9 /share/personal/yekangjie/JAR_lab_primitives_20261111/code/tools/jar_lab_training/upload_service.py start'
```
