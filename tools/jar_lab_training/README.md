# G1 实验室操作：仅用 observation 训练 JAR

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
四卡 DDP，global batch32，FP32 参数/AdamW、BF16 计算，全部参数训练。
20 步 warmup 后按 4 小时时间预算 cosine 降到 10% LR；不跨操作继承机器人权重。
训练预算从初始化和初始验证结束后计时，包含训练期间的常规验证/保存。
到时完成当前 step、验证并保存，最后保存与离线自回归评估可能增加少量墙钟时间。

第 1 步检查有限梯度和参数更新，第 5 步保存并逐 tensor 回读；保留最近两份和验证最佳权重。
监督损失是完整词表的 NLL。每项结束用验证集每个 episode 的固定中点自回归生成 512 个 token，
报告未来 observation 的臂部/夹爪 MAE、RMSE、逐关节 MAE，并对照保持当前状态的基线。
这些结果属于离线预测误差，不是闭环操作成功率；观测预测也不等同于可直接执行的 actuator 命令。

`queue.py` 在项目专用 supervisord 服务中运行，依次训练 6 项，每项 14400 秒。
它使用独立队列锁，失败立即停止并记录状态；重启时从当前项最新完整 checkpoint 恢复剩余预算。
恢复验证 world size、训练配方和数据签名。不操作已有 GPU 保活进程，不更改环境、CUDA 或驱动。
日志、缓存、权重和临时目录都位于 `/share/personal/yekangjie/JAR_lab_primitives_20261111`。
