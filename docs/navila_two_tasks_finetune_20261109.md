# NaVILA 双任务单卡微调

仅使用两个任务：前进并停在试剂瓶与试管架之间、前进并停在烧杯前。底层继续沿用现有 Balance/Walk；训练输入是头部相机的最多 8 帧历史，按官方加载器组成 8 帧，图像处理分辨率为 384×384。腕部相机保留在原始采集数据中。

## 数据及监督语义

| 划分 | 完整轨迹 | 原始动作样本 |
|---|---:|---:|
| 训练 | 32 | 116 |
| 验证 | 5 | 29 |
| 测试 | 3 | 8 |

划分单位为完整轨迹，历史帧不会跨划分，也不会使用动作开始后的图像。所有终点停止均由用户在 2026-11-09 明确确认成功。平移距离及转向角度为速度指令按相机时间戳积分得到的弱监督，不能视为实测里程计。横移、后退及复合动作片段未加入标准动作训练。

补训的 `balanced_train.json` 将训练集中的 32 个停止样本各重复一次，共 148 个样本，其中停止 64 个；验证及测试文件保持原样。重复样本只改变训练采样权重。

生成数据：

```text
gear_sonic/scripts/prepare_navila_weak_sft.py
```

## 服务器与版本

SSH 别名为 `ai2-root-4080`，实际设备为 A800 80GB，使用物理 GPU 2。实验、环境、缓存与权重均位于独立项目目录：

```text
/nix/ykj/navila_two_tasks_20261109
```

环境按官方 NaVILA 安装流程创建于该目录的 `env` 子目录，跳过可选 CUDA 工具链安装。实际依赖保存于 `requirements.lock.txt` 和 `conda-explicit.txt`。主要版本为 PyTorch 2.3.0+cu121、Transformers 4.37.2、FlashAttention 2.5.8、PEFT 0.9.0。

上游代码固定到：

```text
76b98f233dd0fff05dfcd69435eec6740febff9d
```

基础模型与权重版本：

```text
a8cheng/navila-llama3-8b-8f
b2294e96581454468d6b94f38201f4f965ef48b7
```

模型文件的官方 LFS SHA256 校验结果位于基础模型目录的 `download_integrity.json`。下载工具使用普通 HTTP，避免此服务器上已观察到的 Xet 分片传输停滞。

## 训练方式

BF16、LoRA rank 16、alpha 32、dropout 0.05，单步 batch 为 1，累计 8 步更新。实际可训练参数为 41,943,040，均为语言模型的 LoRA 参数；视觉编码器、投影层及其他基础权重保持冻结。`tune_language_model=True` 配合 `lora_llm=True`，实际冻结范围以运行目录中的 `trainable_parameters.json` 为准。

采用单 Python 进程，并对官方前向中的分布式损失缩放使用单进程下等价的常数 1。梯度检查点使用 non-reentrant 方式。入口不会连接机器人。

初训设置为 3 epoch、学习率 2e-5，Trainer 实际完成 42 个更新步，记录 epoch 为 2.9。其验证损失从 4.3209 降至 0.9729，但重新加载后 8 个测试样本均输出前进。因此后续补训以实际生成的动作类别平衡准确率选模型，同时记录停止召回率、停止精确率及指令格式合法率，不仅依赖 token loss。

补训从初训适配器继续，使用 148 个平衡采样样本、学习率 1e-4、6 epoch 配置。每轮验证生成时会剥离监督答案，只将任务描述与历史图像送入模型。测试轨迹不参与优化或模型选择。

最终任务适配器按验证集“前进 F1 与停止 F1 的平均值”选取，同时考虑漏停及提前停止；格式不合法的输出计为错误。训练内部的全动作类别平衡准确率仍保留作诊断。选择工具可在检查点落盘后运行，独立保存当前最优适配器，避免被训练状态轮转移除：

```bash
python gear_sonic/scripts/select_navila_goal_adapter.py --run-dir /nix/ykj/navila_two_tasks_20261109/runs/20261109_navila_two_tasks_stop_balanced_lora_gpu2 --output /nix/ykj/navila_two_tasks_20261109/runs/20261109_navila_two_tasks_selected_goal_lora_gpu2
```

该命令使用训练环境中的 Python，整行执行。`goal_selection_manifest.json` 记录实际选中检查点及验证集 F1，`source_run_summary.json` 保留训练内部选模结果，二者有明确区分。

## 复现补训

先进入服务器终端。以下第一条命令独立执行；第二个代码块整段复制，反斜杠表示续行。

```bash
cd /nix/ykj/navila_two_tasks_20261109/collection_tools
```

```bash
NAVILA_RUN_NAME="$(date +%Y%m%d_%H%M%S)_navila_stop_balanced" \
bash scripts/train_navila_two_tasks_single_gpu.sh \
  --lab-train-file balanced_train.json \
  --lab-initial-adapter /nix/ykj/navila_two_tasks_20261109/runs/20261109_navila_two_tasks_bf16_lora_gpu2 \
  --lab-generation-eval \
  --num_train_epochs 6 --learning_rate 1e-4 \
  --metric_for_best_model action_balanced_accuracy --greater_is_better True
```

## 查看结果

初训与补训分别保存在：

```text
/nix/ykj/navila_two_tasks_20261109/runs/20261109_navila_two_tasks_bf16_lora_gpu2
/nix/ykj/navila_two_tasks_20261109/runs/20261109_navila_two_tasks_stop_balanced_lora_gpu2
```

`run_summary.json` 记录最佳检查点、选模指标、验证及测试损失、显存峰值和非零 LoRA B 张量数量。`*_generation_step_*.json` 记录实际生成的指令；`sample_token_audit.json` 记录真实参与损失计算的 token。检查点仅保留本次运行最新两份及最佳一份，运行目录根部另存最佳适配器。

适配器须结合上述基础模型与官方 NaVILA 实现加载。所有指标均为小规模同场景离线评估，距离标签仍是弱监督，不等同于实机导航成功率。
