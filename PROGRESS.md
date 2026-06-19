# PROGRESS

记录 workflow agent 的进度、测试、阻塞和恢复点。

## 2026-06-19T13:05:00+08:00 - PKU anchor eval 修复并完成三组诊断

### 问题定位
- 上一轮新 anchor logging eval 超时，不是 `im_eval_callback.py` 的 anchor 统计本身卡住，而是运行命令漏了旧成功 eval 的关键 Hydra overrides：
  - `++eval_callbacks=im_eval`
  - `++run_eval_loop=False`
  - `++eval_output_dir=...`
  - `+manager_env/terminations=tracking/eval`
  - `++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=8`
  - `++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered`
- 漏掉这些参数后，`eval_agent_trl.py` 没有触发 `ImEvalCallback.evaluate_policy()`，而是进入普通 eval loop，因此长时间不生成 `metrics_eval.json`。
- PKU `.venv/bin/accelerate` 的 shebang 指向旧路径 `/home/nvme02/GR00T/GR00T-WholeBodyControl/.venv/bin/python3`，直接执行会返回 `126`；本轮改用 `.venv/bin/python -m accelerate.commands.launch`，没有修改环境。

### 已执行
- 在 PKU `/home/nvme02/GR00T/GR00T` 用 GPU `0/1/2` 并行完成三组 eval：
  - `pr12000`: `model_step_012000.pt`
  - `hy2000`: `model_step_002000.pt`
  - `tf2000`: `model_step_002000.pt`
- 数据集：
  `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_eval_compare8`
- 三组均正常进入 `Evaluating policy` 并写出：
  - `logs_eval/20260619_anchor_compare_h2_fixed/metrics_pr12000/metrics_eval.json`
  - `logs_eval/20260619_anchor_compare_h2_fixed/metrics_hy2000/metrics_eval.json`
  - `logs_eval/20260619_anchor_compare_h2_fixed/metrics_tf2000/metrics_eval.json`
- 已生成并同步回本机：
  - `.workflow/artifacts/h2_eval_anchor_diagnostics_pku.md`
  - `.workflow/artifacts/h2_eval_anchor_diagnostics_pku.csv`

### 新 anchor 诊断结论
- `pr12000`: `success=0.000000`, `progress=0.192842`, `mpjpe_g=0.070260`, `anchor_xy_mean=0.026190`, `anchor_xy_max=0.118835`, `heading_mean=0.065750`, `heading_max=0.394442`。
- `hy2000`: `success=0.250000`, `progress=0.399373`, `mpjpe_g=0.121227`, `anchor_xy_mean=0.083130`, `anchor_xy_max=0.433407`, `heading_mean=0.099540`, `heading_max=0.943376`。
- `tf2000`: `success=0.625000`, `progress=0.689094`, `mpjpe_g=0.143952`, `anchor_xy_mean=0.089988`, `anchor_xy_max=0.420850`, `heading_mean=0.079153`, `heading_max=0.358851`。
- 关键判断：`pr12000` 的 global anchor XY / heading 误差已经很小，但仍 8/8 early terminate；因此“效果一般”的主因不应简单归结为 root/heading global drift。它更像是过度追求局部精度/严格 tracking 后，策略鲁棒性和恢复能力不足。
- `hy2000/tf2000` 的成功动作主要是 checking/itching/brush/body stretch/sneeze 等较静态或上肢动作；这些成功样本允许更大的 anchor XY 偏差。尤其 `tf2000` 的 `body_stretch` 成功但 `mpjpe_g=0.273337`、`vr_g=0.379224`、`anchor_xy_mean=0.199024`，说明 survival 是以明显全局/末端漂移换来的。
- `walk_ff_loop_315` 在三组都失败早期，且 `hy2000` 的 `anchor_xy_mean=0.193922`、`heading_mean=0.354494` 最差；后续 locomotion 类动作要单独处理，不应和静态上肢动作共用同一组 reward/termination 权重判断。

### 下一步建议
- 不建议只加大 global-anchor reward/termination。`pr12000` 已证明 anchor 小误差不等于能跑完整动作。
- 下一组最小实验应分两条：
  1. `hy2000/tf2000` 路线加轻量 global drift guard：控制 `anchor_xy_max`/heading 爆点，目标是在保留 progress 的同时压低 `mpjpe_g` 和 VR/foot 漂移。
  2. `pr12000` 路线放松 early termination 或做 recovery/curriculum：目标是提高 progress，而不是继续加精度项。
- 评估必须按 motion type 拆分：locomotion (`jog/walk/jump`) 与 static/upper-body (`checking/itching/brush/body_stretch/sneeze`) 分开看。

## 2026-06-18T23:56:45+08:00 - workflow 恢复检查：确认真实 compare8 metrics 缺失，标记阻塞

### 已读取
- `AGENTS.md`：确认中文回答、SonicMJ/mjlab 迁移使用 uv、不得修改 CUDA/驱动/系统 GPU 组件、不得修改 workflow 源码、代码相关回答需注明位置。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认 `plan_approved=true`，当前应从已确认 PLAN 的代码阶段恢复；上一轮已完成 M0/M1/M2 logging 第一层，但因真实 compare8 metrics 缺失暂停。
- `.workflow/artifacts/open_questions.json`、`.workflow/artifacts/h2_checkpoint_diagnosis.md`、`.workflow/artifacts/h2_eval_diagnostics.md`：确认已有诊断工具与汇总文档，当前诊断输出只记录缺失输入。
- `README.md`、`docs/README.md`：上游文档仍偏 IsaacLab；本轮仍以 `AGENTS.md` 的 SonicMJ/mjlab + uv 约束为准。
- `H2_TRANSFER_EXPERIMENT_LOG.md`：确认目标 compare8 数据与 PKU eval 路径。

### 判断
- 当前没有 `PERSON.md` open 条目需要处理。
- `PLAN.md` 已确认，不能回到计划阶段；应继续 M1 的真实 per-motion/time-series 诊断。
- 本机扫描未找到任何 `metrics_eval.json`，也无法访问 `H2_TRANSFER_EXPERIMENT_LOG.md` 中记录的 PKU 路径：
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260617_precision_compare_h2/pr12000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260618_hybrid_compare_h2/hy2000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260615_tracking_first_compare_h2/tf2000/metrics_eval.json`
- 因缺少真实 per-motion/time-series metrics，继续修改 global XY / heading reward 或 termination ablation 会违反“不盲目继续训练/调参”的阶段目标。

### 本轮命令与结果
- `sed -n '1,220p' /home/ykj/tool/WORKFLOW/skills/workflow/SKILL.md`：成功读取 workflow skill；未修改 workflow 源码。
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md task.json`：成功读取 workflow 必读文档和任务摘要。
- `find .workflow -maxdepth 3 -type f | sort`：成功列出 workflow artifacts/runtime 文件。
- `git status --short --branch`：当前分支 `cross_humanoid`；存在 workflow 文档、`gear_sonic/trl/callbacks/im_eval_callback.py` 和新增诊断脚本等未提交改动；本轮未回退任何已有改动。
- `sed -n ... .workflow/artifacts/open_questions.json .workflow/artifacts/h2_checkpoint_diagnosis.md .workflow/artifacts/h2_eval_diagnostics.md README.md docs/README.md`：成功读取 artifacts 和必要 README/docs。
- `rg --files -g 'metrics_eval.json' -g '*metrics*.json' logs_eval logs_rl .workflow gear_sonic sonic_mj 2>/dev/null`：未找到可用 metrics 文件，命令返回 1。
- `find /home/ykj/project/SONICMJ/GR00T-WholeBodyControl -path '*/wandb' -prune -o -path '*/.venv' -prune -o -name 'metrics_eval.json' -print`：无输出，确认仓库内无 `metrics_eval.json`。
- `find logs_eval logs_rl -maxdepth 6 -iname '*h2*' -o -iname '*pr12000*' -o -iname '*hy2000*' -o -iname '*tf2000*' ...`：只找到早期本地 `logs_rl/TRL_H2_Track/.../sonic_h2_test-*` 目录，未找到目标 compare8 eval 输出。
- `date --iso-8601=seconds`：记录时间戳 `2026-06-18T23:56:45+08:00`。
- 补充恢复核对：
  - `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh resume --workspace "$PWD"`：成功，输出 `phase=resumed`、`plan_approved=true`。
  - `find logs_eval logs_rl .workflow -name metrics_eval.json -print`：无输出，确认这些本地目录下没有可用 `metrics_eval.json`。
  - 逐个 `test -f` 检查 `pr12000`、`hy2000`、`tf2000` 的 PKU `metrics_eval.json` 路径：均为 `MISSING`。
  - `uv run python --version`：通过，输出 `Python 3.10.20`。

### 更新
- 在 `.workflow/artifacts/open_questions.json` 新增 blocking 问题 `Q20260618-004`，要求提供 `pr12000 / hy2000 / tf2000` 的真实 compare8 `metrics_eval.json` 或在可访问 PKU 路径的机器重跑 eval。
- 更新 `task.json`：`last_result=blocked`，写入 `blocked_reason` 和恢复摘要。

### blocked_reason
缺少 `pr12000 / hy2000 / tf2000` 的真实 compare8 `metrics_eval.json`；本机没有任何 `metrics_eval.json`，且 `H2_TRANSFER_EXPERIMENT_LOG.md` 记录的 PKU `logs_eval` 路径当前不可访问。需要用户提供这些文件，或在可访问 PKU 路径的机器运行：

```bash
SONIC_SAVE_EVAL_TRACE=1 WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/eval_agent_trl.py \
  checkpoint=<CHECKPOINT> use_mjlab=True sim_type=mjlab headless=True \
  ++num_envs=8 ++output_dir=<EVAL_OUT>
```

拿到三个输出后再运行：

```bash
uv run python gear_sonic/scripts/h2_eval_diagnostics.py \
  --case pr12000=<PR12000_EVAL_OUT>/metrics_eval.json \
  --case hy2000=<HY2000_EVAL_OUT>/metrics_eval.json \
  --case tf2000=<TF2000_EVAL_OUT>/metrics_eval.json \
  --output-md .workflow/artifacts/h2_eval_diagnostics.md \
  --output-csv .workflow/artifacts/h2_eval_diagnostics.csv
```

## 2026-06-19T01:01:00+08:00 - PKU 同步与 H2 compare8 诊断执行

### 已执行
- 本机提交并推送 `6e57c63 Add H2 eval anchor diagnostics` 到 `origin/cross_humanoid`。
- PKU 服务器 `/home/nvme02/GR00T/GR00T` 因远端 `git fetch` 走 HTTPS 卡住，改用 `git format-patch` + `git am` 同步同一提交；没有 reset 或覆盖远端已有未提交改动。
- 远端工作区仍保留原有 H2 配置、mesh、SonicMJ 迁移等未提交改动；本轮只叠加诊断提交。
- 远端 compile 通过：
  `./.venv/bin/python -m compileall -q gear_sonic/trl/callbacks/im_eval_callback.py gear_sonic/scripts/h2_eval_diagnostics.py`

### 旧 metrics 诊断
- 在 PKU 找到已有 compare8 metrics：
  - `logs_eval/20260617_precision_compare_h2/metrics_pr012000/metrics_eval.json`
  - `logs_eval/20260618_hybrid_compare_h2/metrics_hy002000/metrics_eval.json`
  - `logs_eval/20260615_tracking_first_compare_h2/metrics_tf2000/metrics_eval.json`
- 已运行：
  `./.venv/bin/python gear_sonic/scripts/h2_eval_diagnostics.py ...`
- 结果同步回本机：
  - `.workflow/artifacts/h2_eval_diagnostics_existing.md`
  - `.workflow/artifacts/h2_eval_diagnostics_existing.csv`

### 诊断结论
- `pr12000`：8/8 terminated，平均 progress `0.191803`，`mpjpe_g=0.070255`，是 precision 端点。
- `hy2000`：6/8 terminated，平均 progress `0.400499`，`mpjpe_g=0.125726`，两个成功 motion 的 global error 已明显偏大。
- `tf2000`：3/8 terminated，平均 progress `0.689094`，`mpjpe_g=0.151636`，是 survival/progress 端点但 global/foot/VR drift 更明显。

### 新 anchor logging eval 尝试
- 按用户要求只使用 `0-4` GPU 范围；实际尝试使用 GPU `0`，随后并行使用 GPU `0,1,2`，没有使用 `5,6,7`。
- 尝试用新 `im_eval_callback.py` 重跑 `pr12000/hy2000/tf2000`，数据目录：
  `/home/nvme02/GR00T/dataset/h2_v30_chest_soft_reverse/motion_lib_eval_compare8`
- 先后尝试：
  - 单卡 `pr12000`，保存 trace，超过 12 分钟无 metrics，手动停止。
  - 单卡 `pr12000`，不保存 trace，`algo.config.eval.num_eval_episodes=8`，300 秒 timeout，无 metrics。
  - 三组并行，GPU `0/1/2`，不保存 trace，`algo.config.eval.num_eval_episodes=8`，1200 秒 timeout，无 metrics。
- 三组日志均停在加载 8 motions 后，没有进入最终 `Success Rate`/metrics 输出；没有残留 eval 进程，GPU 已释放。

### 当前阻塞
- 新 anchor scalar/trace metrics 尚未生成。当前问题从“缺少 PKU metrics”变为“新 callback/当前 eval 路径在 PKU 上加载 motion 后无法在 20 分钟内完成”。
- 不建议继续启动 global-anchor reward/termination ablation；下一步应先单独 debug eval loop 性能或 callback 逻辑，确认为什么旧 compare8 eval 能产出 metrics，而当前新 callback 版本不能在合理时间结束。

## 2026-06-18T23:58:00+08:00 - PLAN.md 代码阶段 M0/M1/M2 logging 部分执行，因 compare8 metrics 缺失暂停

### 已读取
- `AGENTS.md`：确认中文回答、uv + `.venv`、不得修改 CUDA/驱动/系统 GPU 组件、不得修改 workflow 源码、代码相关回答需注明位置。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认 `plan_approved=true`，本轮从已确认 PLAN 的代码阶段继续，不重新生成计划。
- `README.md`、`docs/README.md`：上游文档仍偏 IsaacLab；本轮执行以 `AGENTS.md` 的 SonicMJ/mjlab + uv 约束为准。
- `H2_TRANSFER_EXPERIMENT_LOG.md`：确认 `pr12000`、`hy2000`、`tf2000` 固定 8 motion 汇总和 PKU eval/run 路径。
- 关键代码：
  - `gear_sonic/trl/callbacks/im_eval_callback.py`
  - `gear_sonic/eval_agent_trl.py`
  - `sonic_mj/mdp/rewards.py`
  - `sonic_mj/mdp/terminations.py`
  - `sonic_mj/env_cfg.py`
  - `sonic_mj/wrapper.py`

### M0 - 基线恢复与路径核对
- 本机可见 H2 早期本地日志：`logs_rl/TRL_H2_Track/.../sonic_h2_test-*`，但不包含本轮目标 `pr12000/hy2000/tf2000` 的 compare8 `metrics_eval.json`。
- `H2_TRANSFER_EXPERIMENT_LOG.md` 记录的关键 PKU eval 路径当前在本机不可访问：
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260617_precision_compare_h2/pr12000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260618_hybrid_compare_h2/hy2000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260615_tracking_first_compare_h2/tf2000/metrics_eval.json`
- 已生成 `.workflow/artifacts/h2_eval_diagnostics.md` 和 `.workflow/artifacts/h2_eval_diagnostics.csv`，其中明确记录上述输入缺失。

### M1 - checkpoint 诊断工具
- 新增 `gear_sonic/scripts/h2_eval_diagnostics.py`：
  - 读取一个或多个 `metrics_eval.json`。
  - 输出 checkpoint 汇总表和 per-motion 表。
  - 支持 `progress`、`terminated`、`mpjpe_g/mpjpe_l`、`foot/VR`，以及本轮新增的 global-anchor 诊断字段。
- 新增 `.workflow/artifacts/h2_checkpoint_diagnosis.md`：
  - 固化 `pr12000/hy2000/tf2000` 的已有固定 8 motion 汇总。
  - 记录当前判断：`pr12000` 是 precision 端点，`tf2000` 是 survival/global-drift 端点，`hy2000` 是未达目标的折中点。
  - 记录后续 eval 和离线诊断命令模板。

### M2 - global-anchor logging 第一层
- 修改 `gear_sonic/trl/callbacks/im_eval_callback.py`：
  - eval 时从 `env.motion_command` 采集 anchor 和 robot anchor 的位置/姿态。
  - 写入 per-motion 标量：
    - `anchor_xy_error_mean/max/final`
    - `anchor_z_error_mean`
    - `anchor_heading_error_mean/max/final`
    - `anchor_ori_error_mean`
  - 默认不保存逐步大数组；设置 `SONIC_SAVE_EVAL_TRACE=1` 时，保存逐步 trace 到 `eval/all_metrics_dict.anchor_error_traces`。
- 本轮没有加入 global XY / heading reward 或 termination ablation，因为缺少真实 per-motion/time-series metrics，继续改 reward 会违反“不盲目继续训练”的当前目标。

### 本轮命令与结果
- `uv run python --version`：通过，输出 `Python 3.10.20`。
- `uv run python -m compileall -q gear_sonic/trl/callbacks/im_eval_callback.py gear_sonic/scripts/h2_eval_diagnostics.py`：通过。
- `uv run python gear_sonic/scripts/h2_eval_diagnostics.py --case pr12000=... --case hy2000=... --case tf2000=... --output-md .workflow/artifacts/h2_eval_diagnostics.md --output-csv .workflow/artifacts/h2_eval_diagnostics.csv`：通过；输出文件记录三个输入路径均不存在。
- 合成 `metrics_eval.json` smoke：
  - 使用 `mktemp -d` 创建临时目录和临时 metrics JSON。
  - 运行 `gear_sonic/scripts/h2_eval_diagnostics.py` 输出 markdown/csv。
  - 结果通过；命令退出时临时目录已删除。
- `git status --short --branch`：当前分支 `cross_humanoid`；本轮新增/修改文件见最终交付清单；未回退用户已有改动。

### 阻塞与下一步
- 当前阻塞：缺少 `pr12000/hy2000/tf2000` 的真实 compare8 `metrics_eval.json` 或可访问 PKU eval 输出路径，无法完成 per-motion/time-series 诊断，也无法可信判断 global-anchor ablation 是否该启动。
- 建议下一步在可访问 PKU 路径的机器上运行：
  ```bash
  SONIC_SAVE_EVAL_TRACE=1 WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/eval_agent_trl.py \
    checkpoint=<CHECKPOINT> use_mjlab=True sim_type=mjlab headless=True \
    ++num_envs=8 ++output_dir=<EVAL_OUT>
  ```
- 拿到三个 `metrics_eval.json` 后运行：
  ```bash
  uv run python gear_sonic/scripts/h2_eval_diagnostics.py \
    --case pr12000=<PR12000_EVAL_OUT>/metrics_eval.json \
    --case hy2000=<HY2000_EVAL_OUT>/metrics_eval.json \
    --case tf2000=<TF2000_EVAL_OUT>/metrics_eval.json \
    --output-md .workflow/artifacts/h2_eval_diagnostics.md \
    --output-csv .workflow/artifacts/h2_eval_diagnostics.csv
  ```

## 2026-06-18T21:30:01+08:00 - H2 迁移诊断 / global-anchor / adapter 计划完成，等待 PLAN 确认

### 已读取
- `AGENTS.md`：确认中文回答、项目 SonicMJ 使用 uv、不使用 conda、不得修改 CUDA/显卡驱动/系统 GPU 组件、不得修改 workflow 源码、代码相关回答需注明位置。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：上一轮内容是 PKU H2 4卡训练/2卡验证计划，和本轮用户要求不一致，已改为 H2 迁移诊断与结构方案。
- `README.md`、`docs/README.md`：上游文档仍偏 IsaacLab；本项目执行以 `AGENTS.md` 中 SonicMJ/mjlab + uv 约束为准。
- `H2_TRANSFER_EXPERIMENT_LOG.md`：确认已有实验在 `pr12000`、`hy2000`、`tf2000` 等 checkpoint 上呈现 precision/progress trade-off。
- 关键代码：
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
  - `gear_sonic/trl/callbacks/im_eval_callback.py`
  - `gear_sonic/trl/modules/universal_token_modules.py`
  - `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml`
  - `gear_sonic/config/actor_critic/decoders/g1_kin_mf_mlp.yaml`
  - `gear_sonic/eval_agent_trl.py`
  - `sonic_mj/mdp/rewards.py`
  - `sonic_mj/mdp/terminations.py`
  - `sonic_mj/env_cfg.py`
  - `sonic_mj/wrapper.py`

### 关键判断
- `tf2000` 代表 survival/progress 较好但 global tracking 漂移明显：`success=0.625`、`progress=0.6891`、`mpjpe_g=0.1547`。
- `pr12000` 代表短时间 tracking 精度较好但跑不完整：`success=0`、`progress=0.1918`、`mpjpe_g=0.0703`。
- `hy2000` 是折中但仍未达到目标：`success=0.25`、`progress=0.4005`、`mpjpe_g=0.1338`。
- `sonic_mj/mdp/terminations.py` 的 `anchor_pos` 当前只检查 z 高度差，没有独立 root XY / heading drift termination；这支持做 global-anchor 最小实验。
- `sonic_mj/mdp/rewards.py` 虽有 global anchor position/orientation reward，但 local 5-point reward 在 anchor-local frame 中比较，可能掩盖全局漂移。
- `UniversalTokenModule` 已有多 decoder、active decoder、freeze encoder/decoder 机制；H2-specific decoder/adapter 应优先走配置和新增 head，避免全网络 PPO 破坏 G1 表征。

### 已完成
- 更新 `TASK.md`：替换旧 PKU 训练计划，记录本轮四个方向、硬约束、证据、计划产物和验收条件。
- 更新 `.workflow/artifacts/grounding.md`：记录项目现状、关键实验事实、代码证据、参考资料和风险。
- 更新 `.workflow/artifacts/paper_project_map.json`：结构化映射 SONIC 论文/参考实现/目标代码；说明本轮没有新增论文检索。
- 更新 `.workflow/artifacts/open_questions.json`：按 `{ "questions": [...] }` 记录 3 个非 blocking 问题。
- 更新 `PLAN.md`：包含 M0-M6 里程碑、完成标准、测试命令和 review 点，状态为等待用户确认。
- 更新 Codex 多角色 review：
  - `.workflow/artifacts/reviews/scope.md`
  - `.workflow/artifacts/reviews/milestones.md`
  - `.workflow/artifacts/reviews/architecture.md`
  - `.workflow/artifacts/reviews/testing.md`
  - `.workflow/artifacts/reviews/executor.md`
- 更新 `task.json`：写入本轮 request，标记 `plan_ready_waiting_user_confirmation`。

### 本轮命令与结果
- `sed -n ... /home/ykj/tool/WORKFLOW/skills/workflow/SKILL.md`：成功读取 workflow skill；未修改 workflow 源码。
- `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh resume --workspace "$PWD"`：成功，输出 `phase=resumed`、`plan_approved=false`。
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md README.md docs/README.md H2_TRANSFER_EXPERIMENT_LOG.md`：成功读取必需项目文档和 H2 实验记录。
- `jq . task.json`：成功读取旧任务摘要，确认需要替换为本轮 request。
- `rg ... gear_sonic sonic_mj` / `sed -n ...`：读取 H2 config、eval callback、Universal Token module、decoder config、SonicMJ reward/termination/wrapper。
- `git status --short --branch`：确认当前分支 `cross_humanoid`，已有 workflow 文档改动和未跟踪 `H2_TRANSFER_EXPERIMENT_LOG.md`、训练曲线、`wandb/`；本轮未回退任何已有内容。
- `date --iso-8601=seconds`：记录时间戳。

### 阻塞与下一步
- 当前没有 blocking open question。
- workflow 规则要求停在等待用户确认 `PLAN.md`。
- 用户确认后从 `PLAN.md` 的 M0/M1 开始：先固化 `pr12000/hy2000/tf2000` 的 run/eval 文件和 per-motion 诊断，再做 global-anchor logging/ablation；不直接启动长期训练。

## 2026-05-08T21:19:39+08:00 - workflow 恢复检查

### 已读取
- `AGENTS.md`：确认 SonicMJ 迁移约束；项目迁移环境使用 uv，不使用 conda；不得修改 CUDA/显卡驱动/系统级 GPU 环境；workflow 运行时必须先处理 `PERSON.md` open 条目。
- `PERSON.md`：没有实际 open 条目，只有模板。
- `TASK.md`：原状态中用户需求、验收条件、Python 环境均未补全。
- `PLAN.md`：尚未生成计划。
- `task.json`：`request_text=null`，`plan_approved=false`，`last_result=init`，环境字段为空。
- `.workflow/runtime/resume-prompt.txt`：内容与当前恢复指令一致。
- `.workflow/artifacts`：恢复前不存在 artifact 文件。
- `README.md` / `docs/README.md` / `pyproject.toml` / `gear_sonic/pyproject.toml`：确认根项目 `sonicmj` 使用 Python `>=3.10,<3.11`，根 `pyproject.toml` 有 uv 配置和 `uv.lock`；README 中原上游训练仍描述 Isaac Lab，但本项目 `AGENTS.md` 明确 SonicMJ 的 mjlab 迁移使用 uv。
- git 状态：分支 `SONICMJ...origin/SONICMJ`；已有未提交改动 `AGENTS.md`，新增 workflow 文件 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 判断
- 当前没有 `PERSON.md` open 条目需要处理。
- 由于 `request_text=null` 且 `PLAN.md` 未生成，不能直接执行代码迁移或测试；否则会绕过 workflow 的任务确认和计划阶段。
- Python 环境可按项目事实登记为 `uv` + `.venv` + `existing`，但还不能安装依赖或启动训练。
- 最合理下一步是阻塞等待用户明确本轮项目目标，或确认按 `AGENTS.md` 的 SonicMJ/mjlab 迁移顺序生成计划。

### 本轮命令与结果
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md`：成功读取 workflow 必读文档。
- `jq . task.json`：成功读取轻量运行摘要。
- `find .workflow ...`：恢复前仅有 `.workflow/runtime/resume-prompt.txt`。
- `git status --short --branch` / `git status --porcelain=v1`：确认当前工作区已有 workflow 初始化文件和 `AGENTS.md` 修改。
- `rg --files -g 'README*' -g 'pyproject.toml' -g 'uv.lock' ...`：定位必要 README 和环境配置。
- `sed -n ... README.md docs/README.md pyproject.toml gear_sonic/pyproject.toml`：确认根项目 uv 环境事实。
- `mkdir -p .workflow/artifacts`：创建 artifact 目录用于记录阻塞问题。

### blocked_reason
缺少可执行项目任务：`task.json.request_text` 为空，`TASK.md` 只有占位信息，`PLAN.md` 尚未生成。需要用户提供具体要推进的 SonicMJ/mjlab 迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成计划”。

## 2026-05-08T21:23:04+08:00 - grounding / plan / review 完成，等待 PLAN 确认

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复 workflow 状态，并确认本轮用户已明确要求继续 SonicMJ 到 mjlab 迁移。
- `README.md`、`docs/README.md`、`pyproject.toml`：确认项目环境事实为 uv + `.venv`，根项目依赖含 `mjlab==1.2.0`。
- `sonic_mj/assets.py`、`sonic_mj/env_cfg.py`、`sonic_mj/mdp/commands.py`、`sonic_mj/mdp/observations.py`、`sonic_mj/mdp/rewards.py`、`sonic_mj/mdp/terminations.py`、`sonic_mj/mdp/events.py`、`sonic_mj/wrapper.py`、`sonic_mj/train.py`：确认当前 mjlab 迁移骨架和待验证面。
- `gear_sonic/train_agent_trl.py`：确认 `use_mjlab=True` / `sim_type=mjlab` 已接入 `sonic_mj.train.create_mjlab_manager_env`。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`：确认第一阶段目标是 G1、teleop、SMPL 3 encoder。
- `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` 与 `gear_sonic/envs/manager_env/mdp/`：作为原 IsaacLab 语义参考。
- Zotero：搜索 `GEAR SONIC humanoid behavior foundation model`，找到 SONIC 论文 item `GT7EYT6G`，URL `https://arxiv.org/pdf/2511.07820`。

### 已完成
- 更新 `TASK.md`：补全用户需求、硬约束、当前代码事实、参考资料和验收条件。
- 写入 `.workflow/artifacts/grounding.md`：记录项目现状、关键证据、参考资料和风险。
- 写入 `.workflow/artifacts/paper_project_map.json`：结构化映射 SONIC 论文、原实现、目标代码和验证点。
- 写入 `.workflow/artifacts/open_questions.json`：按 `{ "questions": [...] }` 格式记录非 blocking 的计划确认问题；当前没有阻塞生成计划的问题。
- 生成 `PLAN.md`：包含 M0-M8 里程碑、完成标准、测试命令和 review 点。
- 写入 Codex 多角色 review：
  - `.workflow/artifacts/reviews/scope.md`
  - `.workflow/artifacts/reviews/milestones.md`
  - `.workflow/artifacts/reviews/architecture.md`
  - `.workflow/artifacts/reviews/testing.md`
  - `.workflow/artifacts/reviews/executor.md`
- 更新 `task.json`：写入本轮 request，标记 `plan_ready_waiting_user_confirmation`。

### 判断
- 没有 blocking open question；但 workflow 规则要求当前阶段停在等待用户确认 `PLAN.md`。
- 尚未进入代码修改或测试执行；本轮只改 workflow 文档和 artifacts。
- 后续执行必须使用 `uv run`，不得安装到 base，不得触碰 CUDA/显卡驱动/系统 GPU 组件。

### 本轮命令与结果
- `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh resume --workspace "$PWD"`：成功，输出 `phase=resumed`，`plan_approved=false`。
- `sed -n ...`：读取 workflow 必读文档、README、pyproject 和关键代码文件成功。
- `find .workflow -maxdepth 4 -type f`：确认已有 `.workflow/artifacts/open_questions.json` 和 runtime 文件。
- `git status --short`：确认已有未提交 workflow 初始化文件和 `AGENTS.md` 修改；本轮未回退任何已有改动。
- `zotero_search_items` / `zotero_item_fulltext`：找到并读取 SONIC 论文条目。
- `mkdir -p .workflow/artifacts/reviews`：创建 review artifact 目录。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后，从 `PLAN.md` 的 M0 开始执行环境确认、静态检查、asset/order 验证、reset/step smoke、obs shape 对齐和短训练 smoke。

## 2026-05-08T21:22:12+08:00 - 短生命周期 agent 恢复判断

### 已读取
- `AGENTS.md`：确认本项目 SonicMJ 迁移使用 uv + `.venv`，不得修改 CUDA/显卡驱动/系统级 GPU 组件，workflow 恢复必须先处理 `PERSON.md` open 条目。
- `PERSON.md`：无实际 open 条目，只有模板。
- `TASK.md`：仍只记录“恢复 workflow”占位需求，未包含可执行开发任务。
- `PLAN.md`：仍未生成计划。
- `PROGRESS.md`：上一轮已记录缺少任务目标并阻塞。
- `task.json`：`request_text=null`，`plan_approved=false`，环境为 `uv` + `.venv` + `existing`。
- `.workflow/artifacts/open_questions.json`：存在 `blocking: true` 的 `Q0001`，要求用户提供本轮任务或确认按 `AGENTS.md` 建议迁移顺序生成计划。
- `README.md` / `docs/README.md` / `pyproject.toml` / `.python-version`：确认上游 README 仍描述 Isaac Lab 训练，项目根配置为 `sonicmj`、Python 3.10、uv 环境；当前 SonicMJ 迁移约束以 `AGENTS.md` 为准。

### 判断
- 当前用户消息是恢复 workflow 的运行指令，没有回答 `Q0001`。
- `open_questions.json` 中已有 blocking 问题，按 workflow 规则必须停止等待用户补充，不能擅自生成计划、改代码或跑训练测试。
- 当前最合理下一步是保持阻塞状态，并要求用户提供具体迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成 PLAN.md”。

### 本轮命令与结果
- `sed -n '1,220p' /home/ykj/tool/WORKFLOW/skills/workflow/SKILL.md`：成功读取 workflow skill 规则；未修改 workflow 源码仓库。
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md`：成功读取 workflow 必读文档。
- `jq . task.json`：成功读取运行摘要，确认任务请求仍为空。
- `find .workflow -maxdepth 3 -type f -print | sort`：确认 artifact/runtime 文件存在。
- `git status --short --branch`：确认分支 `SONICMJ...origin/SONICMJ`，已有未提交 workflow 文件与 `AGENTS.md` 修改。
- `rg --files -g 'README*' -g 'docs/**' -g 'pyproject.toml' -g 'uv.lock' -g '.python-version'`：定位必要 README、docs 和环境配置。
- `sed -n ... .workflow/artifacts/open_questions.json README.md pyproject.toml .python-version docs/README.md`：确认阻塞问题和 uv/Python 环境事实。
- `date --iso-8601=seconds`：记录本轮时间戳。

### blocked_reason
`.workflow/artifacts/open_questions.json` 存在 blocking 问题 `Q0001`，且当前仍缺少可执行项目任务。需要用户提供本轮要推进的具体 SonicMJ/mjlab 迁移目标，或明确确认“按 AGENTS.md 建议迁移顺序继续生成 PLAN.md”。

## 2026-05-08T21:42:33+08:00 - PLAN.md 代码阶段 M0-M8 执行完成

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认 `plan_approved=true`，环境为 `uv` + `.venv` + `existing`。
- `PERSON.md`：没有实际 open 条目，只有模板；无需同步新增需求。
- `README.md`、`docs/README.md`、`pyproject.toml`：确认根项目使用 uv，`mjlab==1.2.0`，不使用 conda，不安装依赖到 base。
- `.workflow/artifacts/open_questions.json`：仅有非 blocking 问题；本轮按已确认 PLAN 执行。

### M0 - 基线恢复与环境确认
- `uv run python --version`：通过，输出 `Python 3.10.20`。
- `uv run python -c "import mjlab, torch, mujoco; ..."`：通过，输出 `ok mjlab unknown torch 2.11.0+cu130 mujoco 3.8.1`。
- `git status --short --branch`：分支 `SONICMJ...origin/SONICMJ`；已有 `M AGENTS.md` 和未跟踪 workflow 文档/目录。本轮未回退任何已有改动。
- 数据路径检查：仓库内存在 `data/motion_lib_bones_seed/robot_smoke`、`data/motion_lib_bones_seed/soma_uniform_smoke`、`data/smpl_filtered`，可用于 smoke。

### M1 - 静态对齐与导入健康检查
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`：通过。
- Hydra compose `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16` 并调用 `make_sonic_mj_env_cfg`：通过。
- 构造结果：`ManagerBasedRlEnvCfg`，actions `['joint_pos']`，commands `['motion']`，observations `['policy', 'critic', 'tokenizer']`，action dim `29`。
- 诊断脚本中第一次访问 `cfg.experiment_name` 时触发 `HydraConfig was not set`；这是非 Hydra 正式入口访问 `${hydra:runtime...}` 字段导致，调整脚本不解析该字段后通过，不需要改项目代码。

### M2 - G1 资产、顺序和动作闭环
- XML 静态核对命令解析 `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`：通过。
- 结果：XML actuator count `29`，`SONIC_G1_JOINT_NAMES` count `29`，body count `30`，actuator order 与 `sonic_mj/assets.py` canonical joint order 完全一致。
- 按 regex 展开 `SONIC_G1_ACTION_SCALE` 和 `SONIC_G1_DEFAULT_JOINT_POS` 后均为 29 维；action scale 全部为 `0.5`。
- reset smoke 中 `env.print_order_diagnostics()` 输出所有 order checks 为 True：robot joints/body、motion bodies、action joints、policy `joint_pos`/`joint_vel`/`actions` 顺序均匹配 SONIC MuJoCo order，action dim 为 29。

### M3 - MotionCommand reset/step smoke
- 命令：用 `sonic_release` + `robot_smoke` + `smpl_filtered` + `terrain_type=plane` 创建 `create_mjlab_manager_env(cfg, 'cpu')`，执行 reset 和零动作 step。
- 结果：通过。motion command 加载 210 个 robot motion，reset obs shape 为 `actor_obs (2, 930)`、`critic_obs (2, 1789)`、`tokenizer (2, 1767)`；step 返回 reward shape `(2,)`、done shape `(2,)`、info keys `['env_actions', 'episode', 'log', 'time_outs', 'to_log']`。

### M4 - Observation 与 3 encoder 对齐
- 命令：读取 `sonic_release` 的 `algo.config.actor.backbone.encoders`，与 mjlab tokenizer terms 对比。
- 结果：通过。G1、teleop、SMPL 所有输入项均存在，无 missing：
  `command_multi_future_nonflat`、`motion_anchor_ori_b_mf_nonflat`、`command_multi_future_lower_body`、`vr_3point_local_target`、`vr_3point_local_orn_target`、`motion_anchor_ori_b`、`smpl_joints_multi_future_local_nonflat`、`smpl_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_smpl`。

### M5 - Rewards / Terminations / Events / Curriculum 训练必要项
- reset/step smoke 中确认 event manager active terms：reset `reset_scene_to_default`；startup `add_joint_default_pos`、`base_com`、`randomize_rigid_body_mass`、`physics_material`；interval `push_robot`。
- reward terms active：`tracking_anchor_pos`、`tracking_anchor_ori`、`tracking_relative_body_pos`、`tracking_relative_body_ori`、`tracking_body_linvel`、`tracking_body_angvel`、`tracking_vr_5point_local`、`action_rate_l2`、`joint_limit`、`feet_acc`。
- termination terms active：`time_out`、`anchor_pos`、`anchor_ori_full`、`ee_body_pos`、`foot_pos_xyz`。
- 20 步 CPU rollout（4 env，零动作）：通过；reward finite，last obs shapes `actor_obs (4, 930)`、`critic_obs (4, 1789)`、`tokenizer (4, 1767)`，done_count `5`。

### M6 - TRL 小规模训练 smoke
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16 headless=True ++algo.config.num_learning_iterations=10 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：通过，完成 learning iteration 1 到 10。
- 关键输出：mjlab backend 使用 `cuda:0` 现有设备；初始化 G1、teleop、SMPL 三个 encoder；PPO loop 正常采样和更新；第 10 iteration total timesteps `3840`，mean reward `-16.74488`，日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_214120`。
- 未修改 CUDA、显卡驱动或系统 GPU 组件。

### M7 - SOMA / sonic_bones_seed 第二阶段兼容
- 查找配置：存在 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_bones_seed.yaml`，SOMA 数据 smoke 路径存在 `data/motion_lib_bones_seed/soma_uniform_smoke`。
- compose 检查：`sonic_bones_seed` + `robot_smoke` + `soma_uniform_smoke` + `smpl_filtered` 通过；tokenizer 自动包含 `soma_joints_multi_future_local_nonflat`、`soma_root_ori_b_multi_future`、`joint_pos_multi_future_wrist_for_soma`，所有 encoder input 无 missing，action dim 29。
- CPU reset/step smoke：通过；reset obs shapes `actor_obs (2, 930)`、`critic_obs (2, 1789)`、`tokenizer (2, 2668)`；step reward finite。

### M8 - 文档、清理和交付状态
- 本轮未创建临时测试脚本，所有测试均为 inline `uv run python - <<'PY' ... PY`。
- 本轮未修改 Python 源码；只更新 workflow 文档/摘要。
- 训练 smoke 产生日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_214120`，该目录未出现在 `git status --short` 中，应为 ignored 或已按项目规则排除。
- 最终 `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 未验证项
- 未用默认完整数据路径 `data/motion_lib_bones_seed/robot_filtered`、`data/bones_seed_smpl`、`data/motion_lib_bones_seed/soma_filtered` 跑训练；本轮使用仓库 smoke 数据以降低成本。
- 未验证 rough/trimesh terrain 训练；smoke 使用 `manager_env.config.terrain_type=plane`。
- 未验证多 GPU、多进程、大规模 `num_envs=4096` 训练、checkpoint resume/finetune、ONNX export、render/eval、W&B 在线曲线。
- 未修改或重写 `UniversalTokenModule`、PPO trainer、checkpoint key。

## 2026-05-08T23:06:32+08:00 - PLAN.md 代码阶段补充执行：默认训练尝试与非 smoke 正常训练

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：确认用户已要求进入 PLAN.md 代码阶段；`PERSON.md` 没有实际 open 条目。
- `.workflow/artifacts/open_questions.json`：仅有非 blocking 的计划确认问题；本轮用户已确认继续执行。
- `README.md`、`docs/README.md`、`pyproject.toml`：再次确认根项目使用 uv + `.venv`，依赖包含 `mjlab==1.2.0`；未安装任何依赖。

### M0 / M1 复核
- `uv run python --version`：通过，输出 `Python 3.10.20`。
- `uv run python -c "import mjlab, torch, mujoco; ..."`：通过，输出 `ok mjlab unknown torch 2.11.0+cu130 mujoco 3.8.1`。
- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv`：GPU 为 `NVIDIA GeForce RTX 5070 Ti Laptop GPU`，总显存 `12227 MiB`，运行前空闲约 `9539 MiB`。
- 数据路径复核：默认完整目录 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl` 不存在；`data/motion_lib_bones_seed/robot_medium`、`data/motion_lib_bones_seed/soma_uniform_medium`、`data/smpl_filtered`、`/home/ykj/Downloads/dataset/bones-seed`、`/home/ykj/commonly_used/body_models` 存在。
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`：通过。

### M7 默认正常训练路径尝试
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab headless=True`
- 结果：进入训练入口并创建日志目录 `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_230045`，但默认不覆盖 `terrain_type` 时卡在 mjlab rough terrain 生成阶段；运行约 3 分钟后手动 `SIGINT` 中断以避免无界等待。
- 证据：中断堆栈显示 `Terrain generation took 109.7336 seconds`，卡点位于 `mjlab/terrains/terrain_generator.py` 的 `TerrainGenerator.compile`；不是 CUDA/驱动修改问题，也没有进入 PPO iteration。
- 结论：默认完整训练路径在本机当前配置下未完成，主要限制为默认 rough terrain 生成耗时不可接受；此外默认完整数据目录 `robot_filtered` / `bones_seed_smpl` 也不存在，后续按 PLAN 使用最大可用 medium 数据和 plane terrain 验证正常训练代码路径。

### M7 非 smoke 正常训练
- 命令：
  `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=64 headless=True ++algo.config.num_learning_iterations=100 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_medium ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：通过，完成 100 个 learning iterations，真实 PPO 采样和更新正常执行。
- 训练事实：`num_envs=64`，加载 `data/motion_lib_bones_seed/robot_medium` 共 970 个 motion 文件，启动时采样 64 个 motions；使用 `cuda:0`，未修改 CUDA/显卡驱动/系统 GPU 组件。
- 日志与 checkpoint：`logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260508_230400`；保存 `last.pt`，大小约 `432M`；`config.yaml` 已生成。
- 第 100 iteration：total episodes `6400`，total timesteps `153600`，mean rewards `-16.70631`，mean length `8.99000`，computation `1270 steps/s`，total time `122.01s`，reward finite，未出现 NaN/OOM。

### 清理与状态
- 本轮没有创建临时测试脚本。
- 没有修改 Python 源码；只更新 workflow 文档和 `task.json` 摘要。
- `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。
- 检查训练进程：无遗留本轮训练进程。

### 最终未验证项
- 默认不覆盖参数的 rough terrain 正常训练未完成；本机运行中 terrain 生成 109 秒后仍在 `TerrainGenerator.compile`，已记录为当前默认配置耗时限制。
- 默认完整数据目录 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl` 不存在，因此未用官方完整过滤数据跑通默认训练。
- 未验证 `num_envs=4096`、多 GPU、多进程、checkpoint resume/finetune、ONNX export、render/eval、W&B 在线曲线。
- 未修改或重写 `UniversalTokenModule`、PPO trainer、checkpoint key。

## 2026-05-08T23:09:14+08:00 - 补充执行：SOMA / sonic_bones_seed 训练入口

### 命令
- `WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_bones_seed use_mjlab=True sim_type=mjlab num_envs=32 headless=True ++algo.config.num_learning_iterations=20 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_medium ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_uniform_medium ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`

### 结果
- 通过，完成 20 个 learning iterations，真实 PPO 采样和更新正常执行。
- 训练事实：`num_envs=32`，加载 `data/motion_lib_bones_seed/robot_medium` 共 970 个 robot motion 文件，SOMA tokenizer terms 已激活，使用 `cuda:0`，未修改 CUDA/显卡驱动/系统 GPU 组件。
- 日志目录：`logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_bones_seed_test-20260508_230821`。
- 第 20 iteration：total episodes `640`，total timesteps `15360`，mean rewards `-20.23965`，mean length `10.17000`，computation `508 steps/s`，total time `30.84s`，reward finite，未出现 NaN/OOM。
- 该 20-iteration 短训练未生成 `last.pt`；目录中存在 `config.yaml` 和 `meta.yaml`，未达到当前保存 checkpoint 的间隔。
- 检查训练进程：无遗留 `gear_sonic/train_agent_trl.py` / `accelerate launch` 训练进程。

## 2026-05-11T12:05:55+08:00 - H20 服务器训练兼容化 grounding / plan / review 完成

### 已读取
- `AGENTS.md`：确认本项目 SonicMJ 的 mjlab 迁移使用 uv，不使用 conda；不得修改 CUDA、显卡驱动、Isaac Sim 底层安装或系统级 GPU 环境；外部 `mjlab`、`InstinctMJ`、原始仓库只读参考。
- `PERSON.md`：没有实际 open 条目。
- `TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复历史迁移验证上下文，并将本轮目标重定向为 H20 服务器训练环境兼容与 smoke test。
- `README.md`、`docs/README.md`、`pyproject.toml`、`uv.lock`：确认根项目使用 uv；当前锁文件包含 `torch 2.11.0`、CUDA 13 wheel、`mjlab==1.2.0`、`mujoco-warp>=3.8.0,<3.9`。
- `gear_sonic/train_agent_trl.py`、`gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`、`sonic_mj/`：确认 mjlab backend 入口、3 encoder 训练配置和历史可训练代码路径。

### 已完成
- 更新 `TASK.md`：记录服务器事实 `nvcc 11.6`、Driver `570.124.06`、reported CUDA `12.8`、GPU `NVIDIA H20`；保留不修改 CUDA/驱动/系统 GPU 组件、只用项目 uv 环境的硬约束；补充服务器 smoke 验收条件。
- 写入 `.workflow/artifacts/grounding.md`：记录当前锁文件 CUDA 13 依赖与 H20 服务器 Driver 570/CUDA 12.8 的兼容风险，以及历史本机训练验证不能替代服务器验证。
- 写入 `.workflow/artifacts/paper_project_map.json`：保留 SONIC 论文、原实现、目标 `sonic_mj` 代码和服务器验证点之间的结构化映射。
- 写入 `.workflow/artifacts/open_questions.json`：当前没有 blocking 问题；记录等待确认 `PLAN.md`、是否允许服务器上重建项目 `.venv`、smoke 数据路径偏好的非阻塞问题。
- 生成 `PLAN.md`：M0-M8 覆盖服务器基线、当前 lock 兼容性判定、必要时项目级 CUDA 12.x 依赖修复、compile/compose、reset/step、obs/order、PPO smoke、默认数据加载检查和文档清理。
- 写入 Codex 多角色 review：`scope.md`、`milestones.md`、`architecture.md`、`testing.md`、`executor.md`。

### 判断
- 没有阻塞生成计划的问题；服务器信息足够制定执行计划。
- 计划阶段不应直接修改依赖或运行训练；按用户要求停在等待 `PLAN.md` 确认。
- 确认后应先在服务器执行只读 M0/M1；只有现有 CUDA 13 lock 在 Driver 570/CUDA 12.8 上失败时，才通过项目级 uv 依赖和 lock 文件切换到 CUDA 12.x 兼容组合。

### 本轮命令与结果
- `bash /home/ykj/tool/WORKFLOW/skills/workflow/scripts/run-workflow.sh run --workspace "$PWD" --request ...`：workflow 子进程生成本轮 `TASK.md`、`PLAN.md` 和 artifacts；子进程完成后未正常退出，已终止该已完成的 workflow 子进程，未修改 workflow 源码。
- `sed -n ... TASK.md PLAN.md PROGRESS.md .workflow/artifacts/grounding.md`：核对本轮文档内容。
- `python -m json.tool .workflow/artifacts/open_questions.json`：通过，JSON 格式有效。
- `python -m json.tool .workflow/artifacts/paper_project_map.json`：通过，JSON 格式有效。
- `git status --short --branch`：当前分支 `SONICMJ...origin/SONICMJ`；已有 `M AGENTS.md`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后按 `PLAN.md` 从 M0 开始执行；如果用户允许，可在服务器项目目录内用 uv 重建 `.venv`，但不得修改 base、系统 CUDA、驱动或 GPU 底层组件。

## 2026-05-11T12:20:00+08:00 - 更新 PLAN：加入预备服务器 GPU 前置测试阶段

### 用户新增要求
- 在 `PLAN.md` 中加入一台可修改的预备服务器，用于完成真正使用 GPU 前的测试。
- 预备服务器登录入口为 `ssh -p 37716 root@region-42.seetacloud.com`；认证信息由用户提供，不写入项目文档。
- 预备服务器可以随便修改；最好把它配置成最终 H20 训练服务器将使用的项目依赖配置。
- 最终 H20 服务器仍为 Driver `570.124.06`、reported CUDA `12.8`、`nvcc 11.6`、GPU `NVIDIA H20`；等用户开启 GPU 后再运行需要 GPU 的 reset/step 与训练 smoke。

### 已完成
- 更新 `PLAN.md` 的状态说明和“执行环境分层”章节。
- 将 M0-M3 调整为可先在预备服务器执行的 GPU 前置测试：环境基线、依赖解析、uv lock/env 修复、import、Hydra compose、静态检查。
- 将 M4-M7 明确标记为最终 H20 GPU 开启后执行：mjlab GPU reset/step、obs tensor 检查、PPO 训练 smoke、默认数据真实 env reset。
- 更新计划确认项：预备服务器先行；最终 H20 GPU 开启后再继续 GPU 阶段。

### 当前状态
- 尚未登录预备服务器，也未修改远端环境。
- 尚未修改 `pyproject.toml` / `uv.lock`。
- 下一步仍需用户确认按更新后的 `PLAN.md` 执行。

## 2026-05-11T14:39:24+08:00 - 预备服务器 GPU 前置测试完成

### 预备服务器基线
- 登录入口：`ssh -p 37716 root@region-42.seetacloud.com`；认证信息未写入项目文件。
- 系统：Ubuntu 20.04.4，host `autodl-container-e9b742b627-c588c445`。
- GPU 状态：`nvidia-smi` 返回 `No devices were found`，本阶段按 GPU 前置测试处理，不作为失败。
- CUDA 工具链：`nvcc --version` 为 CUDA `11.3, V11.3.109`；未修改 CUDA、显卡驱动或系统 GPU 组件。
- Python/uv：远端用户级安装 `uv 0.11.13`；用 uv 安装 CPython `3.10.20`。
- 磁盘：`/root/autodl-tmp` 50G；项目部署到 `/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`。

### 依赖修复
- 原因：根 `uv.lock` 原先解析到 `torch 2.11.0` 和 CUDA 13 wheel；最终 H20 服务器 Driver `570.124.06` / reported CUDA `12.8` 不适合作为 CUDA 13 wheel 的目标运行环境。
- 修改 `pyproject.toml`：在根项目依赖中显式加入 `torch==2.7.1`。
- 重新运行 `uv lock`：CUDA 13 相关包被移除，锁定到 `torch 2.7.1` / `torchvision 0.22.1` / `triton 3.3.1` 和 NVIDIA CUDA 12.6 wheel 组合。
- 兼容判断：最终 H20 的 Driver 570 可运行 CUDA 12.6 wheel；`nvcc 11.6` 只是本机编译工具链版本，本轮不依赖系统 nvcc 编译 PyTorch。

### 执行过的命令与结果
- 远端只读基线：`nvcc --version`、`nvidia-smi`、`python3 --version`、`df -h`、`which git rsync uv`。
  - 结果：无 GPU；系统 Python 3.8.10；远端原先无 uv；`git`/`rsync` 可用。
- 远端安装 uv 与 Python：
  - `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - `uv python install 3.10`
  - 结果：`uv 0.11.13`、CPython `3.10.20` 可用。
- 项目同步：
  - `rsync -az --delete` 同步代码、配置和小资产到远端；排除 `.git/`、`.venv*/`、`logs_rl/`、`data/`、`sonic_release/` 等大目录。
  - 结果：远端项目目录创建成功；大数据未同步，符合 GPU 前置测试范围。
- 依赖解析：
  - 初始测试 `torch==2.7.1` + PyTorch cu128 可解析，但官方 cu128 源下载在预备服务器和本机都过慢。
  - 改用 PyPI `torch==2.7.1` 的 CUDA 12.6 wheel 组合后，本机 `uv sync --frozen` 成功。
  - 远端直连 wheel 下载仍慢；为完成前置验证，将本机已完成的 `.venv/lib/python3.10/site-packages` 同步到远端，并保留远端自己的 `.venv/bin/python` / `pyvenv.cfg`，随后修正 editable path 到远端项目路径。
- 本机验证：
  - `uv run python - <<'PY' ... import torch/mujoco/mujoco_warp/mjlab ... PY`
  - `uv run python -m compileall -q sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：`torch 2.7.1+cu126`，`torch.version.cuda 12.6`，`torch.cuda.is_available() False`；`mujoco 3.8.1`、`mujoco_warp 3.8.0`、`mjlab` 导入成功；compileall 通过。
- 预备服务器验证：
  - `.venv/bin/python - <<'PY' ... import torch/mujoco/mujoco_warp/mjlab ... PY`
  - `.venv/bin/python -m compileall -q sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：`python 3.10.20`，`torch 2.7.1+cu126`，`torch.version.cuda 12.6`，`torch.cuda.is_available() False`；`mujoco 3.8.1`、`mujoco_warp 3.8.0`、`mjlab` 导入成功；compileall 通过。
- 预备服务器 Hydra / env cfg 检查：
  - compose `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab checkpoint=null num_envs=2 headless=True manager_env.config.terrain_type=plane`。
  - 调用 `make_sonic_mj_env_cfg(cfg)`。
  - 结果：`num_envs 2`；actions `['joint_pos']`；observations `['policy', 'critic', 'tokenizer']`；commands `['motion']`；rewards `10`；terminations `['time_out', 'anchor_pos', 'anchor_ori_full', 'ee_body_pos', 'foot_pos_xyz']`；terrain `plane`。

## 2026-06-18T21:28:23+08:00 - 短生命周期 agent 恢复判断：PKU H2 计划待确认

### 已读取
- `AGENTS.md`：确认当前项目内 workflow 恢复规则、SonicMJ/mjlab 迁移约束、项目使用 uv、不使用 conda、不得修改 CUDA/显卡驱动/系统 GPU 组件。
- `PERSON.md`：无实际 `open` 条目，只有模板；当前不需要先处理用户补充项。
- `TASK.md`：当前目标为 PKU H2 warm-start 训练/验证计划阶段；训练 GPU 固定 `0,1,2,3`，验证 GPU 固定 `4,5`，禁止使用 `6,7`；当前阶段要求停在等待用户确认 `PLAN.md`。
- `PLAN.md`：状态为“等待用户确认”；确认前不启动 PKU 训练或验证。
- `task.json`：`plan_approved=false`，`last_result=plan_ready`，摘要显示已完成 PKU H2 4卡训练/2卡验证 grounding、计划、命令手册和多角色 review，当前停在等待用户确认。
- `.workflow/artifacts/open_questions.json`：有 3 个非 blocking open question，分别关于 W&B online/offline、H2 filtered PKL 是否作为首轮正式训练数据、正式训练是否接受先固定 plane terrain；没有 blocking 问题。
- `.workflow/artifacts/grounding.md` 与 `.workflow/artifacts/reviews/*.md`：确认本阶段范围是计划与命令手册，不启动训练、不改训练代码；executor review 明确“用户确认 `PLAN.md` 前不启动训练”。
- `PKU_H2_TRAIN_VALIDATE_PLAN.md`：已包含 preflight、数据准备、4 GPU warm-start 训练、2 GPU eval、subset training validation、resume、stop 和 reporting checklist。
- `README.md`、`docs/README.md`、`pyproject.toml`：确认项目根环境为 Python `>=3.10,<3.11`，依赖由 uv 管理，包含 `mjlab==1.2.0` 和 `torch==2.7.1`。

### git 状态
- 当前分支：`cross_humanoid`。
- 当前工作区已有多项未提交 workflow 文档/artifact 改动，以及未跟踪文件 `H2_TRANSFER_EXPERIMENT_LOG.md`、`training_curves_szsevsw9.png`、`training_curves_szsevsw9.svg`、`wandb/`。
- 本轮未回退、删除或覆盖这些已有改动。

### 判断
- 当前没有 `PERSON.md` open 条目，也没有 blocking open question。
- 但 `TASK.md`、`PLAN.md`、`task.json` 和 review artifacts 一致表明：当前阶段已经完成计划与 review，下一步需要用户确认 `PLAN.md` 后才能启动 PKU preflight、训练或验证。
- 用户本轮要求是恢复 workflow 并执行最合理下一步；在 `plan_approved=false` 且计划明确要求确认前不训练的情况下，最合理下一步是记录恢复判断并停止等待确认，而不是擅自登录 PKU、启动训练、转换数据或修改代码。

### 本轮命令与结果
- `sed -n '1,240p' /home/ykj/tool/WORKFLOW/skills/workflow/SKILL.md`：成功读取 workflow skill 规则；未修改 workflow 源码仓库。
- `sed -n ... AGENTS.md PERSON.md TASK.md PLAN.md PROGRESS.md task.json`：成功读取 workflow 必读文档和运行摘要。
- `find .workflow -maxdepth 3 -type f | sort`：确认 artifacts 与 runtime 文件存在。
- `git status --short --branch` / `git status --porcelain=v1` / `git diff --stat`：确认当前分支、已有未提交改动和未跟踪文件。
- `sed -n ... .workflow/artifacts/open_questions.json .workflow/artifacts/grounding.md .workflow/artifacts/reviews/*.md`：确认 open questions 非阻塞，但计划阶段仍需用户确认。
- `sed -n ... PKU_H2_TRAIN_VALIDATE_PLAN.md README.md docs/README.md pyproject.toml`：确认命令手册、项目 README/docs 和 uv 环境配置。
- `date --iso-8601=seconds`：记录本轮时间戳。

### blocked_reason
当前 workflow 停在计划确认点：`PLAN.md` 状态为等待用户确认，`task.json.plan_approved=false`，且 review 明确“用户确认 `PLAN.md` 前不启动训练”。需要用户明确确认按当前 `PLAN.md` 执行，或提供要修改的计划项；确认后才能进入 PKU preflight、4 卡训练 smoke、2 卡 eval smoke 和后续正式训练。

### 当前状态与未验证项
- M0-M3 预备服务器 GPU 前置测试完成。
- 没有创建临时测试脚本。
- 没有运行需要 GPU 的 env reset/step 或 PPO 训练；预备服务器没有可见 GPU。
- 未同步完整数据目录 `data/` 和 `sonic_release/`，因此未做默认 motion 数据加载或 checkpoint 相关检查。
- 最终 H20 GPU 开启后，需要继续 M4-M7：`torch.cuda` 初始化、mjlab GPU reset/step、obs/order 诊断、`num_envs=16` / `num_learning_iterations=10` PPO smoke。
- 检查遗留进程：无 `gear_sonic/train_agent_trl.py`、`accelerate launch`、`uv sync` 或 `rsync` 遗留进程。
- `git status --short --branch`：`## SONICMJ...origin/SONICMJ`，`M AGENTS.md`，`M pyproject.toml`，`M uv.lock`，未跟踪 `.workflow/`、`PERSON.md`、`PLAN.md`、`PROGRESS.md`、`TASK.md`、`task.json`。

## 2026-05-11T15:03:52+08:00 - AutoDL H20 GPU 阶段测试完成

### 服务器与数据放置
- 服务器入口：`ssh -p 37716 root@region-42.seetacloud.com`；认证信息未写入项目文件。
- 项目目录：`/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`，数据和项目均放在 AutoDL 数据盘 `/root/autodl-tmp` 下，未放到系统盘 `/root`。
- GPU：2 张 `NVIDIA H20`，每张约 `97871 MiB`；`nvidia-smi` 显示 Driver `580.105.08`、reported CUDA `13.0`。
- `nvcc`：该 AutoDL 服务器上未找到 `nvcc`；本轮未安装或修改 CUDA、显卡驱动、系统级 GPU 组件。
- 项目环境：远端 `.venv/bin/python` 为 Python `3.10.20`；`torch 2.7.1+cu126`、`torch.version.cuda 12.6`、`torch.cuda.is_available() True`、`torch.cuda.device_count() 2`。

### 已同步数据
- 同步 `gear_sonic/data/assets/` 到远端，修复 G1 MuJoCo XML 缺失问题。
- 同步 `data/motion_lib_bones_seed/robot_smoke` 和 `data/motion_lib_bones_seed/robot_medium`。
- 从本地按 `robot_smoke/robot_medium` 引用生成临时 SMPL 文件列表并同步匹配文件到远端 `data/smpl_filtered/`；匹配 `874` 个文件，远端大小约 `949M`。临时文件 `/tmp/sonicmj_smpl_subset.txt` 已删除。
- 同步默认 robot 数据 `data/motion_lib_bones_seed/robot_filtered` 到远端，远端大小约 `7.9G`，训练时加载 `129785` 个 robot motion 文件。
- 本地和远端 `data/bones_seed_smpl` 均为空或缺失，因此完整默认 SMPL 数据语义尚未验证。

### GPU reset/step 与顺序诊断
- 命令要点：
  - `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab checkpoint=null num_envs=2 headless=True`
  - `++algo.config.num_learning_iterations=1 ++algo.config.num_steps_per_env=2`
  - `++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke`
  - `++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy`
  - `manager_env.config.terrain_type=plane`
  - `create_mjlab_manager_env(cfg, "cuda:0")`
- 结果：
  - 加载 `210` 个 `robot_smoke` motions。
  - `actor_obs (2, 930)`，`critic_obs (2, 1789)`，`actions_dim 29`。
  - joint/body/motion/action/policy order 诊断全部通过；motion dof mapping 为 identity；action dim 为 29。
  - 单步 step 成功，reward/done shape 为 `(2,)`，reward finite，mean reward `-0.47829321026802063`。

### 单卡 PPO smoke
- 首次命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m accelerate.commands.launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=2 headless=True ++algo.config.num_learning_iterations=1 ++algo.config.num_steps_per_env=2 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：
  - 训练进程可启动并完成 1 iteration，但出现配置错误：`batch_size must be a multiple of num_mini_batches, inexact division: 2 / 4 = 0.5`。
  - 判定为 smoke 参数太小导致的 batch 配置问题，不是 GPU/环境不可用。
- 修正命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m accelerate.commands.launch --num_processes=1 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 修正结果：
  - 退出码 `0`，完成 learning iteration `1` 和 `2`。
  - 初始化 `g1` / `teleop` / `smpl` encoders 和 `g1_dyn` / `g1_kin` decoders。
  - total episodes `8`，total timesteps `32`，未出现 OOM/NaN。

### 双卡 PPO smoke
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered manager_env.config.terrain_type=plane`
- 结果：
  - 退出码 `0`，Accelerate 进入 multi-GPU 模式。
  - rank 0 使用 `cuda:0`，rank 1 使用 `cuda:1`；两个 rank 分别完成 env 创建、采样和训练。
  - 完成 learning iteration `1` 和 `2`；total episodes `16`，total timesteps `64`，mean rewards `-6.04001`。
  - 未出现 OOM/NaN。
  - 退出时有 PyTorch distributed 的非致命清理警告：`destroy_process_group() was not called before program exit`。

### 默认 robot 数据双卡训练
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=4 headless=True ++algo.config.num_learning_iterations=2 ++algo.config.num_steps_per_env=4 manager_env.config.terrain_type=plane`
- 说明：
  - 未覆盖 `motion_file`，因此使用 `sonic_release.yaml` 默认 robot motion 路径 `data/motion_lib_bones_seed/robot_filtered`。
  - 未覆盖 `smpl_motion_file`，因此尝试使用默认 `data/bones_seed_smpl`；该目录缺失/为空，当前 motion lib 对缺失 SMPL 路径会容忍并使用 `None` 数据项。
- 结果：
  - 退出码 `0`，两个 rank 均加载 `129785` 个 robot motion 文件。
  - 完成 learning iteration `1` 和 `2`；total episodes `16`，total timesteps `64`。
  - mean reward：iteration 1 为 `-2.69212`，iteration 2 为 `-12.68186`。
  - 未出现 OOM/NaN。
  - 同样存在非致命 `destroy_process_group()` 清理警告。

### 清理与当前状态
- 远端检查命令：
  - `pgrep -af "gear_sonic/train_agent_trl.py|accelerate.commands.launch|accelerate launch|rsync|uv sync" || true; nvidia-smi`
- 结果：
  - 无遗留训练、rsync 或 uv sync 进程；`pgrep` 只匹配到本次检查命令自身。
  - `nvidia-smi` 显示两张 H20 显存占用均为 `0MiB`，无运行中的 GPU 进程。

### 结论与未验证项
- 已通过：服务器 GPU import、mjlab reset/step、order diagnostics、单卡 PPO smoke、双卡 PPO smoke、默认 `robot_filtered` 数据的双卡短训练。
- 已证明：当前项目依赖组合 `torch 2.7.1+cu126` 可在该 AutoDL H20 GPU 服务器上运行，不需要修改 CUDA/驱动。
- 未完整验证：`sonic_release.yaml` 的完整默认 SMPL 数据路径 `data/bones_seed_smpl`，因为本地和远端该目录为空或不存在。
- 未验证：默认大规模 `num_envs=4096`、长时间训练、checkpoint resume/finetune、rough/trimesh terrain、render/eval、ONNX export、W&B 在线记录。

## 2026-05-11T15:32:27+08:00 - 补齐 AutoDL `data/bones_seed_smpl` 并完成默认数据双卡训练

### 数据补齐
- 本机检查：
  - `/home/ykj/Downloads/dataset/bones-seed` 总大小约 `650G`，主要是官方 Bones-SEED 原始/衍生数据，不直接整目录同步。
  - 本机 `data/smpl_filtered` 为可用于默认 SMPL motion 的 `.pkl` 目录，大小约 `31G`，文件数 `131455`。
  - 本机 `data/motion_lib_bones_seed/robot_filtered` 的 motion stem 数为 `129785`；`data/smpl_filtered` 的非 metadata stem 数为 `131454`。
  - 对齐检查显示 `robot_filtered` 有 `7` 个 motion stem 在本机 `data/smpl_filtered` 中不存在：`kneeling_loop_002__A098_M`、`neutral_button press_001__A543`、`neutral_button press_001__A543_M`、`neutral_button press_001__A544`、`neutral_button press_001__A544_M`、`neutral_button press_001__A545`、`neutral_button press_001__A545_M`。
  - 在 `/home/ykj/Downloads/dataset/bones-seed` 中也未找到这 7 个同名 `.pkl`，因此不是同步遗漏；当前 `MotionLibBase` 对缺失 SMPL 文件会将对应条目置为 `None`。
- 远端整理：
  - 远端项目仍位于 `/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl`。
  - 将此前由本轮同步的远端小样本 `data/smpl_filtered` 迁移为 `data/bones_seed_smpl`，再把本机完整 `data/smpl_filtered/` 增量同步到远端默认路径。
  - 为兼容此前 smoke 命令，远端保留 `data/smpl_filtered -> bones_seed_smpl` 符号链接。
  - 同步命令：`rsync -az --info=progress2 --partial --inplace -e 'ssh -o UserKnownHostsFile=/tmp/sonicmj_known_hosts -o StrictHostKeyChecking=no -p 37716' data/smpl_filtered/ root@region-42.seetacloud.com:/root/autodl-tmp/sonicmj-work/GR00T-WholeBodyControl/data/bones_seed_smpl/`
- 远端同步结果：
  - `data/bones_seed_smpl` 大小约 `31G`，`.pkl` 文件数 `131455`。
  - `data/smpl_filtered` 是指向 `bones_seed_smpl` 的符号链接。
  - `/root/autodl-tmp` 数据盘剩余约 `2.0G`，使用率 `97%`。该状态能完成小规模 smoke，但正式训练前建议清理不需要的数据或换更大数据盘。

### 完整默认路径双卡训练
- 命令：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab num_envs=16 headless=True ++algo.config.num_learning_iterations=10 manager_env.config.terrain_type=plane`
- 关键点：
  - 未覆盖 `motion_file`，使用 `sonic_release.yaml` 默认 `data/motion_lib_bones_seed/robot_filtered`。
  - 未覆盖 `smpl_motion_file`，使用 `sonic_release.yaml` 默认 `data/bones_seed_smpl`。
  - 双卡多进程：rank 0 使用 `cuda:0`，rank 1 使用 `cuda:1`。
  - 每个 rank 创建 `16` 个 mjlab env。
- 训练结果：
  - 退出码 `0`，完成 learning iteration `1` 到 `10`。
  - 两个 rank 均加载 `129785` 个 robot motion 文件，并从默认完整 SMPL 路径读取对应 SMPL 数据。
  - 初始化 `g1`、`teleop`、`smpl` 三个 encoder；tokenizer 观测中包含 `smpl_joints_multi_future_local_nonflat (10, 72)`、`smpl_root_ori_b_multi_future (10, 6)`、`joint_pos_multi_future_wrist_for_smpl (10, 6)`。
  - policy obs shape 为 `(930,)`，critic obs shape 为 `(1789,)`，action dim 为 `29`。
  - 第 10 iteration：computation `434 steps/s`，total episodes `320`，total timesteps `7680`，mean rewards `-24.92989`，mean length `10.49500`。
  - 未出现 OOM、NaN 或 CUDA 初始化错误。
  - 退出时仍有 PyTorch distributed 非致命清理警告：`destroy_process_group() was not called before program exit`。
- 日志目录：
  - `logs_rl/TRL_G1_Track/manager/universal_token/all_modes/sonic_release_test-20260511_152916`

### 清理与状态
- 远端检查命令：
  - `pgrep -af "gear_sonic/train_agent_trl.py|accelerate.commands.launch|accelerate launch|rsync|uv sync" || true`
  - `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
  - `df -h /root/autodl-tmp`
- 结果：
  - 无遗留训练、rsync 或 uv sync 进程；`pgrep` 只匹配到本次检查命令自身。
  - 两张 H20 显存占用均为 `0 MiB`，GPU 利用率 `0%`。
  - 数据盘 `/root/autodl-tmp` 剩余约 `2.0G`。

### 当前结论
- 完整默认路径的小规模多卡训练已通过：默认 robot 数据 + 默认 `data/bones_seed_smpl` + 双 H20 + mjlab backend + Universal Token 三 encoder。
- 仍未验证：正式规模 `num_envs=4096`、长时间训练、checkpoint resume/finetune、rough/trimesh terrain、render/eval、ONNX export、W&B 在线记录。
- 数据盘空间已接近满载，正式训练前不建议继续往该 AutoDL 数据盘放大数据或大 checkpoint。

## 2026-05-11T16:02:06+08:00 - AutoDL H20 每卡 env 容量测试

### 测试设置
- 目标：估计当前双 H20 AutoDL 服务器上默认真实数据训练时每张卡可开的 env 数。
- 解释：当前 `accelerate --num_processes=2` 下，`num_envs=N` 会让每个 rank / 每张卡各自创建 `N` 个 env，因此这里记录的是每卡 env 数。
- 命令共同设置：
  - `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1`
  - `.venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py`
  - `+exp=manager/universal_token/all_modes/sonic_release use_mjlab=True sim_type=mjlab headless=True`
  - `++algo.config.num_learning_iterations=1`
  - `manager_env.config.terrain_type=plane`
- 数据路径：未覆盖 `motion_file` / `smpl_motion_file`，使用默认真实数据 `data/motion_lib_bones_seed/robot_filtered` 和 `data/bones_seed_smpl`。

### 结果
- `num_envs=4096`（每卡 4096，双卡总 8192）：
  - 失败，退出码 `1`。
  - 两个 rank 均创建了 4096 env 并加载 `129785` 个 motion 文件。
  - 失败点：Warp CUDA graph 创建时 OOM。
  - 关键错误：`Warp CUDA error 2: out of memory (in function wp_cuda_graph_create_exec, /builds/omniverse/warp/warp/native/warp.cu:2899)`。
- `num_envs=2048`（每卡 2048，双卡总 4096）：
  - 通过，退出码 `0`。
  - 完成 learning iteration `1`。
  - computation `12761 steps/s`，collection `6.250s`。
  - total episodes `4096`，total timesteps `98304`，mean rewards `-22.35138`。
- `num_envs=3072`（每卡 3072，双卡总 6144）：
  - 通过，退出码 `0`。
  - 完成 learning iteration `1`。
  - computation `11792 steps/s`，collection `10.570s`。
  - total episodes `6144`，total timesteps `147456`，mean rewards `-18.76445`。
- `num_envs=3584`：
  - 用户要求停止继续测试，因此手动 SIGTERM 中断；不作为容量结论。

### 当前建议
- 保守建议按用户要求先使用 `num_envs=2048`，即每卡 2048、双卡总 4096。
- 已知 `num_envs=3072` 也能完成 1 iteration，但正式训练建议先用 2048 留显存和磁盘余量。
- `num_envs=4096` 每卡不可用，会在 Warp CUDA graph 创建阶段 OOM。
- 停止后检查：无遗留训练进程，双 H20 显存占用为 `0 MiB`。

## 2026-06-06T12:07:30+08:00 - cross_humanoid 方案一 H2 / SonicMJ backend 初版

### 目标
- 使用 workflow skill 推进 `cross_humanoid` 分支。
- 先按方案一做最小版本：复用官方 SONIC H2 配置和机器人参数，让 SonicMJ 重构 backend 能选择 H2 profile，并在 PC 上进入训练闭环。
- 不做 morphology-conditioned 网络结构，不改 CUDA/驱动/底层 GPU 环境。

### 官方原版核对
- 执行 `git fetch upstream`，`upstream/main` 更新到 `NVlabs/GR00T-WholeBodyControl` 最新引用。
- 官方原版包含：
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
  - `gear_sonic/envs/manager_env/robots/h2.py`
  - `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` 中 `h2` robot mapping
- 官方原版不包含 `sonic_mj/`，因此本轮只把官方 H2 profile 迁移到 SonicMJ backend。

### 代码改动
- `sonic_mj/assets.py`
  - 新增 `SonicRobotProfile`。
  - 保留 G1 profile，并新增 H2 profile。
  - H2 使用官方 H2 MJCF、31 DOF joint order、32 body order、官方 IsaacLab H2 joint/body order、H2 action scale、默认站姿。
  - 修正 H2 MJCF 直接加载时的 meshdir：官方 `h2.xml` 写 `meshdir="meshes/"`，实际 mesh 在 `gear_sonic/data/assets/robot_description/urdf/h2/meshes`。
- `sonic_mj/env_cfg.py`
  - 根据 `manager_env.config.robot.type` 选择 robot profile。
  - H2 时 action dim 为 31，motion body dim 为 32。
- `sonic_mj/mdp/commands.py`
  - motion lib 的 `isaaclab_joints` 和 motion DOF mapping 改为来自当前 profile。
  - order summary 改为检查 profile，而不是固定 G1。
- `sonic_mj/wrapper.py`
  - structured diagnostics 改为按当前 profile 检查 joint/body/action/order/action dim。
- workflow 文档已重写为本轮任务：
  - `TASK.md`
  - `PLAN.md`
  - `.workflow/artifacts/grounding.md`
  - `.workflow/artifacts/open_questions.json`

### 验证命令与结果
- 编译检查：
  - 命令：`uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：通过。
- profile 构造：
  - 命令：inline `get_sonic_robot_profile('g1_model_12_dex')` 和 `get_sonic_robot_profile('h2')`，并调用 `robot_cfg_fn().spec_fn()`。
  - 结果：G1 为 29 joints / 30 bodies / 29 DOF mapping；H2 为 31 joints / 32 bodies / 31 DOF mapping；二者均能生成 `MjSpec`。
- H2 env cfg 构造：
  - 命令：Hydra compose `+exp=manager/universal_token/all_modes/sonic_h2 use_mjlab=True sim_type=mjlab num_envs=2 headless=True manager_env.config.terrain_type=plane`，覆盖 `motion_file=data/motion_lib_bones_seed/robot_smoke`、`smpl_motion_file=data/smpl_filtered`。
  - 结果：`robot_type h2`，`action_dim 31`，`body_dim 32`，`isaaclab_dim 32`，`dof_map_dim 31`。
- PC GPU reset/step：
  - 命令：同上，但 device 自动选择 `cuda:0`。
  - 结果：失败于本机 PyTorch wheel 与 GPU 架构不兼容；RTX 5070 Ti Laptop GPU 为 `sm_120`，当前 PyTorch wheel 只支持到 `sm_90`。未修改 CUDA/驱动。
- H2 CPU reset/step：
  - 命令：同 H2 env，强制 `device='cpu'`。
  - 结果：通过。`actor_obs (2, 990)`，`critic_obs (2, 1907)`，`tokenizer (2, 1807)`，reward finite，action dim `31`，所有 profile order checks 为 `True`。
- H2 CPU 极小训练：
  - 命令：
    - `CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled timeout 120s uv run python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_h2 use_mjlab=True sim_type=mjlab num_envs=2 headless=True ++algo.trl.use_cpu=True ++algo.trl.bf16=False ++algo.trl.fp16=False ++algo.config.num_learning_iterations=1 ++manager_env.config.terrain_type=plane ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered`
  - 结果：退出码 `0`，完成 learning iteration `1`。
  - 训练日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260606_120606`
  - 关键结果：total timesteps `48`，mean rewards `-1.15169`，无 NaN/OOM。
  - 注意：CPU 小 batch 下打印 `batch_size must be a multiple of num_mini_batches`，但训练仍完成；正常 GPU/较大 `num_envs` 不应使用这个极小 batch 设置。
- G1 回归 cfg 检查：
  - 命令：Hydra compose `sonic_release + use_mjlab=True` 并构造 env cfg。
  - 结果：`robot_type g1_model_12_dex`，`action_dim 29`，`body_dim 30`。

### 当前结论
- SonicMJ backend 已具备最小 H2 profile 支持，可以创建 H2 env、reset/step，并在 PC CPU 上完成极小 PPO 训练启动闭环。
- 本机 GPU 不能验证 CUDA 训练，因为当前 PyTorch wheel 不支持 `sm_120`；这属于环境 wheel/GPU 架构问题，不是本轮代码问题。
- 方案一下一步应在支持当前 GPU 架构的 PyTorch wheel 或服务器 H20 环境上跑 H2 GPU smoke。
- 从 G1 官方 checkpoint 微调 H2 仍需单独处理 actor output 29 -> 31 的 checkpoint partial load / 输出层初始化策略。

## 2026-06-08T23:24:30+08:00 - PC 临时 torch/cu13 环境下 H2 正常训练验证

### 目标
- 用户要求按此前“临时把项目库切到 PC 能跑，测试完再恢复”的方式，确认 PC 上是否能开始正常训练。
- 重点不是 reset/step smoke，而是实际进入 PPO 多 iteration 训练。
- 同时尽量保证 PC 与服务器库版本差异只限于硬件 wheel，不影响代码效果。

### 初始环境事实
- 锁文件和服务器兼容环境仍是 `torch==2.7.1` / `torchvision==0.22.1` / CUDA 12.6 wheel 组合。
- 本机 GPU 为 `NVIDIA GeForce RTX 5070 Ti Laptop GPU`，capability `(12, 0)`。
- 用锁文件环境直接跑 CUDA 会触发 PyTorch warning：当前 wheel 支持到 `sm_90`，不支持本机 `sm_120`，随后 CUDA kernel 报 `no kernel image is available for execution on the device`。

### 临时 PC 环境调整
- 未修改 `pyproject.toml` 或 `uv.lock`。
- 先用 `uv pip install torch==2.11.0` 将当前 `.venv` 临时切到 `torch 2.11.0+cu130`。
- 发现 `uv run` 会自动按锁文件同步回 `torch 2.7.1`，因此临时 PC 测试必须直接使用 `.venv/bin/python`。
- `torch 2.11.0+cu130` 能在 PC 上使用 CUDA：
  - `torch.cuda.is_available() True`
  - GPU 为 `NVIDIA GeForce RTX 5070 Ti Laptop GPU (12, 0)`
  - CUDA matmul finite。
- 处理临时依赖问题：
  - 锁文件中的 `torchvision 0.22.1` 与临时 `torch 2.11.0` ABI 不匹配，报 `operator torchvision::nms does not exist`。
  - 临时从 PyTorch nightly cu130 源安装 `torchvision 0.26.0+cu130` 后，`torch`、`torchvision`、`trl.PPOConfig`、`gear_sonic.trl.modules.base_module.BaseModule` 均可导入。
  - `torch 2.11/cu130` 下 NVRTC 需要临时增加：
    - `LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`
  - 该环境变量只在测试命令中设置，未写入 shell 配置或项目文件。

### 正常训练命令
- 命令：
  - `export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"`
  - `WANDB_MODE=disabled timeout 1800s .venv/bin/python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_h2 use_mjlab=True sim_type=mjlab num_envs=128 headless=True ++algo.config.num_learning_iterations=20 ++manager_env.config.terrain_type=plane ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_smoke ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered`
- 训练日志目录：
  - `logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260608_232333`

### 训练结果
- 退出码 `0`。
- 完成 learning iteration `1` 到 `20`，不是只完成 smoke/reset/step。
- 环境配置：
  - `num_envs=128`
  - device `cuda:0`
  - H2 action dim `31`
  - policy obs shape `(990,)`
  - critic obs shape `(1907,)`
  - tokenizer 含 3 encoder 输入项。
- 第 20 iteration 关键结果：
  - computation `1880 steps/s`
  - collection `1.179s`
  - learning `0.454s`
  - total episodes `2560`
  - total timesteps `61440`
  - mean rewards `-1.70050`
  - mean entropy `-47.86094`
  - non-finite state / joint / body termination 全部为 `0.0000`
- 未出现 OOM、NaN、CUDA kernel image 错误或训练中断。

### 恢复服务器兼容环境
- 测试后执行：
  - `uv sync --frozen`
- 该命令卸载临时 cu13 / torch 2.11 / torchvision 0.26，并恢复锁文件环境。
- 首次恢复后 `torch` 导入缺 `libcudnn.so.9`，原因是 CUDA 12 依赖 wheel 文件不完整；强制重装 `nvidia-cudnn-cu12==9.5.1.17` 和 `nvidia-nccl-cu12==2.26.2` 后，再运行 `uv run` 自动按锁文件恢复 `nvidia-cublas-cu12==12.6.4.1` / `nvidia-cuda-nvrtc-cu12==12.6.77`。
- 最终恢复验证：
  - `torch 2.7.1+cu126`
  - `torch.version.cuda 12.6`
  - `torch.cuda.is_available() True`
  - `torchvision 0.22.1+cu126`
  - `pyproject.toml` / `uv.lock` diff 行数为 `0`
  - 无遗留 `gear_sonic/train_agent_trl.py` 或 accelerate 训练进程。

### 结论
- PC 上可以正常开始 H2 SonicMJ 训练：已用 GPU 跑完 20 iteration / 61440 timesteps。
- PC 与服务器的版本差异应限定为硬件 wheel 层：
  - 服务器/锁文件：`torch 2.7.1+cu126`，已在 H20 上通过。
  - PC 临时：`torch 2.11.0+cu130` + `torchvision 0.26.0+cu130`，仅用于支持本机 `sm_120`。
- 本轮没有修改训练代码来适配 PC-only 版本；mjlab、mujoco-warp、trl、transformers、SONIC/SonicMJ 代码路径保持同一套。因此目前没有证据表明 PC/服务器 wheel 差异会改变代码语义，主要风险是数值细节和底层 CUDA kernel 差异，后续正式结论仍应以 H20 锁文件环境为准。

## 2026-06-09T22:42:46+08:00 - workflow 恢复与 H2 当前工作区回归验证

### 已读取
- `AGENTS.md`：确认本项目使用 uv 环境，不使用 conda；不得修改 CUDA、显卡驱动、系统 GPU 组件；workflow 恢复必须先处理 `PERSON.md` open 条目并把结论写回项目文档。
- `PERSON.md`：无实际 open 条目，只有模板。
- `TASK.md`、`PLAN.md`：当前目标是 `cross_humanoid` 分支方案一，为 SonicMJ backend 增加最小 H2 robot profile，并让 `sonic_h2.yaml + use_mjlab=True + sim_type=mjlab` 能进入训练闭环。
- `PROGRESS.md`：历史记录显示 H2 profile 初版、PC CPU 极小训练、PC 临时 torch/cu13 GPU 20 iteration 训练均已通过。
- `task.json`：`plan_approved=true`，环境为 `uv` + `.venv` + `existing`，但摘要仍停留在旧 H20 计划阶段，需要本轮更新。
- `.workflow/artifacts/open_questions.json`：只有非 blocking 问题，分别是 H2 正式 motion 数据选择和 G1 checkpoint 到 H2 的 partial load/freeze 策略。
- `.workflow/artifacts/grounding.md`、`README.md`、`pyproject.toml`：确认 SonicMJ backend 是本项目重构层，根项目使用 Python 3.10 uv 环境；当前锁文件仍为服务器兼容的 `torch==2.7.1`。
- git 状态：当前分支 `cross_humanoid`，已有 H2 相关源码改动和 workflow 文档改动；存在未跟踪 `training_curves_szsevsw9.png`、`training_curves_szsevsw9.svg`、`wandb/`，本轮未回退或删除这些既有内容。

### 判断
- 当前没有 blocking open question，不需要停在询问阶段。
- `sonic_mj/assets.py`、`sonic_mj/env_cfg.py`、`sonic_mj/mdp/commands.py`、`sonic_mj/wrapper.py` 已包含 H2 profile 泛化改动；合理下一步不是继续改代码，而是对当前工作区做可复现回归验证并更新 workflow 状态。
- 本机锁文件环境仍使用服务器兼容 torch/cu126；为避免本机 `sm_120` GPU wheel 兼容问题，本轮只执行 CPU reset/step，不修改 `.venv`、`pyproject.toml`、`uv.lock`、CUDA 或驱动。

### 命令与结果
- `uv run python -m compileall sonic_mj gear_sonic/train_agent_trl.py`
  - 结果：通过。
- 第一次 inline profile/config 检查失败：
  - 原因：测试代码访问了当前 `mujoco.MjSpec` 不存在的 `njnt` 属性。
  - 结论：测试脚本写法问题，不是项目源码问题。
- 第二次 inline profile/config 检查失败：
  - 原因：Hydra `config_name='config'` 不存在；训练入口实际使用 `config_name='base'`。
  - 已通过读取 `gear_sonic/train_agent_trl.py` 和 `gear_sonic/config/` 确认。
- 修正后的 profile/config 检查：
  - 命令：inline `uv run python`，构造 `get_sonic_robot_profile('g1_model_12_dex')` / `get_sonic_robot_profile('h2')`，并用 Hydra `config_name='base'` compose `sonic_release` 与 `sonic_h2`。
  - 结果：通过。
  - G1：profile `29` joints / `30` bodies / `29` DOF map；env cfg action dim `29`，body dim `30`，IsaacLab dim `30`。
  - H2：profile `31` joints / `32` bodies / `31` DOF map；env cfg action dim `31`，body dim `32`，IsaacLab dim `32`。
- H2 CPU reset/step：
  - 命令：`CUDA_VISIBLE_DEVICES= uv run python - <<'PY' ... create_mjlab_manager_env(cfg, 'cpu') ... PY`
  - 配置：`+exp=manager/universal_token/all_modes/sonic_h2 use_mjlab=True sim_type=mjlab num_envs=2 headless=True manager_env.config.terrain_type=plane`，并覆盖 `motion_file=data/motion_lib_bones_seed/robot_smoke`、`smpl_motion_file=data/smpl_filtered`。
  - 结果：通过。`device cpu`，`action_dim 31`，reset/step obs shape 均为 `actor_obs (2, 990)`、`critic_obs (2, 1907)`、`tokenizer (2, 1807)`；reward shape `(2,)` 且 finite；done shape `(2,)`；info keys 为 `['env_actions', 'episode', 'log', 'time_outs', 'to_log']`。
  - order diagnostics：`robot_joints_match_profile`、`robot_bodies_match_profile`、`motion_bodies_match_profile`、`action_joints_match_profile`、`policy_joint_pos_order_matches_profile`、`policy_joint_vel_order_matches_profile`、`policy_action_order_matches_profile`、`motion_dof_mapping_identity`、`action_dim_matches_profile` 全部为 `True`。
  - 运行中出现 `Warp CUDA error 100: no CUDA-capable device is detected`，但 env device 为 CPU 且 reset/step 成功；该信息来自 Warp 初始化 CUDA driver 探测，不影响本轮 CPU 验证。

### 当前结论
- 当前工作区的 H2 SonicMJ backend 支持仍然可复现：静态编译、G1/H2 cfg 构造、H2 CPU reset/step 均通过。
- 没有新增代码修改需求；本轮只更新 workflow 记录和 `task.json`。
- 剩余非阻塞问题仍是 H2 正式 motion 数据选择，以及 G1 checkpoint 微调 H2 时 29->31 输出层 partial load/freeze 策略。

### 恢复纠偏
- 重新核对 `TASK.md`、`PLAN.md`、`task.json`、`.workflow/runtime/run-prompt.txt` 后确认：最新 workflow 目标不是继续 H2 backend profile 实现，而是 H2 shape-aware warm-start 的计划阶段。
- `PLAN.md` 当前明确写着“状态：等待用户确认。确认前不修改训练代码、不跑 warm-start 训练。”
- 因此本轮执行的 compile/profile/cfg/reset-step 只作为当前工作区只读回归验证记录，不推进 `PLAN.md` 的 M1-M7，也不代表 warm-start 计划已批准。
- `task.json.last_result` 已修正为 `plan_ready_waiting_user_confirmation`。

### blocked_reason
等待用户确认当前 `PLAN.md`。用户确认后才能从 M1 开始实现 checkpoint 兼容读取、shape-aware state_dict 过滤报告，并执行本机 A-lite / B warm-start smoke。

## 2026-06-09T22:39:50+08:00 - H2 shape-aware warm-start grounding / plan / review 完成

### 用户新增要求
- 把 SONIC 迁移到 H2 的下一步聚焦为：先实现官方 G1 checkpoint 的 shape-aware partial load / warm-start。
- 官方 G1 权重只作为 H2 initialization，不假设 G1 latent 可直接迁移。
- A-lite 作为诊断方向，B 作为正式 warm-start 微调方向，尽早为 D-small robot-conditioning 预留结构。
- 测试顺序为本机先初测，再到 PKU 服务器 `/home/nvme02/GR00T` 测试；PKU H2 数据在 `/home/nvme02/GR00T/dataset`。

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复 workflow 状态和项目约束。
- `README.md`、`docs/source/user_guide/training.md`、`docs/source/user_guide/new_embodiments.md`：确认官方 G1 finetune、SONIC 训练和新 embodiment 要求。
- `gear_sonic/train_agent_trl.py`：确认 `pretrained_model` 加载入口存在，但当前不是 shape-aware。
- `gear_sonic/trl/trainer/ppo_trainer.py`：确认 trainer resume 已有旧 TRL checkpoint shim，但属于完整恢复训练状态路径。
- `gear_sonic/trl/callbacks/model_save_callback.py`：确认 checkpoint 保存 key 包括 `policy_state_dict`、`value_state_dict`、optimizer、scheduler、state、env_state。
- `gear_sonic/eval_agent_trl.py`：确认 eval 有 `std` / `log_std` 兼容逻辑，可作为 checkpoint 工具参考。
- `gear_sonic/trl/modules/universal_token_modules.py`：确认 action dim 来自 env，已有 `meta_action_dim`、`body_action_dim`、active encoder/decoder 扩展点。
- `sonic_mj/assets.py`、`sonic_mj/env_cfg.py`、`sonic_mj/mdp/commands.py`、`sonic_mj/wrapper.py`：确认当前分支已有 H2 backend 初版。

### grounding 证据
- `sonic_release/last.pt` 存在，大小约 `447.7M`。
- 直接 `uv run python` 读取 `sonic_release/last.pt` 当前失败：
  - 错误：`AttributeError: Can't get attribute 'OnlineTrainerState' on trl.trainer.utils`
  - 结论：warm-start loader 必须先复用旧 TRL checkpoint 兼容 shim。
- 本机 H2 数据路径 `/home/ykj/Downloads/dataset/bones-seed/h2_v30_chest_soft` 存在，约 `729M`，当前发现 `128` 个 CSV 文件；尚不是 motion_lib PKL 训练目录。
- 当前 `git status` 已有未提交代码和文档改动；本轮未回退任何已有内容。

### 已完成文档
- 更新 `TASK.md`：写入本轮 H2 warm-start 硬需求、环境/数据约束、当前代码事实和验收条件。
- 更新 `.workflow/artifacts/grounding.md`：记录 checkpoint、loader、H2 数据、A-lite/B/D-small 风险。
- 更新 `.workflow/artifacts/paper_project_map.json`：结构化记录 SONIC 论文、官方实现、目标代码、A-lite/B/D-small 映射。
- 更新 `.workflow/artifacts/open_questions.json`：记录非 blocking 问题；当前没有阻塞生成计划的问题。
- 重写 `PLAN.md`：包含 M0-M8 里程碑、完成标准、测试命令和 review 点。
- 更新 Codex 多角色 review：
  - `.workflow/artifacts/reviews/scope.md`
  - `.workflow/artifacts/reviews/milestones.md`
  - `.workflow/artifacts/reviews/architecture.md`
  - `.workflow/artifacts/reviews/testing.md`
  - `.workflow/artifacts/reviews/executor.md`

### 判断
- 当前没有 blocking open question；可以生成计划。
- 按 workflow 规则，本轮停在等待用户确认 `PLAN.md`。
- 代码尚未修改；下一步确认后才实现 checkpoint 工具和 `train_agent_trl.py` warm-start 接入。

### 本轮命令与结果
- `sed -n ...` / `tail -220 PROGRESS.md`：读取 workflow 必读文档和历史进度。
- `rg -n "checkpoint|resume|load_state|pretrained|partial|strict" gear_sonic sonic_mj ...`：定位 checkpoint 与加载入口。
- `sed -n ... gear_sonic/train_agent_trl.py`：确认 `pretrained_model` 当前加载逻辑。
- `sed -n ... gear_sonic/trl/modules/universal_token_modules.py`：确认 action dim 和预留扩展点。
- `sed -n ... README.md docs/source/user_guide/training.md docs/source/user_guide/new_embodiments.md`：确认官方训练和新机器人说明。
- `find . -maxdepth 4 -name last.pt ...`：确认 `sonic_release/last.pt` 存在。
- `uv run python - <<'PY' ... torch.load('sonic_release/last.pt') ... PY`：失败于旧 TRL `OnlineTrainerState` pickle 路径，已记录为计划内修复点。
- `find /home/ykj/Downloads/dataset/bones-seed/h2_v30_chest_soft ...`：确认本机 H2 数据当前为 CSV 结构。
- `git status --short --branch`：确认分支 `cross_humanoid`，存在既有未提交改动。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后从 M1 开始实现 checkpoint 兼容读取与 shape-aware loader，然后执行 A-lite 本机诊断。

## 2026-06-09T23:18:00+08:00 - H2 shape-aware warm-start 本机实现与测试完成

### 代码实现
- 新增 `gear_sonic/trl/utils/checkpoint.py`：
  - `install_legacy_trl_checkpoint_shim()`：复用旧 TRL checkpoint 所需的 `OnlineTrainerState` / `exact_div` 兼容符号。
  - `load_checkpoint()`：集中读取旧 SONIC checkpoint，不修改 checkpoint 文件。
  - `select_checkpoint_state_dict()`：支持 `policy_state_dict`、`actor_model_state_dict`、`value_state_dict` 和 `state_dict` alias。
  - `shape_aware_filter_state_dict()`：只保留 key 存在且 shape 完全一致的参数。
  - `convert_std_log_std()`：支持 `std` 与 `log_std` 双向转换；本轮 G1->H2 因 action dim 29->31，`std` 被正确归入 `skipped_shape`。
  - `ShapeAwareLoadReport` / `save_shape_aware_report()`：输出 JSON 报告。
- 修改 `gear_sonic/train_agent_trl.py`：
  - `algo.config.pretrained_model.shape_aware=true` 时走 shape-aware warm-start。
  - 默认 `load_policy=true`，不需要额外写 `module_mapping`；可选 `load_value=true` 追加 critic partial load。
  - warm-start 只加载模型初始化参数，不加载 optimizer、scheduler、env state、global step；`resume=True` / `+checkpoint=` 完整恢复语义未改。
  - 默认仍保留旧的 strict `module_mapping` 加载路径。
- 修改 `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`：
  - 增加 no-op `algo.config.robot_conditioning.enabled=false`，只作为 D-small 后续结构边界；不改变 Universal Token 网络结构。

### 本机环境与数据
- 使用项目 `uv` / `.venv`，未修改 CUDA、显卡驱动、系统 GPU 组件、`pyproject.toml` 或 `uv.lock`。
- 本机 H2 CSV 路径 `/home/ykj/Downloads/dataset/bones-seed/h2_v30_chest_soft` 当前可见，约 `1.5G`，包含 `320` 个 CSV 文件；尚不是 motion_lib PKL 训练目录。
- 因 H2 CSV 尚未转换为 motion_lib PKL，本机 warm-start smoke 使用已有 `data/motion_lib_bones_seed/robot_smoke` 和 `data/smpl_filtered` 验证代码路径。

### 测试结果
- `uv run python -m compileall gear_sonic/trl/utils/checkpoint.py gear_sonic/train_agent_trl.py sonic_mj gear_sonic/trl`
  - 结果：通过。
- checkpoint 读取：
  - 命令：inline `uv run python` 调用 `load_checkpoint('sonic_release/last.pt')`。
  - 结果：通过；checkpoint keys 为 `args`、`env_state_dict`、`lr_scheduler_state_dict`、`optimizer_state_dict`、`policy_state_dict`、`state`、`value_state_dict`。
  - `policy_state_dict` 有 `55` 个参数，`value_state_dict` 有 `17` 个参数。
- A-lite policy-only 诊断：
  - 日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260609_231018`
  - 命令要点：`sonic_h2 use_mjlab=True sim_type=mjlab num_envs=2 ++algo.trl.use_cpu=True ++algo.config.num_learning_iterations=1 ++algo.config.pretrained_model.path=sonic_release/last.pt ++algo.config.pretrained_model.shape_aware=True`。
  - shape 报告：`loaded=47`、`skipped_shape=8`、`missing=0`、`unexpected=0`、`source_action_dim=29`、`target_action_dim=31`。
  - 跳过项包括：
    - `std`: `[29] -> [31]`
    - `actor_module.decoders.g1_dyn.module.12.weight`: `[29, 512] -> [31, 512]`
    - `actor_module.decoders.g1_dyn.module.12.bias`: `[29] -> [31]`
    - H2 输入维度变化导致的 `g1_dyn` 首层、`g1_kin` 输出层、`g1/teleop` encoder 首层。
  - 训练完成 `1` iteration / `48` timesteps，退出码 `0`，mean rewards `-1.04373`，非有限状态/关节/body termination 均为 `0.0000`。
- B policy-only warm-start smoke：
  - 日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260609_231149`
  - 命令要点：同 A-lite，额外 `++algo.config.num_learning_iterations=2 ++algo.config.num_mini_batches=1 ++algo.config.num_learning_epochs=1`。
  - 结果：完成 `2` iterations / `96` timesteps，退出码 `0`。
  - 第 2 iteration mean rewards `-1.11388`；非有限状态/关节/body termination 均为 `0.0000`。
- policy + value warm-start smoke：
  - 日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260609_231548`
  - 命令要点：同 A-lite，额外 `++algo.config.pretrained_model.load_value=True ++algo.config.num_mini_batches=1 ++algo.config.num_learning_epochs=1`。
  - policy 报告：`loaded=47`、`skipped_shape=8`。
  - value 报告：`loaded=14`、`skipped_shape=3`；跳过 `critic_module.module.0.weight` `[2048, 1645] -> [2048, 1907]` 和 critic running mean/var `[1645] -> [1907]`。
  - 训练完成 `1` iteration / `48` timesteps，退出码 `0`，mean rewards `-1.04373`，非有限状态/关节/body termination 均为 `0.0000`。
- G1/H2 cfg 回归：
  - G1 `sonic_release + use_mjlab=True`：action dim `29`，profile joints `29`，profile bodies `30`。
  - H2 `sonic_h2 + use_mjlab=True`：action dim `31`，profile joints `31`，profile bodies `32`，`robot_conditioning_enabled=False`。

### 已知测试噪声
- `num_envs=2` 且默认 `num_mini_batches=4` 时 TRL 会打印 `batch_size must be a multiple of num_mini_batches`；后续 smoke 通过 `++algo.config.num_mini_batches=1` 消除该配置噪声。
- CPU 小 batch 会触发 `ratio_stats.var()` 的 PyTorch degrees-of-freedom warning；训练退出码为 `0`。

### PKU 测试状态
- 已确认 SSH alias `PKU` 可免密登录，远端主机 `instance-afs92r3e`。
- 远端项目目录：`/home/nvme02/GR00T/GR00T`；远端数据目录：`/home/nvme02/GR00T/dataset`。
- 远端环境：
  - 非交互 shell 中 `uv` 不在 PATH；测试使用项目 `.venv/bin/python`，未安装依赖、未改 base、未改 CUDA/驱动。
  - Python `3.10.6`，`torch 2.7.1+cu126`，`torch.cuda.is_available() True`。
  - GPU：`8` 张 `NVIDIA H20`，device capability `(9, 0)`。
- 远端数据：
  - `/home/nvme02/GR00T/dataset` 约 `158G`。
  - 当前 `COUNT_CSV=142220`，`COUNT_PKL=0`；因此正式 H2 数据训练仍缺 motion_lib PKL。
  - 为测试代码路径，已同步本机 `data/motion_lib_bones_seed/robot_smoke` 到远端项目 `data/motion_lib_bones_seed/robot_smoke`。
- 已同步到远端的本轮必要代码文件：
  - `gear_sonic/train_agent_trl.py`
  - `gear_sonic/trl/utils/checkpoint.py`
  - `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`
  - `sonic_mj/assets.py`
  - `sonic_mj/env_cfg.py`
  - `sonic_mj/mdp/commands.py`
  - `sonic_mj/wrapper.py`
  - 未覆盖远端已有 mesh 修改。
- 远端静态与 checkpoint 测试：
  - `.venv/bin/python -m compileall gear_sonic/trl/utils/checkpoint.py gear_sonic/train_agent_trl.py sonic_mj gear_sonic/trl`：通过。
  - `load_checkpoint("sonic_release/last.pt")`：通过；checkpoint keys 包含 `policy_state_dict` 和 `value_state_dict`，policy 参数 `55` 个，value 参数 `17` 个。
- 远端单卡 H20 policy-only warm-start：
  - 命令要点：`CUDA_VISIBLE_DEVICES=0 .venv/bin/python gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_h2 use_mjlab=True sim_type=mjlab num_envs=128 ... ++algo.config.pretrained_model.shape_aware=True`。
  - 日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260609_232952`。
  - shape 报告：`loaded=47`、`skipped_shape=8`、`missing=0`、`unexpected=0`，正确跳过 `29->31` action/std 相关参数和 H2 维度变化层。
  - 完成 `2` iterations / `6144` timesteps，退出码 `0`。
  - 第 2 iteration：`547 steps/s`，mean rewards `-1.56658`，非有限 state/joint/body termination 均为 `0.0000`。
- 远端 2 卡 H20 accelerate warm-start：
  - 命令要点：`CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29617 .venv/bin/python -m accelerate.commands.launch --num_processes=2 gear_sonic/train_agent_trl.py ... num_envs=64 ... ++algo.config.pretrained_model.shape_aware=True`。
  - 日志目录：`logs_rl/TRL_H2_Track/manager/universal_token/all_modes/sonic_h2_test-20260609_233410`。
  - 两个 rank 均完成 H2 env 构造、motion loading 和 shape-aware policy load。
  - shape 报告：`loaded=47`、`skipped_shape=8`、`missing=0`、`unexpected=0`。
  - 完成 `1` iteration / `3072` timesteps，退出码 `0`。
  - 训练日志：`303 steps/s`，mean rewards `-1.44314`，非有限 state/joint/body termination 均为 `0.0000`。
- 远端测试噪声：
  - SSH 输出 `remote port forwarding failed for listen port 10408`，不影响命令执行。
  - motion loader 输出 `Could not increase file descriptor limit`，不影响本次 smoke。
  - 2 卡结束时 rank0 有 `destroy_process_group() was not called` warning，退出码仍为 `0`。
  - 这些都不是当前代码逻辑错误，本轮未因此修改代码。

## 2026-06-13T16:47:23+08:00 - PKU H2 4卡训练/2卡验证计划阶段完成

### 用户新增要求
- 在 PKU 服务器上按计划训练。
- 最多只能用 GPU `0-5` 共 6 张。
- GPU 分组固定为 4 卡训练、2 卡验证：
  - 训练：`CUDA_VISIBLE_DEVICES=0,1,2,3`。
  - 验证：`CUDA_VISIBLE_DEVICES=4,5`。
  - 禁止使用 GPU `6,7`。
- 开始训练前，在项目文件夹中提供详细 Markdown 文件，写清训练、验证、断点续训等每一步命令。
- 本轮作为 workflow 计划阶段，完成 grounding、计划和 Codex 多角色 review 后停在等待用户确认 `PLAN.md`。

### 已读取
- `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`：恢复 workflow 状态与项目硬约束。
- `README.md`、`docs/source/user_guide/training.md`、`docs/source/user_guide/new_embodiments.md`：确认官方训练、multi-GPU、eval 和新 embodiment 说明。
- `gear_sonic/train_agent_trl.py`：确认 SonicMJ/mjlab 训练入口、warm-start 与 full resume 的不同路径。
- `gear_sonic/trl/utils/checkpoint.py`：确认 shape-aware checkpoint loader 已存在。
- `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2.yaml`：确认 H2 配置、31 action dim、motion_file 需命令行指定。
- `gear_sonic/config/algo/ppo_im_phc.yaml`：确认默认 `num_learning_iterations=100000`、`save_interval=500`、`eval_frequency=500`。
- `gear_sonic/eval_agent_trl.py`、`gear_sonic/eval_exp.py`：确认 eval 入口；`eval_exp.py` 内部 launcher 未显式 `--num_processes`，因此计划优先使用 2 卡 one-shot eval。

### PKU 只读 grounding
- 命令：`ssh PKU 'nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader ...'`
  - 结果：远端主机 `instance-afs92r3e`，8 张 `NVIDIA H20`，每张约 `97871 MiB`；本计划只使用 `0-5`。
- 命令：`ssh PKU 'cd /home/nvme02/GR00T/GR00T && .venv/bin/python ...'`
  - 结果：远端项目存在，分支只读显示 `SONICMJ`。
  - Python `3.10.6`，torch `2.7.1+cu126`，`torch.cuda.is_available() True`，`torch.cuda.device_count() 8`。
  - 远端工作区已有 mesh/H2 配置改动，本轮未回退或覆盖。
- 命令：`ssh PKU 'du -sh /home/nvme02/GR00T/dataset; find ...'`
  - 结果：数据目录约 `423G`。
  - `COUNT_PKL=3716`，`COUNT_CSV=284440`。
  - `/home/nvme02/GR00T/dataset/h2_motion_lib_chest_soft_filtered` 约 `3696` 个 PKL。
  - `/home/nvme02/GR00T/dataset/h2_motion_lib_local_sample` 约 `20` 个 PKL。
  - SMPL 候选为 `/home/nvme02/GR00T/GR00T/data/smpl_filtered`。

### 已完成文档
- 更新 `TASK.md`：写入本轮 PKU 训练/验证硬需求、GPU 限制、环境约束、当前代码事实和验收条件。
- 新增 `PKU_H2_TRAIN_VALIDATE_PLAN.md`：详细列出登录检查、数据检查、preflight、4 卡训练 smoke、2 卡验证 smoke、正式 4 卡训练、正式 2 卡验证、周期验证、断点续训、监控和停止命令。
- 更新 `.workflow/artifacts/grounding.md`：记录项目现状、关键证据、远端只读结果和风险。
- 更新 `.workflow/artifacts/paper_project_map.json`：结构化记录 SONIC 论文、参考实现、目标代码、PKU 环境映射。
- 更新 `.workflow/artifacts/open_questions.json`：记录 W&B online/offline、H2 filtered PKL 是否作为首轮正式训练数据、是否先用 plane terrain 的非阻塞问题。
- 更新 `PLAN.md`：M0-M6 覆盖文档、preflight、smoke、正式训练、验证、断点续训、监控。
- 更新 Codex 多角色 review：
  - `.workflow/artifacts/reviews/scope.md`
  - `.workflow/artifacts/reviews/milestones.md`
  - `.workflow/artifacts/reviews/architecture.md`
  - `.workflow/artifacts/reviews/testing.md`
  - `.workflow/artifacts/reviews/executor.md`

### 判断
- 当前没有 blocking open question；PKU 已有 H2 PKL 候选目录，足以制定可执行计划。
- 仍有非阻塞决策：W&B online/offline、首轮是否直接用 3696 个 H2 filtered PKL、是否先固定 plane terrain。
- 按 workflow 规则，本轮停在等待用户确认 `PLAN.md`；尚未启动训练或验证。

### 本轮命令与结果
- `sed -n ...` / `tail -n ...`：读取 workflow 必读文件、README、训练文档、H2 配置和训练/eval 入口。
- `rg -n "pretrained_model|shape_aware|checkpoint|resume|eval|sonic_h2" ...`：定位 warm-start、resume、eval 和 H2 配置证据。
- `ssh PKU ... nvidia-smi ...`：只读确认 PKU GPU 与数据，未修改远端环境。
- `ssh PKU ... .venv/bin/python ...`：只读确认远端 Python/torch/CUDA 可见性，未安装依赖。
- `python -m json.tool .workflow/artifacts/open_questions.json .workflow/artifacts/paper_project_map.json task.json`：JSON 格式检查通过。
- `rg -n 'CUDA_VISIBLE_DEVICES=|0,1,2,3|4,5|6,7' PKU_H2_TRAIN_VALIDATE_PLAN.md PLAN.md TASK.md .workflow/artifacts`：确认本轮计划命令只使用训练 `0,1,2,3` 和验证 `4,5`，没有 `CUDA_VISIBLE_DEVICES` 指向 `6` 或 `7`。
- 本轮未创建临时测试脚本，未运行训练，未修改 CUDA/驱动/系统 GPU 组件。

### 阻塞与下一步
- 当前阻塞类型：等待用户确认 `PLAN.md`。
- 用户确认后，按 `PKU_H2_TRAIN_VALIDATE_PLAN.md` 从 preflight 开始执行；先跑 4 卡训练 smoke 和 2 卡 eval smoke，再进入正式 4 卡训练。

## 2026-06-18T21:28:23+08:00 - 当前恢复结论补充

- 本轮已按 workflow 要求读取 `AGENTS.md`、`PERSON.md`、`TASK.md`、`PLAN.md`、`PROGRESS.md`、`task.json`、`.workflow/artifacts`、`PKU_H2_TRAIN_VALIDATE_PLAN.md`、必要 README/docs 和 git status。
- `PERSON.md` 无实际 open 条目；`.workflow/artifacts/open_questions.json` 只有非 blocking 问题。
- 当前最新可执行状态仍是等待用户确认 `PLAN.md`：`task.json.plan_approved=false`，`PLAN.md` 写明确认前不启动 PKU 训练或验证。
- 本轮未登录 PKU、未运行训练/验证、未改训练代码、未安装依赖、未触碰 CUDA/驱动/系统 GPU 组件。
- 已同步更新 `task.json.blocked_reason`：等待用户确认 `PLAN.md`；确认后才能进入 PKU preflight、4 卡训练 smoke、2 卡 eval smoke 和正式训练。
