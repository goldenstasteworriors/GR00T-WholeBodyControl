# H2 checkpoint diagnosis

## 输入状态

- 本机已读取 `H2_TRANSFER_EXPERIMENT_LOG.md`，其中记录了 `pr12000`、`hy2000`、`tf2000` 的固定 8 motion eval 汇总。
- 本机当前不能访问 PKU 路径下的 compare8 `metrics_eval.json`，以下路径不存在：
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260617_precision_compare_h2/pr12000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260618_hybrid_compare_h2/hy2000/metrics_eval.json`
  - `/home/nvme02/GR00T/GR00T/logs_eval/20260615_tracking_first_compare_h2/tf2000/metrics_eval.json`
- 已新增 `gear_sonic/scripts/h2_eval_diagnostics.py`，后续拿到真实 `metrics_eval.json` 后可直接生成 per-motion markdown/csv。

## 固定 8 motion 汇总

| checkpoint | success | progress | mpjpe_g | mpjpe_l | foot_g | vr_g | terminated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pr12000` | 0.000 | 0.1918 | 0.0703 | 0.0538 | 0.0921 | 0.0668 | 8/8 |
| `hy2000` | 0.250 | 0.4005 | 0.1338 | 0.0670 | 0.1600 | 0.1477 | 6/8 |
| `tf2000` | 0.625 | 0.6891 | 0.1547 | 0.0889 | 0.1727 | 0.1668 | 3/8 |

## 当前判断

- `pr12000` 是 precision 端点：global/local MPJPE、foot、VR 都最好，但 8/8 termination，说明它只在早期短片段保持较好 tracking。
- `tf2000` 是 survival 端点：success/progress 明显最好，但 global MPJPE、foot、VR 都恶化，符合“能跑但 global drift”的现象。
- `hy2000` 是折中点：progress 比 `pr12000` 高，但 MPJPE 已接近 drift 端点，不满足 `progress > 0.4` 且 `mpjpe_g < 0.10` 的目标。
- 仅凭平均 MPJPE 无法解释失败，需要补 per-motion 和时间序列。当前缺少真实 `metrics_eval.json`，所以 per-motion 失败 motion 和 drift 时间曲线还不能在本机复现。

## 已落地的 global-anchor logging

`gear_sonic/trl/callbacks/im_eval_callback.py` 现在会在 eval 时记录：

- `anchor_xy_error_mean/max/final`
- `anchor_z_error_mean`
- `anchor_heading_error_mean/max/final`
- `anchor_ori_error_mean`

默认只写 per-motion 标量，设置 `SONIC_SAVE_EVAL_TRACE=1` 时额外保存逐步 trace 到 `metrics_eval.json` 的 `eval/all_metrics_dict.anchor_error_traces`。

## 后续命令模板

```bash
SONIC_SAVE_EVAL_TRACE=1 WANDB_MODE=disabled uv run accelerate launch --num_processes=1 gear_sonic/eval_agent_trl.py \
  checkpoint=<CHECKPOINT> use_mjlab=True sim_type=mjlab headless=True \
  ++num_envs=8 ++output_dir=<EVAL_OUT>
```

```bash
uv run python gear_sonic/scripts/h2_eval_diagnostics.py \
  --case pr12000=<PR12000_EVAL_OUT>/metrics_eval.json \
  --case hy2000=<HY2000_EVAL_OUT>/metrics_eval.json \
  --case tf2000=<TF2000_EVAL_OUT>/metrics_eval.json \
  --output-md .workflow/artifacts/h2_eval_diagnostics.md \
  --output-csv .workflow/artifacts/h2_eval_diagnostics.csv
```
