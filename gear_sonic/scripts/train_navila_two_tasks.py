"""Single-GPU LoRA adaptation using the official NaVILA training implementation.

Adds lab dataset registration, trajectory-held-out evaluation and memory metrics.
All robot actions and model prompts are English; this script never controls a robot.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from unittest import mock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navila-repo", type=Path, required=True)
    parser.add_argument("--lab-dataset", type=Path, required=True)
    options, training_arguments = parser.parse_known_args()
    sys.path.insert(0, str(options.navila_repo.resolve()))
    dataset_root = options.lab_dataset.resolve()

    import torch
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback

    from llava.data import datasets_mixture
    from llava.data.builder import build_dataset
    from llava.train.transformer_normalize_monkey_patch import patched_normalize

    module = importlib.import_module("llava.train.train")
    original_data = module.make_supervised_data_module
    original_trainer = module.LLaVATrainer
    held_out = {}
    for split in ("train", "validation", "test"):
        datasets_mixture.add_dataset(datasets_mixture.Dataset(
            dataset_name=f"navila_lab_{split}", dataset_type="vlnce",
            data_path=str(dataset_root / f"{split}.json"), image_path=str(dataset_root),
            description="Two lab tasks; command-integral weak metric labels; human-confirmed successful stops.",
        ))

    def lab_data(tokenizer, data_args, training_args):
        data = original_data(tokenizer, data_args, training_args)
        validation = build_dataset("navila_lab_validation", data_args, training_args, tokenizer)
        test = build_dataset("navila_lab_test", data_args, training_args, tokenizer)
        training_args.eval_sample_lens = [len(d) for d in validation.datasets]
        data["eval_dataset"] = validation
        held_out["test"] = test
        return data

    class RunEvidence(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and torch.cuda.is_available():
                logs["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                logs["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30

        def on_save(self, args, state, control, **kwargs):
            # All checkpoints in this new run belong to this training job.
            folders = sorted(Path(args.output_dir).glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]))
            keep = {str(p.resolve()) for p in folders[-2:]}
            if state.best_model_checkpoint:
                keep.add(str(Path(state.best_model_checkpoint).resolve()))
            for path in folders:
                if str(path.resolve()) not in keep:
                    shutil.rmtree(path)

    class LabTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(RunEvidence())
            trainable = {name: p.numel() for name, p in self.model.named_parameters() if p.requires_grad}
            if not trainable or any("lora_" not in name for name in trainable):
                raise ValueError("This recipe must train LoRA parameters only")
            out = Path(self.args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "trainable_parameters.json").write_text(json.dumps({
                "trainable_count": sum(trainable.values()),
                "trainable_names": list(trainable),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "distance_labels": "weak_command_integral_not_measured_odometry",
                "terminal_success": "user_confirmed_2026-11-09",
            }, indent=2) + "\n")
            shutil.copyfile(dataset_root / "dataset_report.json", out / "dataset_report.json")
            sample = self.train_dataset[0]
            target = sample["labels"][sample["labels"] != -100]
            if not target.numel():
                raise ValueError("Tokenization masked every target token")
            (out / "sample_token_audit.json").write_text(json.dumps({
                "input_tokens": sample["input_ids"].numel(),
                "supervised_tokens": target.numel(),
                "decoded_supervision": self.tokenizer.decode(target.tolist()),
            }, indent=2) + "\n")

        def _get_eval_sampler(self, eval_dataset):
            # Sequential evaluation includes every held-out sample exactly once.
            return SequentialSampler(eval_dataset)

        def train(self, *args, **kwargs):
            baseline = self.evaluate(metric_key_prefix="baseline_validation")
            self.save_metrics("baseline_validation", baseline)
            result = super().train(*args, **kwargs)
            # load_best_model_at_end selects by validation loss only.
            validation = self.evaluate(metric_key_prefix="best_validation")
            self.save_metrics("best_validation", validation)
            test = self.evaluate(eval_dataset=held_out["test"], metric_key_prefix="heldout_test")
            self.save_metrics("heldout_test", test)
            self.save_metrics("train", result.metrics)
            evidence = {"best_model_checkpoint": self.state.best_model_checkpoint,
                        "best_validation_loss": self.state.best_metric,
                        "global_step": self.state.global_step,
                        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                        "nonzero_lora_b_tensors": sum(int(torch.count_nonzero(p).item() > 0)
                            for name, p in self.model.named_parameters() if "lora_B" in name),
                        "baseline_validation": baseline, "final_validation": validation, "heldout_test": test,
                        "evaluation_scope": "offline teacher-forced loss; not real-robot navigation success"}
            (Path(self.args.output_dir) / "run_summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
            return result

    module.make_supervised_data_module = lab_data
    module.LLaVATrainer = LabTrainer
    sys.argv = [sys.argv[0], *training_arguments]
    # The official forward unconditionally all-reduces this factor. With one
    # process it is identically 1, so no distributed process group is needed.
    checkpoint = torch.utils.checkpoint.checkpoint
    with (
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("llava.model.language_model.llava_llama.calculate_loss_weight",
                   new=lambda labels, ignore_index=-100: 1.0),
        mock.patch("torch.utils.checkpoint.checkpoint", new=partial(checkpoint, use_reentrant=False)),
    ):
        module.train()


if __name__ == "__main__":
    main()
