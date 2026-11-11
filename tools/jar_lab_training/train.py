"""Distributed JAR training with a wall-clock budget and observation-only labels."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from jar.mixed_cache import atomic_json
from jar.mixed_data import mixed_pack_collate
from jar.olmo2 import OLMo2Config, OLMo2Model
from jar.olmo2_checkpoint import publish_checkpoint
from jar.olmo2_pretrained import initialize_official
from dataset import ObservationDataset, GlobalBatchSampler, validation_indices


STOP_REQUESTED = False


def stop_requested(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def inputs(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "metadata"}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    nll, count = 0., 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**inputs(batch, device), return_logits=False)
        nll += out["loss_sum"].item()
        count += out["num_target_tokens"].item()
    model.train()
    return dict(nll=nll / count, supervised_tokens=count, samples=len(loader.dataset),
                metric="held-out future observation teacher-forced full-vocabulary NLL; not execution success")


def save(model, optimizer, output, step, epoch, next_batch, best, elapsed, args, rank, world, signature, readback=False):
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               numpy=np.random.get_state(), python=random.getstate())
    states = [None] * world
    dist.all_gather_object(states, rng)
    if rank == 0:
        target = output / f"step{step}"
        publish_checkpoint(target, model.state_dict(),
            dict(optimizer=optimizer.state_dict(), step=step, epoch=epoch, next_batch=next_batch,
                 rng=states, world=world, best=best, elapsed_seconds=elapsed,
                 arguments=vars(args), dataset_signature=signature),
            dict(utc=utc(), step=step, epoch=epoch, next_batch=next_batch,
                 representation="uniform512-future-observation", horizon=32, dimensions=16,
                 first_target_offset=1, dataset_signature=signature))
        if readback:
            saved = torch.load(target / "raw_model.pt", map_location="cpu", weights_only=True, mmap=True)
            for name, tensor in model.state_dict().items():
                if not torch.equal(tensor.detach().cpu(), saved[name]):
                    raise RuntimeError("Checkpoint mismatch: " + name)
            trainer = torch.load(target / "trainer.pt", map_location="cpu", weights_only=False, mmap=True)
            if trainer["step"] != step or not trainer["optimizer"]["state"]:
                raise RuntimeError("Optimizer readback failed")
            atomic_json(output / "checkpoint-readback.json", dict(passed=True, step=step, optimizer_saved=True,
                                                                  tensors=len(saved), world=world))
            del saved, trainer
        atomic_json(output / "latest.json", dict(step=step, path=str(target)))
        if best["step"] == step:
            atomic_json(output / "best.json", dict(best, path=str(target)))
        completed = sorted((p for p in output.iterdir() if p.is_dir() and re.fullmatch(r"step\d+", p.name)
                            and (p / "complete.json").is_file()), key=lambda p: int(p.name[4:]))
        keep = {p.name for p in completed[-2:]} | {f"step{best['step']}"}
        for old in completed:
            if old.name not in keep:
                shutil.rmtree(old)
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=14400)
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke-test cap; zero means time budget only")
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resume")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_requested)
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    torch.manual_seed(6198)
    np.random.seed(6198)
    random.seed(6198)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = OLMo2Config(**json.loads((Path(args.cache) / "config.json").read_text()))
    assert config.state_dim == config.action_dim == 16
    output = Path(args.output).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "latest.json").exists() and not args.resume:
            raise FileExistsError("Use --resume for an existing output")
        atomic_json(output / "config.json", asdict(config))
    dist.barrier()
    train = ObservationDataset(args.cache, args.task, config, "train")
    validation = ObservationDataset(args.cache, args.task, config, "validation") if rank == 0 else None
    sampler = GlobalBatchSampler(len(train), args.global_batch, rank, world)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=mixed_pack_collate, pin_memory=True,
                        **(dict(persistent_workers=True, prefetch_factor=2) if args.workers else {}))
    vloader = None
    if rank == 0:
        cuts = validation_indices(validation)
        # Smoke verifies data and numerics without running the whole validation.
        if args.max_steps:
            cuts = cuts[:2]
        vloader = DataLoader(Subset(validation, cuts), batch_size=1, num_workers=2,
                            collate_fn=mixed_pack_collate, pin_memory=True, persistent_workers=True)
        atomic_json(output / "observation_representation.json", train.manifest)
        atomic_json(output / "training.json", dict(utc=utc(), arguments=vars(args), world=world,
                    train_episodes=len(train.rows), train_cuts=len(train), validation_episodes=len(validation.rows),
                    validation_cuts=len(cuts), steps_per_epoch=len(sampler), input_dim=16, target_dim=16,
                    target="strictly future observation.state; no action data read",
                    budget="14400 seconds per operation after initialization and initial validation; final checkpoint/evaluation extra",
                    parameters="FP32", compute="BF16", optimizer="AdamW beta=(0.9,0.95), WD=0.01, grad clip=1",
                    scheduler="20-step linear warmup then wall-clock cosine to 10% of peak LR",
                    checkpoint_retention="latest two plus best validation; step5 full readback",
                    name=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d") + "_" + args.task))
    with torch.device("meta"):
        model = OLMo2Model(config)
    model.to_empty(device=device)
    from transformers.models.olmo2.modeling_olmo2 import Olmo2RotaryEmbedding
    model.backbone.rotary_emb = Olmo2RotaryEmbedding(model.backbone.config, device=device)
    restored = None
    if args.resume:
        resume = Path(args.resume)
        if not (resume / "complete.json").exists():
            raise ValueError("Cannot resume an incomplete checkpoint")
        model.load_state_dict(torch.load(resume / "raw_model.pt", map_location="cpu", weights_only=True, mmap=True))
        restored = torch.load(resume / "trainer.pt", map_location="cpu", weights_only=False, mmap=True)
        if restored["world"] != world or restored["dataset_signature"] != train.manifest["signature"]:
            raise ValueError("Resume world/dataset mismatch")
        for key in ("global_batch", "lr", "warmup_steps", "max_seconds", "task", "compile"):
            if restored["arguments"][key] != vars(args)[key]:
                raise ValueError("Resume recipe differs: " + key)
    else:
        initial = initialize_official(model, args.official, config, verify=rank == 0)
        fingerprints = [None] * world
        dist.all_gather_object(fingerprints, initial["initial_parameter_sample_sha256"])
        if any(f != fingerprints[0] for f in fingerprints):
            raise RuntimeError("Ranks have different initial parameters")
        if rank == 0:
            atomic_json(output / "initialization.json", dict(initial, utc=utc(), world=world, ranks_match=True))
    if args.compile:
        model.compile_training_kernels()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and "embed" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.01), dict(params=no_decay, weight_decay=0.)],
                                  lr=args.lr, betas=(.9, .95), fused=True)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    step, start_epoch, skip, elapsed_before = 0, 0, 0, 0.
    best = dict(nll=1e30, step=-1)
    if restored is not None:
        optimizer.load_state_dict(restored["optimizer"])
        step, start_epoch, skip, elapsed_before, best = (restored[k] for k in ("step", "epoch", "next_batch", "elapsed_seconds", "best"))
        state = restored["rng"][rank]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        del restored
    if rank == 0 and not args.resume:
        initial_metrics = evaluate(model, vloader, device)
        append_json(output / "validation.jsonl", dict(step=0, utc=utc(), **initial_metrics))
        print(json.dumps(dict(event="initial_validation", **initial_metrics)), flush=True)
    dist.barrier()
    if rank == 0:
        print(json.dumps(dict(event="ready", utc=utc(), world=world, target_dim=16, train_cuts=len(train))), flush=True)
    probe = model.lm_head.weight[train.tokenizer.low_token_id:train.tokenizer.high_token_id + 1].detach().clone()
    started = time.monotonic()
    for epoch in range(start_epoch, 1000000):
        sampler.epoch = epoch
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < skip:
                continue
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            data = inputs(batch, device)
            local_tokens = int(data["labels"].ne(-100).sum())
            if local_tokens != sum(m["target_tokens"] for m in batch["metadata"]):
                raise RuntimeError("Target mask mismatch")
            totals = torch.tensor([local_tokens, len(batch["metadata"])], device=device, dtype=torch.int64)
            dist.all_reduce(totals)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ddp(**data, return_logits=False)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError("Nonfinite loss")
            (out["loss"] * (world * local_tokens / totals[0])).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
            step += 1
            elapsed = elapsed_before + time.monotonic() - started
            factor = min(1., step / max(1, args.warmup_steps))
            if step > args.warmup_steps:
                factor = .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., elapsed / args.max_seconds)))
            lr = args.lr * factor
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            loss_sum = out["loss_sum"].double()
            dist.all_reduce(loss_sum)
            torch.cuda.synchronize()
            elapsed = elapsed_before + time.monotonic() - started
            stopped = torch.tensor(int(STOP_REQUESTED or elapsed >= args.max_seconds or
                                        bool(args.max_steps and step >= args.max_steps)), device=device)
            dist.all_reduce(stopped, op=dist.ReduceOp.MAX)
            finished = bool(stopped.item())
            if rank == 0:
                record = dict(utc=utc(), step=step, epoch=epoch + (batch_index + 1) / len(sampler),
                              train_nll=float(loss_sum / totals[0]), lr=lr, grad_norm=norm.item(),
                              supervised_tokens=int(totals[0]), samples=int(totals[1]),
                              seconds=time.monotonic() - tick, elapsed_seconds=elapsed,
                              gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
                append_json(output / "metrics.jsonl", record)
                atomic_json(output / "status.json", dict(state="training", **record))
                print(json.dumps(record), flush=True)
                if step == 1:
                    delta = (model.lm_head.weight[train.tokenizer.low_token_id:train.tokenizer.high_token_id + 1] - probe).abs().max().item()
                    if not delta > 0:
                        raise RuntimeError("Model parameters did not update")
                    atomic_json(output / "first-step-check.json", dict(passed=True, finite_loss=True,
                                finite_grad_norm=True, maximum_bin_head_change=delta, world=world))
            improved = False
            if step % args.eval_every == 0 or finished:
                if rank == 0:
                    result = evaluate(model, vloader, device)
                    append_json(output / "validation.jsonl", dict(step=step, utc=utc(), **result))
                    if result["nll"] < best["nll"]:
                        best, improved = dict(nll=result["nll"], step=step), True
                payload = [best, improved]
                dist.broadcast_object_list(payload, src=0)
                best, improved = payload
            if step == 5 or step % args.save_every == 0 or improved or finished:
                save(model, optimizer, output, step, epoch, batch_index + 1, best,
                     elapsed_before + time.monotonic() - started, args, rank, world,
                     train.manifest["signature"], readback=step == 5)
            if finished:
                if rank == 0:
                    atomic_json(output / "status.json", dict(utc=utc(), state="interrupted" if STOP_REQUESTED else "complete",
                                step=step, elapsed_seconds=elapsed_before + time.monotonic() - started,
                                budget_seconds=args.max_seconds, best=best))
                dist.destroy_process_group()
                return
        skip = 0


if __name__ == "__main__":
    main()
