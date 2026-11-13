"""Independent train/held-out NLL minima and bounded checkpoint retention."""
import json
from pathlib import Path
import re
import shutil


def lower_nll(previous, value, step):
    import math
    if not math.isfinite(value):
        raise ValueError("Checkpoint selection requires finite NLL")
    if previous is None or value < previous["nll"]:
        return dict(nll=float(value), step=int(step)), True
    return previous, False


def retained_steps(completed, best, best_train=None, *, compact=False, finished=False):
    winners = {int(x["step"]) for x in (best, best_train) if x and x["step"] >= 0}
    return winners if compact and finished else set(sorted(completed)[-2:]) | winners


def prune_checkpoints(output, best, best_train, *, compact=False, finished=False):
    output = Path(output)
    completed = {int(p.name[4:]): p for p in output.iterdir()
                 if re.fullmatch(r"step\d+", p.name) and p.is_dir()
                 and not p.is_symlink() and (p/"complete.json").is_file()}
    keep = retained_steps(completed, best, best_train, compact=compact, finished=finished)
    for step, folder in completed.items():
        if step not in keep:
            shutil.rmtree(folder)
        elif compact and (finished or step != max(completed)):
            # Only this opt-in experiment's optimizer state is compacted.
            trainer = folder/"trainer.pt"
            if trainer.is_file():
                trainer.unlink()
    retained = sorted(set(completed) & keep)
    if compact and finished:
        latest = output/"latest.json"
        final = json.loads(latest.read_text())
        (output/"last_training_checkpoint.json").write_text(json.dumps(dict(final, retained=final["step"] in keep), indent=2)+"\n")
        step = max(retained)
        latest.write_text(json.dumps(dict(step=step, path=str(completed[step]),
                                         meaning="latest retained NLL winner; training is complete"), indent=2)+"\n")
    return dict(retained_steps=retained, optimizer_steps=[] if compact and finished else
                [s for s in retained if (completed[s]/"trainer.pt").is_file()],
                finished=finished, compact=compact)
