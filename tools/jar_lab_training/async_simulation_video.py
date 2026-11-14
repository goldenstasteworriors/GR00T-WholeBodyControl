"""Wall-clock asynchronous JAR inference with a continuously ticking MuJoCo loop.

No robot transport. Physics runs at 30 control Hz while a worker computes JAR
predictions. Rendering happens afterwards from recorded simulator qpos, so the
renderer cannot slow inference or pause the controller during this experiment.
"""
import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim_physics import ArmPhysics
from simulation_video import Policy, RecordedRGB, VideoCanvas, TASKS, BG, BLUE, WHITE


class ReferenceBuffer:
    """Explicit distinction between slow references and time-aligned trajectories."""
    def __init__(self, initial, mode, max_age=.75):
        if mode not in ("latest", "aligned32", "stream32"):raise ValueError(mode)
        self.mode,self.max_age=mode,max_age
        self.held=np.asarray(initial,dtype=float).copy()
        self.current=None
        self.accepted=self.expired=self.skipped=0

    def accept(self, result, frame, now):
        positions=np.asarray(result["positions"],dtype=float)
        horizon=32 if self.mode=="aligned32" else 1
        if positions.shape!=(horizon,16) or not np.isfinite(positions).all():
            raise ValueError("Invalid asynchronous prediction")
        # Streaming retimes each decoded row as an arrival-paced reference.
        # Preserve observation/forecast ages separately; never claim 30Hz alignment.
        age=now-result["ready" if self.mode=="stream32" else "captured"]
        skip=max(0,frame-result["source_frame"])
        if age<0:raise ValueError("Result arrived before its observation")
        if (self.mode in ("latest","stream32") and age>self.max_age) or (self.mode=="aligned32" and skip>=horizon):
            self.expired+=1
            self.skipped+=horizon
            return False
        self.current=result
        self.accepted+=1
        self.skipped+=skip if self.mode=="aligned32" else 0
        return True

    def sample(self, frame, now):
        if self.current is None:return self.held.copy(),-1,0,"hold: predictions expired" if self.expired else "waiting"
        r=self.current
        if self.mode in ("latest","stream32"):
            if now-r["ready" if self.mode=="stream32" else "captured"]>self.max_age:
                return self.held.copy(),r["source_frame"],0,"hold: stale reference"
            index=0
        else:
            index=frame-r["source_frame"]
            if index<0:raise ValueError("Future observation")
            if index>=len(r["positions"]):return self.held.copy(),r["source_frame"],0,"hold: buffer empty"
        self.held=np.asarray(r["positions"][index],dtype=float).copy()
        remaining=len(r["positions"])-index
        return self.held.copy(),r["source_frame"],remaining,"buffer" if self.mode=="aligned32" else "reference"


class MemoryRGB:
    def __init__(self, source, frames):
        self.images=[source.frame(i).copy() for i in range(frames)]

    def frame(self, index):return self.images[index]


def benchmark(policy, rgb, timing, initial, instruction):
    import torch
    frame=min(24,len(rgb.images)-1)
    policy.predict(initial,rgb,timing,instruction,frame)
    results={}
    tokens={}
    for horizon in (1,8,32):
        seconds=[]
        for _ in range(3):
            _,ids,latency=policy.predict(initial,rgb,timing,instruction,frame,horizon=horizon)
            seconds.append(latency);tokens[horizon]=ids
        results[str(horizon)]=dict(samples_seconds=seconds,median_seconds=float(np.median(seconds)),
                                  covered_seconds=horizon/30)
    assert np.array_equal(tokens[1],tokens[32][:16])
    assert np.array_equal(tokens[8],tokens[32][:128])
    return dict(device=torch.cuda.get_device_name(policy.device),checkpoint=policy.choice,
                history_frame=frame,results=results,prefix_equivalence=True,robot_io=False)


def rollout(args, task):
    policy=Policy(args.training_root,args.cache,task,args.device,args.memory_fraction)
    rows=[r for r in policy.manifest["rows"] if r["episode"] in policy.manifest["validation"]]
    row=rows[0]
    with np.load(args.cache/"episodes"/str(row["episode"])/"observations.npz") as data:
        numeric={k:data[k] for k in data.files}
    initial=numeric["state"][0].copy()
    frames=min(row["length"],round(args.seconds*30))
    source=RecordedRGB(args.cache,row["episode"],row["length"])
    try:rgb=MemoryRGB(source,frames)
    finally:source.close()
    timing={k:numeric[k] for k in ("camera_time","timestamp")}
    physics=ArmPhysics(args.xml,initial,dex1_urdf=args.dex1_urdf)
    output=args.output/task;output.mkdir(parents=True,exist_ok=False)
    if args.benchmark:
        measured=benchmark(policy,rgb,timing,initial,row["instruction"])
        (output/"benchmark.json").write_text(json.dumps(measured,indent=2)+"\n")
        print(json.dumps(dict(event="benchmark",task=task,**measured)),flush=True)
    else:policy.predict(initial,rgb,timing,row["instruction"],0)
    stream_verification=None
    if args.mode=="stream32":
        check_frame=min(24,frames-1)
        predicted,ids,_=policy.predict(initial,rgb,timing,row["instruction"],check_frame,horizon=args.verify_rows)
        emitted=list(policy.stream(initial,rgb,timing,row["instruction"],check_frame,horizon=args.verify_rows))
        np.testing.assert_array_equal(np.concatenate([e["tokens"] for e in emitted]),ids)
        np.testing.assert_allclose(np.concatenate([e["positions"] for e in emitted]),predicted,rtol=0,atol=1e-7)
        stream_verification=dict(passed=True,rows=args.verify_rows,tokens=args.verify_rows*16,
                                 first_row_seconds=emitted[0]["elapsed_seconds"],
                                 total_seconds=emitted[-1]["elapsed_seconds"],
                                 median_row_interval_seconds=float(np.median([e["interval_seconds"] for e in emitted])))
        (output/"stream-verification.json").write_text(json.dumps(stream_verification,indent=2)+"\n")
        print(json.dumps(dict(event="stream_verified",task=task,**stream_verification)),flush=True)
    horizon=1 if args.mode=="latest" else 32
    stopped=threading.Event();condition=threading.Condition()
    snapshot=None;completed=queue.Queue();requests=[];errors=[]
    def infer():
        last_frame=-1;chunk=0
        try:
            while not stopped.is_set():
                with condition:
                    condition.wait_for(lambda:stopped.is_set() or (snapshot is not None and snapshot[0]>last_frame))
                    if stopped.is_set():break
                    frame,state,captured=snapshot
                last_frame=frame
                if args.mode=="stream32":
                    generator=policy.stream(state,rgb,timing,row["instruction"],frame,horizon=32)
                    try:
                        for emitted in generator:
                            if stopped.is_set():break
                            result=dict(source_frame=frame,input_state=state.tolist(),captured=captured,
                                        ready=time.monotonic(),inference_seconds=emitted["interval_seconds"],
                                        positions=emitted["positions"].tolist(),chunk_id=chunk,
                                        generated_row=emitted["row"],prediction_frame=frame+emitted["row"]+1,
                                        stream_elapsed_seconds=emitted["elapsed_seconds"],tokens=emitted["tokens"].tolist())
                            requests.append(result);completed.put(result)
                    finally:generator.close()
                    chunk+=1
                    continue
                predicted,_,latency=policy.predict(state,rgb,timing,row["instruction"],frame,horizon=horizon)
                result=dict(source_frame=frame,input_state=state.tolist(),captured=captured,
                            ready=time.monotonic(),inference_seconds=latency,positions=predicted.tolist())
                requests.append(result);completed.put(result)
        except BaseException as error:
            errors.append(repr(error));stopped.set()
    worker=threading.Thread(target=infer,name="jar-inference")
    buffer=ReferenceBuffer(initial,args.mode)
    inputs=[];integrated=[];targets=[];qpos=[];ticks=[];lateness=[];origins=[];remaining=[];phases=[];latencies=[]
    applied=[];stream_rows=[];prediction_frames=[];start=time.monotonic();last_latency=0.
    worker.start()
    try:
        for frame in range(frames):
            deadline=start+frame/30
            stopped.wait(max(0.,deadline-time.monotonic()))
            if errors:raise RuntimeError(errors[0])
            now=time.monotonic();state=physics.state().astype(np.float32)
            with condition:
                snapshot=(frame,state.copy(),now);condition.notify()
            while True:
                try:r=completed.get_nowait()
                except queue.Empty:break
                accepted=buffer.accept(r,frame,time.monotonic())
                applied.append(dict(source_frame=r["source_frame"],received_frame=frame,accepted=accepted,
                                    latency_seconds=r["ready"]-r["captured"],
                                    chunk_id=r.get("chunk_id"),generated_row=r.get("generated_row"),
                                    prediction_frame=r.get("prediction_frame")))
                last_latency=r["inference_seconds"]
            target,origin,left,phase=buffer.sample(frame,time.monotonic())
            inputs.append(state);qpos.append(physics.data.qpos.copy());targets.append(target)
            ticks.append(now-start);lateness.append(now-deadline);origins.append(origin);remaining.append(left)
            phases.append(phase);latencies.append(last_latency)
            current=buffer.current or {}
            stream_rows.append(current.get("generated_row",-1));prediction_frames.append(current.get("prediction_frame",-1))
            physics.advance(target);integrated.append(physics.state())
        stopped.wait(max(0.,start+frames/30-time.monotonic()))
    finally:
        stopped.set()
        with condition:condition.notify_all()
        worker.join(timeout=60)
        if worker.is_alive():raise RuntimeError("Inference did not finish during shutdown")
        policy.close()
    if errors:raise RuntimeError(errors[0])
    inputs,integrated,targets=np.asarray(inputs),np.asarray(integrated),np.asarray(targets)
    np.testing.assert_allclose(inputs[1:],integrated[:-1],rtol=0,atol=3e-7)
    for r in requests:np.testing.assert_array_equal(np.asarray(r["input_state"],dtype=np.float32),inputs[r["source_frame"]])
    for value in (inputs,integrated,targets,qpos):
        if not np.isfinite(value).all():raise RuntimeError("Nonfinite asynchronous trajectory")
    reference=numeric["state"][:frames]
    np.savez_compressed(output/"rollout.npz",model_input_state=inputs,integrated_next_state=integrated,
                        executed_reference=targets,recorded_reference=reference,simulator_qpos=qpos,
                        tick_wall_seconds=ticks,tick_lateness_seconds=lateness,source_frame=origins,
                        buffer_remaining=remaining,generated_row=stream_rows,prediction_frame=prediction_frames)
    for r in requests:
        r["captured"]-=start;r["ready"]-=start
    (output/"requests.json").write_text(json.dumps(dict(requests=requests,arrivals=applied),indent=2)+"\n")
    result=dict(passed=True,task=task,episode=row["episode"],mode=args.mode,frames=frames,
                simulated_seconds=frames/30,control_wall_seconds=ticks[-1]+1/30,
                checkpoint=policy.choice,inference_horizon=horizon,inference_requests=len(requests),
                accepted_requests=buffer.accepted,expired_requests=buffer.expired,skipped_rows=buffer.skipped,
                active_frames=sum(p in ("reference","buffer") for p in phases),
                median_inference_seconds=float(np.median([r["inference_seconds"] for r in requests])) if requests else None,
                max_control_lateness_seconds=float(max(lateness)),p99_control_lateness_seconds=float(np.quantile(lateness,.99)),
                arms_mae_rad=float(np.abs(inputs[:,:14]-reference[:,:14]).mean()),
                grippers_mae_rad=float(np.abs(inputs[:,14:]-reference[:,14:]).mean()),
                recorded_state_input_frames=[0],feedback_verified=True,request_states_verified=True,
                stream_verification=stream_verification,
                streamed_rows=len(requests) if args.mode=="stream32" else None,
                stream_chunks=len({r["chunk_id"] for r in requests}) if args.mode=="stream32" else None,
                execution_semantics=("Each decoded row immediately updates a slow reference, retimed to arrival; persistent KV within 32 rows. Observe fresh state/images only between chunks. Not a time-aligned 30Hz forecast."
                                     if args.mode=="stream32" else args.mode),
                physics=physics.summary(),robot_io=False,
                interpretation="Real concurrent inference and 30Hz control clock; render afterwards from simulator qpos. Fixed dataset RGB, no task-object success evaluation.")
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    import mujoco,imageio_ffmpeg
    canvas=VideoCanvas(physics,args.font)
    writer=imageio_ffmpeg.write_frames(str(output/"rollout.mp4"),(1920,1080),fps=30,codec="libx264",
        pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=6,macro_block_size=1,
        output_params=["-preset","veryfast","-threads","4","-movflags","+faststart"],ffmpeg_log_level="error")
    writer.send(None)
    try:
        for frame in range(frames):
            # Replay achieved simulator qpos for rendering only, after inference
            # has stopped. No recorded state can enter the closed feedback loop.
            physics.data.qpos[:]=qpos[frame];physics.data.qvel[:]=0
            mujoco.mj_forward(physics.model,physics.data);physics.set_reference(reference[frame])
            view=Image.fromarray(canvas.frame(rgb.frame(frame),task,row["episode"],frame,inputs[frame],
                reference[frame],targets[frame],latencies[frame],policy.choice["nll"]))
            draw=ImageDraw.Draw(view);draw.rectangle((0,0,1040,65),fill=BG)
            title={"latest":"latest reference","aligned32":"32-frame aligned buffer","stream32":"STREAM | emit each 16 tokens"}[args.mode]
            draw.text((24,18),"JAR "+title,font=canvas.fonts[34],fill=WHITE)
            draw.rectangle((0,1018,1920,1080),fill=BG)
            if args.mode=="stream32":
                observation_age=(frame-origins[frame])/30 if origins[frame]>=0 else 0
                status=f"30Hz physics | AR row {stream_rows[frame]+1}/32 | observation #{origins[frame]} age {observation_age:.2f}s | forecast frame #{prediction_frames[frame]} -> control #{frame}"
                footer="Arrival-paced position references, not a time-aligned 30Hz trajectory. Recorded RGB does not react to motion. No robot I/O."
            else:
                status=f"30Hz physics | input #{frame} | target from #{origins[frame]} | {phases[frame]} | buffer {remaining[frame]} | expired requests {sum(not a['accepted'] and a['received_frame']<=frame for a in applied)}"
                footer="Real inference waiting is included. Recorded images do not react to motion. No robot I/O or object-success evaluation."
            draw.text((24,1022),status,font=canvas.fonts[22],fill=BLUE)
            draw.text((24,1056),footer,font=canvas.fonts[18],fill=WHITE)
            if frame==0:view.save(output/"first_frame.png")
            writer.send(np.asarray(view))
        view.save(output/"last_frame.png")
    finally:writer.close();canvas.close()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-root",type=Path,required=True);p.add_argument("--cache",type=Path)
    p.add_argument("--xml",type=Path,required=True);p.add_argument("--dex1-urdf",type=Path,required=True)
    p.add_argument("--font",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tasks",nargs="+",choices=TASKS,default=TASKS)
    p.add_argument("--mode",choices=("latest","aligned32","stream32"),default="latest")
    p.add_argument("--verify-rows",type=int,choices=(8,32),default=8)
    p.add_argument("--seconds",type=float,default=15);p.add_argument("--benchmark",action="store_true")
    p.add_argument("--device",default="cuda:0");p.add_argument("--memory-fraction",type=float,default=.25)
    args=p.parse_args()
    if not 0<args.seconds<=60 or not 0<args.memory_fraction<=.3:raise ValueError("Invalid execution budget")
    if args.cache is None:args.cache=args.training_root/"cache"
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[rollout(args,task) for task in args.tasks]
    import imageio_ffmpeg
    listing=args.output/"concat.txt"
    listing.write_text("".join(f"file '{(args.output/t/'rollout.mp4').resolve()}'\n" for t in args.tasks))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-hide_banner","-loglevel","error","-f","concat","-safe","0",
                    "-i",str(listing),"-c","copy","-movflags","+faststart",str(args.output/"all_operations.mp4")],check=True)
    (args.output/"summary.json").write_text(json.dumps(dict(passed=True,operations=summaries),indent=2)+"\n")


if __name__=="__main__":main()
