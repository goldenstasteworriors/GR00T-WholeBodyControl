"""Recorded RGB + simulated proprioception rollouts, with zero robot transport.

Only reset uses the recorded initial state. Subsequent model states come from
MuJoCo arms and Dex1 dynamics (or the explicit legacy ideal servo). Recorded states are
used in the reference panel/metrics, never as subsequent policy inputs.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_physics import ArmPhysics, ARMS, JOINTS


TASKS = ["TAKE_TUBE", "TAKE_DROPPER_AND_ASPIRATE", "DISPENSE_TO_HELD_TUBE",
         "RETURN_DROPPER", "RETURN_TUBE_TO_RACK", "PLACE_TUBE_IN_BATH"]
BG, WHITE, MUTED, BLUE, GREEN, ORANGE = "#101722", "#edf2f8", "#a8b6c8", "#6ab8f5", "#9cce8b", "#efbe71"


class IdealDex1:
    """Native-angle state with a 0.5rad/s ideal servo; no jaw geometry/contact."""
    def __init__(self, initial):
        self.q = np.asarray(initial, dtype=float).copy()
        if self.q.shape != (2,) or not np.isfinite(self.q).all() or np.any(self.q < 0) or np.any(self.q > 5.4):
            raise ValueError("Invalid initial Dex1 native angles")

    def advance(self, target, dt=1/30):
        self.q += np.clip(np.clip(target, 0., 5.4)-self.q, -.5*dt, .5*dt)


class RecordedRGB:
    def __init__(self, cache, episode, length):
        folder = Path(cache)/"episodes"/str(episode)
        self.fd = os.open(folder/"rgb.lz4", os.O_RDONLY)
        self.offsets = np.load(folder/"rgb-offsets.npy")
        if len(self.offsets) != length+1:
            raise ValueError("RGB frame count differs from the observation clock")
        self.frames = {}

    def frame(self, index):
        from jar.mixed_cache import decode_rgb
        if index not in self.frames:
            a,b = map(int,self.offsets[index:index+2])
            self.frames[index] = decode_rgb(os.pread(self.fd,b-a,a),240)
            if len(self.frames)>32:
                del self.frames[next(iter(self.frames))]
        return self.frames[index]

    def close(self):
        os.close(self.fd)


class Policy:
    def __init__(self, training, cache, task, device, memory_fraction):
        import torch
        from jar.olmo2 import load_olmo2_bundle
        from jar.uniform_action_tokenizer import OLMo2UniformTokenizer
        from jar.data import build_prefix, history_indices
        from jar.uniform_inference import generate_uniform_chunk
        self.torch,self.build_prefix,self.history_indices,self.generate = torch,build_prefix,history_indices,generate_uniform_chunk
        self.manifest = json.loads((Path(cache)/task/"manifest.json").read_text())
        if self.manifest["joint_names"] != JOINTS or self.manifest["source_indices"] != list(range(15,31)):
            raise ValueError("Unexpected observation joint order")
        choice = json.loads((Path(training)/"runs"/task/"best_validation.json").read_text())
        self.choice = choice
        weight_path=Path(choice["path"])/"raw_model.pt"
        # The shared filesystem is fast for sequential reads but slow for the
        # many small mmap page faults triggered by model.to(cuda).
        started=time.monotonic();buffer=bytearray(32*1024*1024)
        with weight_path.open("rb",buffering=0) as stream:
            while stream.readinto(buffer):pass
        del buffer
        print(json.dumps(dict(event="checkpoint_prefetched",task=task,seconds=time.monotonic()-started)),flush=True)
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.matmul.allow_tf32=True
        torch.set_num_threads(4)
        self.model,self.config = load_olmo2_bundle(weight_path,device=device)
        self.model.eval()
        if self.config.state_dim != 16 or self.config.action_dim != 16:
            raise ValueError("Expected the unchanged 16D OLMo2 checkpoint")
        self.tokenizer=OLMo2UniformTokenizer(self.config.text_tokenizer_path,bins=512,action_dim=16)
        self.low,self.high=(np.array(self.manifest["normalization"][k],dtype=np.float32) for k in ("low","high"))
        self.prefixes={}

    def predict(self, state, rgb, timing, instruction, frame, horizon=1):
        torch=self.torch
        from dataset import normalize
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(16,) or not np.isfinite(state).all():
            raise ValueError("Policy state must be the finite simulated 16D state")
        history=self.history_indices(frame,self.config.history_frames,self.config.history_stride)
        pictures=np.stack([rgb.frame(int(index)) for index in history])
        # [history,camera,H,W,RGB] -> the exact training [history*camera,RGB,H,W].
        pictures=pictures.transpose(0,1,4,2,3).reshape(-1,3,240,240).copy()
        if set(timing)!={"camera_time","timestamp"}:raise ValueError("Only frame timing enters the policy clock")
        ages=timing["camera_time"][history].reshape(-1)-timing["timestamp"][frame]
        if np.any(ages>1e-7):raise ValueError("Future image leakage")
        text=f"Robot: Unitree G1 with Dex1 grippers. Control rate: 30 Hz. {instruction}"
        key=text,len(pictures)
        if key not in self.prefixes:
            self.prefixes[key]=self.build_prefix(self.tokenizer,self.config,text,len(pictures))
        prefix=self.prefixes[key]
        obs=dict(state=torch.from_numpy(normalize(state,self.manifest["normalization"]))[None].to(self.device),
            images=torch.from_numpy(pictures)[None].to(self.device),
            image_positions=prefix["image_positions"][None].to(self.device),
            state_positions=prefix["state_positions"][None].to(self.device),
            camera_ids=torch.arange(3,device=self.device).repeat(len(history))[None],
            frame_times=torch.tensor(np.minimum(ages,0),dtype=torch.float32,device=self.device)[None])
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        started=time.monotonic()
        with torch.inference_mode(),torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            values,tokens=self.generate(self.model,prefix["input_ids"][None].to(self.device),obs,self.tokenizer,horizon=horizon)
        if self.device.type=="cuda":torch.cuda.synchronize(self.device)
        seconds=time.monotonic()-started
        physical=(values+1)*(self.high-self.low)/2+self.low
        if not np.isfinite(physical).all():raise RuntimeError("Nonfinite predicted position")
        return physical.astype(float),tokens[0].cpu().numpy(),seconds

    def close(self):
        del self.model
        gc.collect()
        if self.device.type=="cuda":self.torch.cuda.empty_cache()


class VideoCanvas:
    def __init__(self, physics, font):
        import mujoco
        self.physics,self.mujoco=physics,mujoco
        self.renderer=mujoco.Renderer(physics.model,height=800,width=560)
        self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=[.15,0.,1.13] if physics.dex1 else [.08,0.,.75]
        self.camera.distance,self.camera.azimuth,self.camera.elevation=(1.55,135.,-13.) if physics.dex1 else (2.25,10.,-12.)
        self.view=mujoco.MjvOption();self.view.geomgroup[3:]=0
        self.fonts={size:ImageFont.truetype(str(font),size) for size in (18,22,26,34,38)}

    def frame(self, images, task, episode, index, state, truth, previous_target, latency, nll):
        image=Image.new("RGB",(1920,1080),BG);draw=ImageDraw.Draw(image)
        def text(x,y,value,size=22,color=WHITE):draw.text((x,y),value,font=self.fonts[size],fill=color)
        text(24,18,"JAR  |  simulated-state rollout",38)
        text(24,72,task.replace("_"," "),26,BLUE)
        text(1080,29,f"HELD-OUT EPISODE {episode}     {index/30:5.2f} s",26)
        text(1080,71,f"1h validation-best NLL {nll:.4f}   |   offline 30 fps playback",22,MUTED)
        labels=["EGO / recorded RGB","LEFT WRIST","RIGHT WRIST"]
        for camera in range(3):
            x=24+camera*248;text(x,126,labels[camera],18,MUTED)
            image.paste(Image.fromarray(images[camera]).resize((232,232)),(x,158))
        text(24,426,"Model input",26)
        for y,value in zip((470,506,542,578),(
            "3-view causal RGB history + fixed English task text",
            "Arms: integrated MuJoCo qpos (14D)",
            "Dex1: integrated jaw positions -> motor angles (2D)" if self.physics.dex1 else "Dex1: ideal rate-limited servo state (2D)",
            "Only reset uses recorded state; then feedback is simulated")):
            text(24,y,value,22,MUTED)
        text(24,638,"Dex1 native angle   /   simulated sliders, approximate mapping" if self.physics.dex1 else
             "Dex1 native angle   /   ideal state, no jaw contacts",22)
        for side in range(2):
            y=686+side*112
            text(24,y,f"{'LEFT' if side==0 else 'RIGHT'}   sim {state[14+side]:.3f} rad    recorded {truth[14+side]:.3f} rad",22)
            x0,y0,w=24,y+42,690
            draw.rounded_rectangle((x0,y0,x0+w,y0+18),radius=5,fill="#2c3a4c")
            draw.rounded_rectangle((x0,y0,x0+w*np.clip(state[14+side]/5.4,0,1),y0+18),radius=5,fill=BLUE)
            for value,color in ((truth[14+side],GREEN),(previous_target[14+side],ORANGE)):
                x=x0+w*np.clip(value/5.4,0,1);draw.line((x,y0-5,x,y0+23),fill=color,width=3)
        text(24,929,f"Arm pose MAE vs recorded: {np.abs(state[:14]-truth[:14]).mean():.3f} rad",22)
        text(24,970,f"Last inference: {latency*1000:.0f} ms  |  arm reference cap 0.25 rad/s",22,MUTED)
        text(776,126,"MODEL -> PHYSICS",26,BLUE);text(1352,126,"RECORDED OBSERVATION POSE",26,GREEN)
        for x,data in ((776,self.physics.data),(1352,self.physics.reference)):
            self.renderer.update_scene(data,camera=self.camera,scene_option=self.view)
            image.paste(Image.fromarray(self.renderer.render()),(x,168))
        text(776,985,"Fixed pelvis / legs / waist. Arms have gravity, inertia and bounded PD.",22,MUTED)
        text(24,1033,"Offline state-feedback check: recorded images do not react to simulated motion. No object/task-success simulation.",22,MUTED)
        return np.asarray(image)

    def close(self):self.renderer.close()


def rollout(args, task):
    import imageio_ffmpeg
    policy=Policy(args.training_root,args.cache,task,args.device,args.memory_fraction)
    rows=[r for r in policy.manifest["rows"] if r["episode"] in policy.manifest["validation"]]
    row=rows[0]
    folder=Path(args.cache)/"episodes"/str(row["episode"])
    with np.load(folder/"observations.npz") as loaded:numeric={k:loaded[k] for k in loaded.files}
    timing={key:numeric[key] for key in ("camera_time","timestamp")}
    initial=numeric["state"][0].copy()
    if args.gripper_mode=="mujoco":
        if args.dex1_urdf is None:raise ValueError("MuJoCo grippers require the verified Dex1 URDF")
        physics=ArmPhysics(args.xml,initial,dex1_urdf=args.dex1_urdf);gripper=None
    else:
        physics=ArmPhysics(args.xml,initial[:14]);gripper=IdealDex1(initial[14:])
    rgb=RecordedRGB(args.cache,row["episode"],row["length"])
    canvas=VideoCanvas(physics,args.font)
    frames=min(row["length"],round(args.seconds*30))
    if args.max_frames:frames=min(frames,args.max_frames)
    output=args.output/task;output.mkdir(parents=True,exist_ok=True)
    writer=imageio_ffmpeg.write_frames(str(output/"rollout.mp4"),(1920,1080),fps=30,codec="libx264",
        pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=7,macro_block_size=1,
        output_params=["-preset","veryfast","-threads","4","-movflags","+faststart"],ffmpeg_log_level="error")
    writer.send(None)
    inputs,targets,actual,latencies=[],[],[],[]
    previous=initial.copy();previous_latency=0.;prefix_equal=None
    try:
        for frame in range(frames):
            state=(physics.state() if physics.dex1 else np.r_[physics.state(),gripper.q]).astype(np.float32)
            truth=numeric["state"][frame]
            physics.set_reference(truth if physics.dex1 else truth[:14])
            view=canvas.frame(rgb.frame(frame),task,row["episode"],frame,state,truth,previous,
                              previous_latency,policy.choice["nll"])
            if frame==0:Image.fromarray(view).save(output/"first_frame.png")
            writer.send(view)
            predicted,tokens,latency=policy.predict(state,rgb,timing,row["instruction"],frame)
            if frame==0 and not args.skip_prefix_check:
                _,full_tokens,_=policy.predict(state,rgb,timing,row["instruction"],frame,horizon=32)
                prefix_equal=bool(np.array_equal(tokens,full_tokens[:16]))
                if not prefix_equal:raise RuntimeError("One-frame AR prefix differs from full generation")
            target=predicted[0]
            inputs.append(state);targets.append(target);latencies.append(latency)
            if physics.dex1:
                physics.advance(target);actual.append(physics.state())
            else:
                physics.advance(target[:14]);gripper.advance(target[14:])
                actual.append(np.r_[physics.state(),gripper.q])
            previous,previous_latency=target,latency
            if frame%60==0:print(json.dumps(dict(task=task,episode=row["episode"],frame=frame,frames=frames,
                                               inference_ms=latency*1000)),flush=True)
        Image.fromarray(view).save(output/"last_frame.png")
    finally:
        writer.close();canvas.close();rgb.close();policy.close()
    inputs,targets,actual=np.asarray(inputs),np.asarray(targets),np.asarray(actual)
    # The next model input must be exactly the previous integrated state, not
    # an overwritten recorded state or the predicted position target.
    np.testing.assert_allclose(inputs[1:],actual[:-1],rtol=0,atol=3e-7)
    reference=numeric["state"][:frames]
    np.savez_compressed(output/"rollout.npz",model_input_state=inputs,prediction_target=targets,
                        integrated_next_state=actual,recorded_reference=reference,inference_seconds=latencies)
    result=dict(passed=True,task=task,episode=row["episode"],frames=frames,simulated_seconds=frames/30,
                checkpoint=policy.choice,gripper_model=physics.summary()["gripper"],
                model_inputs="causal dataset RGB, simulated arms14+Dex1x2, fixed English operation instruction",
                recorded_state_input_frames=[0],state_feedback_chain_verified=True,one_frame_prefix_equals_full=prefix_equal,
                arms_mae_rad=float(np.abs(inputs[:,:14]-reference[:,:14]).mean()),
                grippers_mae_rad=float(np.abs(inputs[:,14:]-reference[:,14:]).mean()),
                median_inference_seconds=float(np.median(latencies)),physics=physics.summary(),
                interpretation="Offline mixed-source input rollout. Images are fixed recorded observations, not rendered policy feedback; no object interaction success is measured.")
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--xml",type=Path,required=True)
    parser.add_argument("--font",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",choices=TASKS,default=TASKS)
    parser.add_argument("--seconds",type=float,default=15)
    parser.add_argument("--max-frames",type=int,default=0)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--memory-fraction",type=float,default=.25)
    parser.add_argument("--gripper-mode",choices=("ideal","mujoco"),required=True)
    parser.add_argument("--dex1-urdf",type=Path)
    parser.add_argument("--skip-prefix-check",action="store_true")
    args=parser.parse_args()
    if args.cache is None:args.cache=args.training_root/"cache"
    if not 0<args.seconds<=60 or not 0<args.memory_fraction<=.3:raise ValueError("Invalid video budget/GPU memory quota")
    args.output.mkdir(parents=True,exist_ok=True)
    summaries=[rollout(args,task) for task in args.tasks]
    import imageio_ffmpeg
    listing=args.output/"concat.txt"
    listing.write_text("".join(f"file '{(args.output/task/'rollout.mp4').resolve()}'\n" for task in args.tasks))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-hide_banner","-loglevel","error",
                    "-f","concat","-safe","0","-i",str(listing),"-c","copy","-movflags","+faststart",
                    str(args.output/"all_operations.mp4")],check=True)
    (args.output/"summary.json").write_text(json.dumps(dict(passed=True,operations=summaries),indent=2)+"\n")


if __name__=="__main__":main()
