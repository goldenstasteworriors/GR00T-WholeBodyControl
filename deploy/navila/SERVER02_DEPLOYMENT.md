# server02 NaVILA inference deployment

This deployment runs the high-level NaVILA-8B model (FP16, eight frames).
It never connects to G1 or publishes robot commands. The separate ThinkPad
project provides camera input, preview-only output, and optional WBC integration.

## Paths

- Code: `/home/user/ykj/project/NaVILA`
- Storage, weights, environment, cache, logs: `/data/ykj/navila`
- Python: `/data/ykj/navila/envs/navila/bin/python`
- Model: `/data/ykj/navila/models/navila-llama3-8b-8f`

## Start and access

Default: GPU 0, FP16, TCP `127.0.0.1:54321`.
Hugging Face and Transformers run in offline mode. No OpenAI API is used.

One-command startup on server02, as one complete line:

```bash
bash /home/user/ykj/project/NaVILA/deploy_server02/one_click.sh
```

The existing `navila-server02` tmux session is reused.
To run in the foreground instead:

```bash
bash /home/user/ykj/project/NaVILA/deploy_server02/serve.sh
```

View logs:

```bash
tail -f /data/ykj/navila/logs/server.log
```

Stop this deployment:

```bash
tmux kill-session -t navila-server02
```

The current workstation relay connects ThinkPad's loopback port 54321 to
this server. It uses existing SSH identities. The workstation must remain
online. The server itself does not start a robot controller.

`NAVILA_GPU`, `NAVILA_HOST`, and `NAVILA_PORT` can override the defaults.
The recorded deployment tests used the defaults.

## Protocol and client

Request and response framing: an eight-byte big-endian JSON byte length,
followed by UTF-8 JSON. Requests contain eight base64 JPEG/PNG images in
`images` and an English instruction in `query`.

A normal response is a JSON string containing the raw action text.
Errors are JSON objects with an `error` field. The existing server accepts
requests up to 32 MiB and handles one request per connection, sequentially.

The client only prints responses. This example is one complete command:

```bash
/data/ykj/navila/envs/navila/bin/python /home/user/ykj/project/NaVILA/deploy_server02/client.py --instruction 'Walk through the small room, turn the corner, enter the kitchen, and stop near the long wooden dining table.' --images /data/ykj/navila/validation/frame_0{0,1,2,3,4,5,6,7}.jpg --repeat 4
```

Use uniformly sampled navigation history plus the current frame. Initial
history is front-padded with black frames. A sequence of static images is
not evidence that predicted actions were executed.

For output-only tests with the saved G1 image or live camera, use
`deploy/navila/preview.sh` in the ThinkPad GR00T project. It does not create
a robot controller or ROS motion publisher.

## Reproducibility

Python 3.10; torch 2.3.0+cu121; torchvision 0.18.0; transformers 4.37.2;
FlashAttention 2.5.8; DeepSpeed 0.9.5. System drivers and CUDA were not changed.
See `conda-explicit.txt`, `requirements.lock.txt`, and `PROVENANCE.md`.

Installation followed the official environment setup and replacement files.
FlashAttention uses the official cu122torch2.3 CPython 3.10 wheel and passed
GPU inference tests. Habitat/Isaac Sim evaluation and SFT training were not
installed or verified as part of this inference deployment.

The lock file alone is insufficient to recreate the patched environment:
copy these repository directories into the new environment's package paths:
- `llava/train/transformers_replace/` into `site-packages/transformers/`
- `llava/train/deepspeed_replace/` into `site-packages/deepspeed/`

The official decord 0.6.0 wheel contains cp36-cp36m metadata, so pip 25.3
check reports a platform mismatch. Python 3.10 import and eight-frame MP4
decoding passed. Metadata was not changed to conceal the warning.
This server uses PIL to decode request images.

The server is adapted from the official NaVILA-Bench server, with its MIT
license preserved. Deployment changes include loopback binding, complete
header reads, request/frame validation, recovery after request errors,
explicit attention mask/pad token, a 64-token generation limit, and CUDA
timing/memory reporting. These are server details, not robot control logic.

## Verification and backups

Records: `/data/ykj/navila/validation` and `/data/ykj/navila/logs`.

Verified: sizes of all 18 model files, SHA-256 of six safetensors files,
core imports, CUDA computation, decord decoding, offline model loading,
eight-frame inference, and SSH access. Official sample GIF images were
cropped to remove maps/subtitles for pipeline checks.

The separate ThinkPad adapter also passed real-camera preview and real
joint-state read-only tests. These do not establish navigation success,
beaker recognition accuracy, stopping precision, or obstacle avoidance.

Branch: `codex/server02-deploy-01a19ca8`.
Session: `01a19ca8-ac6e-7460-a4da-6a193efe13ca`.
The upstream GitHub repository denies this account write access. Changes
are committed locally and backed up as Git bundles under `/data/ykj/navila`.
The one-command launcher and this document also have copies in the user's
GR00T repository under `deploy/navila/`.
