# Portable Wan2.2 TI2V-5B 720P Setup

This file records the environment and model setup used for the Wan2.2 TI2V-5B
720P baseline harness. It is intended for moving the run to another server with
idle GPU memory.

The tracked repository contains the reproducibility harness, prompt set,
configs, and logging scripts. It intentionally does not contain model weights,
generated videos, run logs, or the cloned Wan2.2 source tree.

## Hardware

- NVIDIA GPU with at least 24 GiB free VRAM for the official offload command.
- More free VRAM is recommended for the full `121` frame, `50` step baseline.
- CUDA driver new enough for the PyTorch build you install.

The failed smoke on `lsh-stable` was caused by an existing vLLM service holding
about 72 GiB of an 80 GiB A100, not by missing files.

## Clone This Repo

```bash
git clone git@github.com:Da1suKE66/WAN.git
cd WAN
```

If SSH is not configured, use HTTPS:

```bash
git clone https://github.com/Da1suKE66/WAN.git
cd WAN
```

## Create Python Environment

Use Python 3.10. The commands below keep the environment outside the git
checkout so it does not pollute the repository.

```bash
conda create -p /cache/llc/conda_envs/wan22 python=3.10 -y
conda activate /cache/llc/conda_envs/wan22

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the Wan2.1 repository dependencies and the small overlay packages
used by the baseline harness:

```bash
python -m pip install -r requirements.txt
python -m pip install -r experiments/wan_baseline/requirements_overlay.txt
python -m pip install -r experiments/wan22_ti2v5b/requirements_overlay_wan22.txt
```

If `flash_attn` fails to build from `requirements.txt`, install a wheel that
matches the selected PyTorch/CUDA/Python ABI, or remove it only if the selected
Wan command path does not import it on your machine.

## Clone Official Wan2.2 Source

The Wan2.2 source is kept under the experiment directory but is ignored by git.

```bash
mkdir -p experiments/wan22_ti2v5b/src
git clone https://github.com/Wan-Video/Wan2.2.git experiments/wan22_ti2v5b/src/Wan2.2

python -m pip install -r experiments/wan22_ti2v5b/src/Wan2.2/requirements.txt
python -m pip install -r experiments/wan22_ti2v5b/requirements_overlay_wan22.txt
```

The setup on `lsh-stable` used Wan2.2 source HEAD `42bf4cf fix readme`.

## Download Wan2.2 TI2V-5B Weights

Install the Hugging Face CLI and download to a cache or shared model directory.

```bash
python -m pip install -U "huggingface_hub[cli]"

mkdir -p /cache/llc/WAN-models
HF_ENDPOINT=https://hf-mirror.com \
  hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /cache/llc/WAN-models/Wan2.2-TI2V-5B
```

If the normal Hugging Face endpoint is available, omit `HF_ENDPOINT`.

Create the project-local symlink expected by the configs:

```bash
mkdir -p experiments/wan22_ti2v5b/models
ln -sfn /cache/llc/WAN-models/Wan2.2-TI2V-5B \
  experiments/wan22_ti2v5b/models/Wan2.2-TI2V-5B
```

Expected key files:

```text
Wan2.2_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
diffusion_pytorch_model-00001-of-00003.safetensors
diffusion_pytorch_model-00002-of-00003.safetensors
diffusion_pytorch_model-00003-of-00003.safetensors
diffusion_pytorch_model.safetensors.index.json
google/umt5-xxl/spiece.model
```

## Validate Before Running

Use the same Python executable for preflight and generation:

```bash
WAN_PYTHON=/cache/llc/conda_envs/wan22/bin/python \
  ./experiments/wan22_ti2v5b/run_smoke_720p.sh --dry-run
```

The smoke preflight should pass only when the Wan2.2 source tree, model files,
prompt config, and at least `24576` MiB free GPU memory are visible.

Run the 720P smoke:

```bash
WAN_PYTHON=/cache/llc/conda_envs/wan22/bin/python \
  ./experiments/wan22_ti2v5b/run_smoke_720p.sh --run-id smoke_720p_free_gpu
```

Run the 100 prompt x 3 seed baseline dry-run:

```bash
WAN_PYTHON=/cache/llc/conda_envs/wan22/bin/python \
  ./experiments/wan22_ti2v5b/run_baseline_720p.sh --dry-run --run-id dryrun_300
```

Run the full baseline:

```bash
WAN_PYTHON=/cache/llc/conda_envs/wan22/bin/python \
  ./experiments/wan22_ti2v5b/run_baseline_720p.sh
```

Outputs are written under `experiments/wan22_ti2v5b/runs/<run_id>/` with
`videos/`, `logs/`, and `metadata/results.jsonl`.

## Reproducibility Notes

- Official Wan2.2 TI2V 720P size is `1280*704` or `704*1280`.
- This harness uses `1280*704`, `frame_num=121`, `sample_steps=50`, and seeds
  `1000`, `1001`, `1002` for the full baseline.
- The fixed 100-prompt set is `experiments/wan_baseline/configs/prompts_100.jsonl`.
- The generation command adds `--offload_model True --convert_model_dtype --t5_cpu`.
- Do not commit `models/`, `src/`, `runs/`, or `preflight_reports/`; they are
  intentionally ignored.
