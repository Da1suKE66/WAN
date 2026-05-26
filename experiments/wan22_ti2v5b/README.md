# Wan2.2 TI2V-5B 720P Candidate

This directory isolates the Wan2.2 TI2V-5B attempt from the Wan2.1 checkout.

## Current State

- Official model name: `Wan-AI/Wan2.2-TI2V-5B`.
- Official 720P TI2V size: `1280*704` or `704*1280`, not `1280*720`.
- Model download path: `/cache/llc/WAN-models/Wan2.2-TI2V-5B`.
- Project symlink: `experiments/wan22_ti2v5b/models/Wan2.2-TI2V-5B`.
- Wan2.2 source checkout: `experiments/wan22_ti2v5b/src/Wan2.2`, observed HEAD `42bf4cf fix readme`.
- Python runtime: `experiments/wan_baseline/envs/lightx_overlay`.

The model was downloaded successfully from Hugging Face mirror. A minimal 720P smoke was attempted with `--task ti2v-5B --size 1280*704 --frame_num 5 --sample_steps 1 --offload_model True --convert_model_dtype --t5_cpu`. It loaded T5, Wan2.2 VAE, and all 3 DiT shards, then failed when moving the DiT to CUDA because another process occupied about 70.9 GiB of the 80 GiB GPU.

## Commands

Portable setup on another server:

```bash
cat experiments/wan22_ti2v5b/SETUP_PORTABLE.md
```

Smoke:

```bash
./experiments/wan22_ti2v5b/run_smoke_720p.sh --run-id smoke_after_gpu_free
```

Baseline dry-run:

```bash
./experiments/wan22_ti2v5b/run_baseline_720p.sh --dry-run
```

Baseline real run:

```bash
./experiments/wan22_ti2v5b/run_baseline_720p.sh
```

## Files

- `configs/smoke_720p_ti2v5b.json`: 5-frame, 1-step smoke config.
- `configs/baseline_720p_ti2v5b.json`: 100 prompts x 3 seeds baseline config, reusing `experiments/wan_baseline/configs/prompts_100.jsonl`.
- `scripts/preflight_wan22_ti2v5b.py`: checks Wan2.2 code, model files, prompt/seed counts, official 720P size, and free GPU memory.
- `scripts/run_wan22_batch.py`: runs Wan2.2 `generate.py`, saving per-result logs and JSONL metadata.
- `requirements_overlay_wan22.txt`: extra packages installed into the shared overlay venv for Wan2.2 imports.
- `SETUP_PORTABLE.md`: complete setup notes for cloning this repo, installing the environment, cloning Wan2.2 source, downloading weights, and running the smoke/baseline on another GPU server.

## Blocker

Current GPU free memory is about 8.6 GiB. The official Wan2.2 TI2V-5B card says the single-GPU command needs at least 24 GiB VRAM with offload/conversion/T5-on-CPU. Free the existing large GPU process before rerunning the smoke or baseline.
