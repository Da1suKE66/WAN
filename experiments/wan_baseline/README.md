# Wan Baseline Reproduction Notes

This directory contains project-local scripts, configs, logs, and checkpoint views for reproducible Wan text-to-video baseline inference on the remote workspace.

## Current Status

- Remote repo: `/home/ma-user/workspace/llc/WAN`
- Git HEAD observed during setup: `9737cba Update README with community projects using Wan2.1 (#582)`
- Runtime: `experiments/wan_baseline/envs/lightx_overlay`, a project-local venv that inherits `/cache/envs/lightx` and installs only small missing packages.
- Successful smoke output: `experiments/wan_baseline/runs/smoke_1p3b_480p_f1/videos/p000_seed1000_832x480_f1_s1.mp4`
- Smoke ffprobe: 832x480, 1 frame, 16 fps.

The requested 720p baseline is not complete yet. The remote machine currently has no complete `Wan2.1-T2V-14B` checkpoint visible under `/home/ma-user` or `/cache`, and the only GPU had about 72.6 GB already occupied during setup. The available Wan2.1 T2V 1.3B checkpoint supports only 480x832 / 832x480 in this repo.

## Files

- `configs/prompts_100.jsonl`: fixed 100 prompt set for baseline batches.
- `configs/baseline_720p_t2v14b.json`: target 720p config for 100 prompts x 3 seeds. Replace `ckpt_dir` with a complete Wan2.1-T2V-14B checkpoint before running.
- `configs/smoke_1p3b_480p_f1.json`: minimal smoke config used to verify the runtime and save path.
- `scripts/run_batch_inference.py`: one-command batch runner that writes per-result logs and JSONL metadata.
- `scripts/preflight_baseline.py`: checks prompt count, seed count, supported resolution, checkpoint completeness, and free GPU memory before the 720p run.
- `run_smoke.sh`: runs the minimal smoke config.
- `run_baseline_720p.sh`: runs the target 720p baseline config.
- `requirements_overlay.txt`: small packages installed into the project-local overlay venv.
- `checkpoints/Wan2.1-T2V-1.3B-linked`: symlink-only checkpoint view assembled from existing remote files; original checkpoint files are not modified.

## Commands

From repo root:

```bash
./experiments/wan_baseline/run_smoke.sh --run-id smoke_check
```

Target baseline, after replacing `ckpt_dir` in `configs/baseline_720p_t2v14b.json`:

```bash
./experiments/wan_baseline/run_baseline_720p.sh
```

Dry-run the 300 planned jobs without launching generation:

```bash
./experiments/wan_baseline/run_baseline_720p.sh --dry-run
```

Run only the baseline preflight:

```bash
experiments/wan_baseline/envs/lightx_overlay/bin/python \
  experiments/wan_baseline/scripts/preflight_baseline.py \
  --config experiments/wan_baseline/configs/baseline_720p_t2v14b.json
```

## Metadata

Each run creates:

- `metadata/run_manifest.json`
- `metadata/config.json`
- `metadata/prompts.jsonl`
- `metadata/results.jsonl`
- `logs/*.log`
- `videos/*.mp4`

Each `results.jsonl` row records prompt, seed, task, resolution, frame count, sampling steps, command, elapsed seconds, return code, status, video path, log path, ffprobe metadata, global GPU peak/min-free samples, and GPU peak delta from `nvidia-smi`. PID-local memory is attempted but may be null when the remote container and `nvidia-smi` use different PID namespaces.

Preflight reports are written under `preflight_reports/`.

## Known Blockers

1. Complete 720p Wan2.1 T2V baseline requires `Wan2.1-T2V-14B`; only incomplete 1.3B T2V folders and VACE/Wan2.2 weights were found.
2. Current visible GPU memory was too occupied for normal work: a separate process used about 72.6 GB. A 5-frame 1.3B smoke reached VAE decode and then OOMed.
3. The successful 1-frame smoke proves the prompt-to-save pipeline can run, but it is not evidence for the requested 720p/1080p baseline.
