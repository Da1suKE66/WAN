#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_MODEL_FILES = [
    "config.json",
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "google/umt5-xxl/spiece.model",
]
SUPPORTED_SIZES = {"1280*704", "704*1280"}


def relpath(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def now_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(cfg):
    if "prompts" in cfg:
        return list(cfg["prompts"]), []
    prompt_path = relpath(cfg["prompt_file"])
    prompts = []
    errors = []
    if not prompt_path.exists():
        return prompts, [f"prompt_file does not exist: {prompt_path}"]
    with prompt_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "id" not in item or "prompt" not in item:
                    errors.append(f"{prompt_path}:{line_no} missing id or prompt")
                prompts.append(item)
            except Exception as exc:
                errors.append(f"{prompt_path}:{line_no} {exc}")
    return prompts, errors


def nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return [], [], "nvidia-smi not found"
    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_out = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.STDOUT)
        proc_out = subprocess.check_output(proc_cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return [], [], str(exc)
    gpus = []
    for raw in gpu_out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            })
    procs = []
    for raw in proc_out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 3:
            procs.append({
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mb": int(parts[2]),
            })
    return gpus, procs, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-dir", default="experiments/wan22_ti2v5b/preflight_reports")
    args = parser.parse_args()

    cfg_path = relpath(args.config)
    cfg = load_json(cfg_path)
    failures = []
    warnings = []

    if cfg.get("task") != "ti2v-5B":
        failures.append(f"task must be ti2v-5B, got {cfg.get('task')}")
    if cfg.get("size") not in SUPPORTED_SIZES:
        failures.append(f"size {cfg.get('size')} is not official TI2V 720P size; use one of {sorted(SUPPORTED_SIZES)}")
    if int(cfg.get("frame_num", 0)) % 4 != 1:
        failures.append("frame_num must be 4n+1")

    code_dir = relpath(cfg["code_dir"])
    if not (code_dir / "generate.py").exists():
        failures.append(f"Wan2.2 generate.py not found under {code_dir}")

    ckpt_dir = relpath(cfg["ckpt_dir"])
    model_files = {}
    for rel in REQUIRED_MODEL_FILES:
        p = ckpt_dir / rel
        model_files[rel] = {"exists": p.exists(), "path": str(p)}
        if not p.exists():
            failures.append(f"missing model file: {rel}")

    prompts, prompt_errors = load_prompts(cfg)
    failures.extend(prompt_errors)
    if "expected_prompt_count" in cfg and len(prompts) != int(cfg["expected_prompt_count"]):
        failures.append(f"prompt count {len(prompts)} != expected {cfg['expected_prompt_count']}")
    if "expected_seed_count" in cfg and len(cfg.get("seeds", [])) != int(cfg["expected_seed_count"]):
        failures.append(f"seed count {len(cfg.get('seeds', []))} != expected {cfg['expected_seed_count']}")

    gpus, gpu_processes, gpu_error = nvidia_smi()
    if gpu_error:
        failures.append(f"GPU query failed: {gpu_error}")
    elif gpus:
        best_free = max(g["memory_free_mb"] for g in gpus)
        min_free = int(cfg.get("min_free_gpu_mb", 0))
        if best_free < min_free:
            failures.append(f"GPU free memory {best_free} MiB < required {min_free} MiB")

    report = {
        "created_at": now_iso(),
        "config_path": str(cfg_path),
        "task": cfg.get("task"),
        "size": cfg.get("size"),
        "frame_num": cfg.get("frame_num"),
        "sample_steps": cfg.get("sample_steps"),
        "prompt_count": len(prompts),
        "seed_count": len(cfg.get("seeds", [])),
        "expected_result_count": len(prompts) * len(cfg.get("seeds", [])),
        "code_dir": str(code_dir),
        "ckpt_dir": str(ckpt_dir),
        "model_files": model_files,
        "gpus": gpus,
        "gpu_processes": gpu_processes,
        "warnings": warnings,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    report_dir = relpath(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"preflight_wan22_ti2v5b_{now_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"preflight_report={report_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
