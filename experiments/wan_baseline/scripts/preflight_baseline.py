#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


REQUIRED_SHARED_FILES = [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "google/umt5-xxl/spiece.model",
]


def now_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def relpath(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def count_jsonl(path):
    count = 0
    errors = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "id" not in item or "prompt" not in item:
                    errors.append(f"line {line_no}: missing id or prompt")
                count += 1
            except Exception as exc:
                errors.append(f"line {line_no}: {exc}")
    return count, errors


def nvidia_smi_gpu():
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found"
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return None, str(exc)
    gpus = []
    for raw in out.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            })
    return gpus, None


def nvidia_smi_processes():
    if not shutil.which("nvidia-smi"):
        return []
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return []
    rows = []
    for raw in out.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 3:
            rows.append({
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mb": int(parts[2]),
            })
    return rows


def validate_size(task, size):
    sys.path.insert(0, str(REPO_ROOT))
    from wan.configs import SUPPORTED_SIZES

    if task not in SUPPORTED_SIZES:
        return False, f"unsupported task {task}; known tasks: {sorted(SUPPORTED_SIZES)}"
    if size not in SUPPORTED_SIZES[task]:
        return False, f"unsupported size {size} for {task}; supported: {SUPPORTED_SIZES[task]}"
    return True, None


def check_checkpoint(ckpt_dir, task):
    failures = []
    warnings = []
    files = {}
    if not ckpt_dir.exists():
        failures.append(f"ckpt_dir does not exist: {ckpt_dir}")
        return failures, warnings, files
    for rel in REQUIRED_SHARED_FILES:
        p = ckpt_dir / rel
        files[rel] = {
            "exists": p.exists(),
            "path": str(p),
            "resolved": str(p.resolve()) if p.exists() else None,
        }
        if not p.exists():
            failures.append(f"missing checkpoint file: {rel}")
    config_path = ckpt_dir / "config.json"
    files["config.json"] = {
        "exists": config_path.exists(),
        "path": str(config_path),
        "resolved": str(config_path.resolve()) if config_path.exists() else None,
    }
    if not config_path.exists():
        failures.append("missing checkpoint file: config.json")
    else:
        try:
            config = load_json(config_path)
            model_type = config.get("model_type")
            if model_type != "t2v":
                failures.append(f"checkpoint config model_type is {model_type!r}, expected 't2v'")
        except Exception as exc:
            failures.append(f"invalid checkpoint config.json: {exc}")

    has_diffusion = any(ckpt_dir.glob("diffusion_pytorch_model*.safetensors"))
    files["diffusion_pytorch_model*.safetensors"] = {"exists": has_diffusion}
    if not has_diffusion:
        failures.append("missing diffusion_pytorch_model*.safetensors")

    if task == "t2v-14B" and "14B" not in ckpt_dir.name:
        warnings.append("task is t2v-14B but checkpoint directory name does not contain 14B")
    return failures, warnings, files


def main():
    parser = argparse.ArgumentParser(description="Preflight checks before Wan baseline inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-dir", default="experiments/wan_baseline/preflight_reports")
    parser.add_argument("--min-free-gpu-mb", type=int, default=None)
    args = parser.parse_args()

    config_path = relpath(args.config)
    cfg = load_json(config_path)
    prompt_path = relpath(cfg["prompt_file"])
    ckpt_dir = relpath(cfg["ckpt_dir"])

    failures = []
    warnings = []
    prompt_count = None
    prompt_errors = []
    if not prompt_path.exists():
        failures.append(f"prompt_file does not exist: {prompt_path}")
    else:
        prompt_count, prompt_errors = count_jsonl(prompt_path)
        failures.extend(f"prompt_file error: {x}" for x in prompt_errors)
        expected_prompt_count = cfg.get("expected_prompt_count")
        if expected_prompt_count is not None and prompt_count != expected_prompt_count:
            failures.append(f"prompt count {prompt_count} != expected {expected_prompt_count}")

    seed_count = len(cfg.get("seeds", []))
    expected_seed_count = cfg.get("expected_seed_count")
    if expected_seed_count is not None and seed_count != expected_seed_count:
        failures.append(f"seed count {seed_count} != expected {expected_seed_count}")

    ok, size_error = validate_size(cfg["task"], cfg["size"])
    if not ok:
        failures.append(size_error)

    ckpt_failures, ckpt_warnings, ckpt_files = check_checkpoint(ckpt_dir, cfg["task"])
    failures.extend(ckpt_failures)
    warnings.extend(ckpt_warnings)

    gpus, gpu_error = nvidia_smi_gpu()
    gpu_processes = nvidia_smi_processes()
    min_free = args.min_free_gpu_mb
    if min_free is None:
        min_free = int(cfg.get("min_free_gpu_mb", 0))
    if gpu_error:
        failures.append(f"GPU query failed: {gpu_error}")
    elif min_free:
        best_free = max(g["memory_free_mb"] for g in gpus) if gpus else 0
        if best_free < min_free:
            failures.append(f"GPU free memory {best_free} MiB < required {min_free} MiB")

    report = {
        "created_at": now_iso(),
        "config_path": str(config_path),
        "task": cfg["task"],
        "size": cfg["size"],
        "frame_num": cfg.get("frame_num"),
        "sample_steps": cfg.get("sample_steps"),
        "prompt_file": str(prompt_path),
        "prompt_count": prompt_count,
        "seed_count": seed_count,
        "expected_result_count": (prompt_count * seed_count) if prompt_count is not None else None,
        "ckpt_dir": str(ckpt_dir),
        "checkpoint_files": ckpt_files,
        "min_free_gpu_mb": min_free,
        "gpus": gpus,
        "gpu_processes": gpu_processes,
        "warnings": warnings,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }

    report_dir = relpath(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"preflight_{cfg['task'].replace('-', '_')}_{now_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"preflight_report={report_path}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
