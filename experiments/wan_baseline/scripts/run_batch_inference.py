#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path):
    prompts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "prompt" not in item:
                raise ValueError(f"{path}:{line_no} must contain id and prompt")
            prompts.append(item)
    return prompts


def relpath(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def run_capture(cmd, cwd=REPO_ROOT):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def git_snapshot():
    return {
        "head": run_capture(["git", "rev-parse", "HEAD"]),
        "branch": run_capture(["git", "branch", "--show-current"]),
        "status_short": run_capture(["git", "status", "--short"]),
    }


class GpuMonitor:
    def __init__(self, interval_sec, target_pid=None):
        self.interval_sec = float(interval_sec)
        self.target_pid = int(target_pid) if target_pid is not None else None
        self.samples = []
        self.process_samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if shutil.which("nvidia-smi"):
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self):
        gpu_cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(gpu_cmd, text=True, stderr=subprocess.DEVNULL)
                for raw in out.strip().splitlines():
                    parts = [p.strip() for p in raw.split(",")]
                    if len(parts) >= 4:
                        self.samples.append({
                            "timestamp": parts[0],
                            "memory_used_mb": int(parts[1]),
                            "memory_free_mb": int(parts[2]),
                            "utilization_gpu_pct": int(parts[3]),
                        })
            except Exception:
                pass
            if self.target_pid is not None:
                try:
                    out = subprocess.check_output(proc_cmd, text=True, stderr=subprocess.DEVNULL)
                    for raw in out.strip().splitlines():
                        parts = [p.strip() for p in raw.split(",")]
                        if len(parts) >= 2 and int(parts[0]) == self.target_pid:
                            self.process_samples.append({
                                "timestamp": now_iso(),
                                "pid": self.target_pid,
                                "process_memory_mb": int(parts[1]),
                            })
                except Exception:
                    pass
            self._stop.wait(self.interval_sec)

    def summary(self):
        if not self.samples:
            return {
                "gpu_start_memory_used_mb": None,
                "gpu_peak_memory_used_mb": None,
                "gpu_peak_delta_memory_mb": None,
                "gpu_min_memory_free_mb": None,
                "gpu_peak_utilization_pct": None,
                "process_peak_memory_mb": None,
                "gpu_monitor_samples": 0,
                "process_monitor_samples": 0,
            }
        start_used = self.samples[0]["memory_used_mb"]
        peak_used = max(s["memory_used_mb"] for s in self.samples)
        return {
            "gpu_start_memory_used_mb": start_used,
            "gpu_peak_memory_used_mb": peak_used,
            "gpu_peak_delta_memory_mb": peak_used - start_used,
            "gpu_min_memory_free_mb": min(s["memory_free_mb"] for s in self.samples),
            "gpu_peak_utilization_pct": max(s["utilization_gpu_pct"] for s in self.samples),
            "process_peak_memory_mb": (
                max(s["process_memory_mb"] for s in self.process_samples)
                if self.process_samples
                else None
            ),
            "gpu_monitor_samples": len(self.samples),
            "process_monitor_samples": len(self.process_samples),
        }


def ffprobe(path):
    if not path.exists() or not shutil.which("ffprobe"):
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        return json.loads(subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT))
    except Exception as exc:
        return {"ffprobe_error": str(exc)}


def build_command(python_bin, cfg, prompt, seed, video_path):
    cmd = [
        str(python_bin),
        "generate.py",
        "--task",
        str(cfg["task"]),
        "--size",
        str(cfg["size"]),
        "--frame_num",
        str(cfg["frame_num"]),
        "--sample_steps",
        str(cfg["sample_steps"]),
        "--sample_shift",
        str(cfg.get("sample_shift", 5.0)),
        "--sample_guide_scale",
        str(cfg.get("sample_guide_scale", 5.0)),
        "--base_seed",
        str(seed),
        "--ckpt_dir",
        str(relpath(cfg["ckpt_dir"])),
        "--prompt",
        prompt,
        "--save_file",
        str(video_path),
    ]
    cmd.extend(str(x) for x in cfg.get("extra_args", []))
    return cmd


def run_one(cmd, log_path, env, monitor_interval_sec):
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {' '.join(cmd)}\n")
        log.write(f"# start_time: {now_iso()}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        monitor = GpuMonitor(monitor_interval_sec, target_pid=proc.pid)
        monitor.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = proc.wait()
        monitor.stop()
        elapsed = time.time() - start
        log.write(f"# end_time: {now_iso()}\n")
        log.write(f"# returncode: {returncode}\n")
        log.write(f"# elapsed_sec: {elapsed:.3f}\n")
    return returncode, elapsed, monitor.summary()


def main():
    parser = argparse.ArgumentParser(description="Batch Wan text-to-video inference with JSONL metadata.")
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run generate.py.")
    parser.add_argument("--run-id", default=None, help="Run directory name. Defaults to config run_name plus timestamp.")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--limit-seeds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = relpath(args.config)
    cfg = load_json(cfg_path)
    prompt_path = relpath(cfg["prompt_file"])
    prompts = load_prompts(prompt_path)
    seeds = list(cfg["seeds"])
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if args.limit_seeds is not None:
        seeds = seeds[: args.limit_seeds]

    ckpt_dir = relpath(cfg["ckpt_dir"])
    if not args.dry_run and not ckpt_dir.exists():
        raise FileNotFoundError(f"ckpt_dir does not exist: {ckpt_dir}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"{cfg.get('run_name', 'wan_batch')}_{stamp}"
    run_dir = relpath(cfg.get("output_root", "experiments/wan_baseline/runs")) / run_id
    video_dir = run_dir / "videos"
    log_dir = run_dir / "logs"
    meta_dir = run_dir / "metadata"
    for d in (video_dir, log_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "config_path": str(cfg_path),
        "prompt_path": str(prompt_path),
        "prompt_count": len(prompts),
        "seeds": seeds,
        "expected_result_count": len(prompts) * len(seeds),
        "dry_run": args.dry_run,
        "git": git_snapshot(),
        "config": cfg,
    }
    (meta_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(cfg_path, meta_dir / "config.json")
    shutil.copy2(prompt_path, meta_dir / "prompts.jsonl")

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    python_bin = relpath(args.python) if not Path(args.python).is_absolute() else Path(args.python)
    results_path = meta_dir / "results.jsonl"

    with results_path.open("a", encoding="utf-8") as results:
        for prompt_index, prompt_item in enumerate(prompts):
            for seed in seeds:
                result_id = f"{prompt_item['id']}_seed{seed}_{cfg['size'].replace('*', 'x')}_f{cfg['frame_num']}_s{cfg['sample_steps']}"
                video_path = video_dir / f"{result_id}.mp4"
                log_path = log_dir / f"{result_id}.log"
                cmd = build_command(python_bin, cfg, prompt_item["prompt"], seed, video_path)
                record = {
                    "result_id": result_id,
                    "prompt_id": prompt_item["id"],
                    "prompt_index": prompt_index,
                    "prompt": prompt_item["prompt"],
                    "seed": seed,
                    "task": cfg["task"],
                    "resolution": cfg["size"],
                    "frame_num": cfg["frame_num"],
                    "sample_steps": cfg["sample_steps"],
                    "sample_shift": cfg.get("sample_shift"),
                    "sample_guide_scale": cfg.get("sample_guide_scale"),
                    "ckpt_dir": str(ckpt_dir),
                    "video_path": str(video_path),
                    "log_path": str(log_path),
                    "command": cmd,
                    "started_at": now_iso(),
                }
                if args.dry_run:
                    record.update({"status": "dry_run", "returncode": None, "elapsed_sec": 0.0})
                else:
                    returncode, elapsed, gpu = run_one(cmd, log_path, env, cfg.get("monitor_interval_sec", 1.0))
                    record.update(gpu)
                    record.update({
                        "returncode": returncode,
                        "elapsed_sec": round(elapsed, 3),
                        "finished_at": now_iso(),
                        "status": "success" if returncode == 0 and video_path.exists() else "failed",
                        "video_probe": ffprobe(video_path),
                    })
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()

    print(f"Wrote metadata to {results_path}")


if __name__ == "__main__":
    main()
