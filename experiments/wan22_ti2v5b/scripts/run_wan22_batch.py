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


def relpath(path):
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts(cfg):
    if "prompts" in cfg:
        return list(cfg["prompts"])
    prompts = []
    with relpath(cfg["prompt_file"]).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def run_capture(cmd, cwd=REPO_ROOT):
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


class GpuMonitor:
    def __init__(self, interval_sec=1.0):
        self.interval_sec = interval_sec
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if shutil.which("nvidia-smi"):
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

    def _loop(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
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
            self.stop_event.wait(self.interval_sec)

    def summary(self):
        if not self.samples:
            return {
                "gpu_start_memory_used_mb": None,
                "gpu_peak_memory_used_mb": None,
                "gpu_peak_delta_memory_mb": None,
                "gpu_min_memory_free_mb": None,
                "gpu_peak_utilization_pct": None,
                "gpu_monitor_samples": 0,
            }
        start_used = self.samples[0]["memory_used_mb"]
        peak_used = max(s["memory_used_mb"] for s in self.samples)
        return {
            "gpu_start_memory_used_mb": start_used,
            "gpu_peak_memory_used_mb": peak_used,
            "gpu_peak_delta_memory_mb": peak_used - start_used,
            "gpu_min_memory_free_mb": min(s["memory_free_mb"] for s in self.samples),
            "gpu_peak_utilization_pct": max(s["utilization_gpu_pct"] for s in self.samples),
            "gpu_monitor_samples": len(self.samples),
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


def build_cmd(python_bin, cfg, prompt, seed, video_path):
    cmd = [
        str(python_bin),
        "generate.py",
        "--task",
        cfg["task"],
        "--size",
        cfg["size"],
        "--frame_num",
        str(cfg["frame_num"]),
        "--sample_steps",
        str(cfg["sample_steps"]),
        "--sample_shift",
        str(cfg.get("sample_shift", 5.0)),
        "--sample_guide_scale",
        str(cfg.get("sample_guide_scale", 5.0)),
        "--ckpt_dir",
        str(relpath(cfg["ckpt_dir"])),
        "--base_seed",
        str(seed),
        "--prompt",
        prompt,
        "--save_file",
        str(video_path),
    ]
    cmd.extend(str(x) for x in cfg.get("extra_args", []))
    return cmd


def run_one(cmd, cwd, log_path, env):
    monitor = GpuMonitor()
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {' '.join(cmd)}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# start_time: {now_iso()}\n")
        log.flush()
        monitor.start()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--limit-seeds", type=int)
    args = parser.parse_args()

    cfg_path = relpath(args.config)
    cfg = load_json(cfg_path)
    prompts = load_prompts(cfg)
    seeds = list(cfg["seeds"])
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if args.limit_seeds is not None:
        seeds = seeds[: args.limit_seeds]

    run_id = args.run_id or f"{cfg['run_name']}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = relpath(cfg["output_root"]) / run_id
    video_dir = run_dir / "videos"
    log_dir = run_dir / "logs"
    meta_dir = run_dir / "metadata"
    for d in (video_dir, log_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=False)

    code_dir = relpath(cfg["code_dir"])
    python_bin = relpath(args.python) if not Path(args.python).is_absolute() else Path(args.python)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.get("env", {}).items()})
    manifest = {
        "created_at": now_iso(),
        "config_path": str(cfg_path),
        "code_dir": str(code_dir),
        "model_dir": str(relpath(cfg["ckpt_dir"])),
        "prompt_count": len(prompts),
        "seeds": seeds,
        "expected_result_count": len(prompts) * len(seeds),
        "dry_run": args.dry_run,
        "wan21_git_head": run_capture(["git", "rev-parse", "HEAD"]),
        "wan22_git_head": run_capture(["git", "rev-parse", "HEAD"], cwd=code_dir),
        "config": cfg,
    }
    (meta_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (meta_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    results_path = meta_dir / "results.jsonl"
    with results_path.open("a", encoding="utf-8") as results:
        for prompt_index, item in enumerate(prompts):
            for seed in seeds:
                result_id = f"{item['id']}_seed{seed}_{cfg['size'].replace('*', 'x')}_f{cfg['frame_num']}_s{cfg['sample_steps']}"
                video_path = video_dir / f"{result_id}.mp4"
                log_path = log_dir / f"{result_id}.log"
                cmd = build_cmd(python_bin, cfg, item["prompt"], seed, video_path)
                record = {
                    "result_id": result_id,
                    "prompt_id": item["id"],
                    "prompt_index": prompt_index,
                    "prompt": item["prompt"],
                    "seed": seed,
                    "task": cfg["task"],
                    "resolution": cfg["size"],
                    "frame_num": cfg["frame_num"],
                    "sample_steps": cfg["sample_steps"],
                    "video_path": str(video_path),
                    "log_path": str(log_path),
                    "command": cmd,
                    "started_at": now_iso(),
                }
                if args.dry_run:
                    record.update({"status": "dry_run", "returncode": None, "elapsed_sec": 0.0})
                else:
                    returncode, elapsed, gpu = run_one(cmd, code_dir, log_path, env)
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
