"""Run the fixed Mini-Weka protocol against an existing service, with a wall limit.

The service owns the GPU lock. This client never deploys or reconfigures it.
An interrupted run retains diagnostics and cannot qualify as a benchmark score.
Periodic memory sampling uses proc/status; PSS is collected before replay so
walking a large model's page tables does not stall its timed work.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import urllib.request

from tools.build.files import atomic_json, file_hash


def command(args):
    return [
        str(args.harness),
        "profile",
        "--url",
        args.url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.model,
        "--tokenizer",
        str(args.tokenizer),
        "--input-file",
        str(args.dataset),
        "--custom-dataset-type",
        "weka-trace",
        "--trajectory-start-min-ratio",
        "0",
        "--trajectory-start-max-ratio",
        "0",
        "--burst-phase-starts",
        "--trace-idle-gap-cap-seconds",
        "2",
        "--system-idle-gap-cap-seconds",
        "1",
        "--inter-turn-delay-cap-seconds",
        "1",
        "--max-context-length",
        "262144",
        "--request-count",
        "64",
        "--concurrency",
        "16",
        "--random-seed",
        "42",
        "--warmup-requests-per-lane",
        "1",
        "--benchmark-duration",
        "180",
        "--benchmark-grace-period",
        "600",
        "--request-timeout-seconds",
        "1800",
        "--extra-inputs",
        '{"temperature":0}',
        "--export-level",
        "records",
        "--server-metrics-formats",
        "json",
        "jsonl",
        "--ui",
        "simple",
        "--output-artifact-dir",
        str(args.output),
        "--unsafe-override",
    ]


def fetch(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("harness", "tokenizer", "dataset", "manifest", "service-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18818")
    parser.add_argument("--model", default="deepseek-v41-flash-1017-w4a8")
    parser.add_argument("--service-pid", type=int, required=True)
    parser.add_argument("--wall-limit", type=int, default=900)
    args = parser.parse_args()
    if not 60 <= args.wall_limit <= 1200:
        parser.error("Mini diagnostics require a 60..1200 second wall limit")
    args.url = args.url.rstrip("/")
    health = json.loads(fetch(args.url + "/health"))
    if health["status"] != "ready" or health["model"] != args.model:
        raise ValueError("The selected service is not ready for this model")
    if health["max_context"] < 262144 or health["mtp"]["enabled"]:
        raise ValueError("Require 262K context with speculation disabled")
    if health["active_requests"] or health["queued_requests"]:
        raise ValueError("Start the Mini replay with an idle service")
    process = Path("/proc") / str(args.service_pid)
    executable = process / "exe"
    service_command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    service_environment = dict.fromkeys(
        (
            "ORINFER_A2_HOST_TABLE_MIB",
            "ORINFER_A2_HOST_ROW_CACHE_MIB",
            "ORINFER_A2_WORKER_CPUS",
            "ORINFER_HOST_ROW_CACHE_REPORT",
            "HCCL_OP_EXPANSION_MODE",
            "HCCL_DETERMINISTIC",
            "HCCL_NPU_SOCKET_PORT_RANGE",
        )
    )
    for entry in (process / "environ").read_bytes().split(b"\0"):
        key, _, value = entry.partition(b"=")
        name = key.decode(errors="replace")
        if name in service_environment:
            service_environment[name] = value.decode(errors="replace")
    if "serve " not in service_command:
        raise ValueError("The recorded service PID is not a serve process")
    manifest = json.loads(args.manifest.read_text())
    inputs = {}
    for window in manifest["windows"]:
        name = window["trace"] + ".json"
        digest = file_hash(args.dataset / name)
        if digest != window["sha256"]:
            raise ValueError("Mini source window differs: " + name)
        inputs[name] = digest
    if len(inputs) != 16:
        raise ValueError("Require the sixteen original Mini windows")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = args.output.parent / "agentx-mini.lock"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = os.environ.copy()
        environment.update(
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="1",
            HF_DATASETS_OFFLINE="1",
            AIPERF_DATASET_WEKA_PARALLEL_WORKERS="16",
        )
        # Reuse the original harness's cache locations without changing its code.
        environment.setdefault("HF_HOME", "/data/research/huggingface")
        environment.setdefault("AIPERF_DATASET_MMAP_CACHE_DIR", "/data/research/agentx-mmap-cache")
        # Leave room for ZMQ's IPC suffix under the 107-byte sun_path limit.
        temporary = tempfile.TemporaryDirectory(prefix="ori-mini-", dir="/tmp")
        environment["TMPDIR"] = temporary.name
        invocation = command(args)
        memory_start = time.monotonic()
        (args.output / "memory-before.txt").write_text((process / "smaps_rollup").read_text())
        memory_before_s = time.monotonic() - memory_start
        provenance = dict(
            started_at=datetime.datetime.now().astimezone().isoformat(),
            command=invocation,
            wall_limit_s=args.wall_limit,
            service_pid=args.service_pid,
            service_command=service_command,
            service_environment=service_environment,
            periodic_memory_source="proc/status",
            memory_before_s=memory_before_s,
            service_binary_sha256=file_hash(executable),
            service_model_sha256=file_hash(args.service_model / "cache/model.json"),
            manifest_sha256=file_hash(args.manifest),
            inputs=inputs,
            input_identity=hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            health_before=health,
            scope="Fixed Mini-Weka tuning surrogate; interrupted runs have no accepted score",
        )
        atomic_json(args.output / "provenance.json", provenance)
        (args.output / "metrics-before.txt").write_text(fetch(args.url + "/metrics"))
        start = time.monotonic()
        limited = False
        with (
            (args.output / "console.log").open("w") as console,
            (args.output / "service-samples.jsonl").open("w") as samples,
        ):
            child = subprocess.Popen(
                invocation,
                stdout=console,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            try:
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed >= args.wall_limit:
                        limited = True
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    record = dict(timestamp=time.time(), elapsed_s=elapsed)
                    sample_start = time.monotonic()
                    try:
                        record["health"] = json.loads(fetch(args.url + "/health"))
                        record["metrics"] = fetch(args.url + "/metrics")
                        record["memory"] = (process / "status").read_text()
                        record["process_io"] = (process / "io").read_text()
                        record["process_stat"] = (process / "stat").read_text()
                    except (OSError, ValueError) as error:
                        record["error"] = str(error)
                    record["observer_s"] = time.monotonic() - sample_start
                    samples.write(json.dumps(record) + "\n")
                    samples.flush()
                    time.sleep(2)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                status = child.wait()
        result = dict(exit_code=status, wall_limit_reached=limited, wall_s=time.monotonic() - start)
        try:
            rows = [
                json.loads(line)
                for line in (args.output / "profile_export.jsonl").read_text().splitlines()
                if line
            ]
            profiled = [r for r in rows if r["metadata"]["benchmark_phase"] == "profiling"]
            formal = json.loads((args.output / "profile_export_aiperf.json").read_text())
            complete = (
                len(profiled) == 64
                and all(
                    not row.get("error")
                    and not row["metadata"].get("was_cancelled")
                    and not row["metadata"].get("context_overflow_skip")
                    for row in profiled
                )
                and formal.get("was_cancelled") is False
            )
            result["profiled_requests"] = len(profiled)
        except (OSError, ValueError, KeyError) as error:
            complete = False
            result["validation_error"] = str(error)
        result["complete_mini_replay"] = complete and status == 0 and not limited
        atomic_json(args.output / "runner-result.json", result)
        try:
            (args.output / "metrics-after.txt").write_text(fetch(args.url + "/metrics"))
            atomic_json(args.output / "health-after.json", json.loads(fetch(args.url + "/health")))
        except OSError as error:
            (args.output / "service-after-error.txt").write_text(str(error))
        print(json.dumps(result), flush=True)
        temporary.cleanup()
        if not result["complete_mini_replay"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
