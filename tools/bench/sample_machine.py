"""Read-only machine snapshot; does not change clocks or power settings."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"unavailable": str(exc)}


def sample():
    power = subprocess.run(["/usr/sbin/nvpmodel", "-q"], capture_output=True, text=True)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "l4t": read("/etc/nv_tegra_release"),
        "meminfo": read("/proc/meminfo"),
        "gpu_devfreq_hz": {
            name: read("/sys/class/devfreq/17000000.gpu/" + name)
            for name in ["cur_freq", "min_freq", "max_freq"]
        },
        "power_mode_query": {
            "returncode": power.returncode,
            "stdout": power.stdout,
            "stderr": power.stderr,
        },
        "background_containers": subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], check=True, capture_output=True, text=True
        ).stdout.splitlines(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(sample(), ensure_ascii=False, indent=2))
