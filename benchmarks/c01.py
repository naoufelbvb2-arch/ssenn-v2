"""Run every configured trial in its own process and preserve unsuccessful results."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "c01.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("results.local.json"))
    parser.add_argument("--worker", nargs=3, metavar=("TASK", "METHOD", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.worker:
        from ssenn.experiment import run_trial
        task, method, seed = args.worker
        trial = run_trial(task, config["tasks"][task], method, int(seed))
        print(json.dumps(trial, allow_nan=False))
        return 0

    import torch
    runs = []
    report = {"config": config, "environment": {
        "python": platform.python_version(), "pytorch": str(torch.__version__),
        "os": platform.platform(), "cpu": os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
        "device": "cpu", "threads": 1, "dtype": "float32",
    }, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task in config["tasks"]:
        for method in config["methods"]:
            for seed in config["seeds"]:
                proc = subprocess.run(
                    [sys.executable, "-m", "benchmarks.c01", "--config", str(args.config),
                     "--worker", task, method, str(seed)], capture_output=True, text=True)
                if proc.returncode:
                    run = {"task": task, "method": method, "seed": seed, "status": "FAIL",
                           "error": proc.stderr.strip()}
                else:
                    run = json.loads(proc.stdout)
                runs.append(run)
                print(task, method, seed, run["status"], run.get("validation", run.get("error")), flush=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(any(r["status"] != "PASS" for r in runs))


if __name__ == "__main__":
    raise SystemExit(main())
