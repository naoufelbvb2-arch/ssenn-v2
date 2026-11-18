"""Count dense matrix arithmetic in one CPU training update, outside timed trials."""

import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.c01 import DEFAULT_CONFIG
from ssenn.data import token_task, xor
from ssenn.dynamics import Rates
from ssenn.experiment import Trial, build


def measure(config: dict) -> list[dict]:
    torch.set_num_threads(1)
    rows = []
    for task, settings in config["tasks"].items():
        values = settings.copy()
        data = xor() if task == "T01" else token_task(**values.pop("data"))
        cfg = Trial(**(values | {"rates": Rates(**values["rates"])}))
        x, y = (t[:cfg.batch_size] for t in data.train)
        for method in config["methods"]:
            model = build(cfg, data, method, config["seeds"][0])
            model.train_step(x, y)
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as events:
                model.train_step(x, y)
            ops = [e for e in events.key_averages() if e.key in {"aten::mm", "aten::addmm", "aten::bmm"}]
            count = sum(e.flops for e in ops)
            if count <= 0:
                raise RuntimeError(f"Profiler did not count matrix operations for {task} {method}")
            rows.append({"task": task, "method": method, "batch_size": len(y),
                         "matrix_operations": sum(e.count for e in ops), "matrix_flops": count})
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(measure(json.loads(args.config.read_text())), indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)
