"""Export per-seed curves and inspect each local parameter group's effect on validation CE."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch

from .c02a import ROOT, evaluate
from ssenn.decoder import DecoderConfig
from ssenn.lm_data import load_data
from ssenn.lm_training import learner


def save_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for r in records for k in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


@torch.no_grad()
def reversion(row, cfg, data):
    """Validation-only diagnostic. Reset one deployed tensor while retaining all others."""
    net = DecoderConfig(len(data.vocab) + 3, **cfg["model"])
    model = learner(row["method"], net, row["seed"], cfg)
    initial = {k:w.clone() for k,w in model.net.weights.items()}
    rng = torch.Generator().manual_seed(row["seed"] + 2)
    for _ in range(cfg["steps"]):
        ids = torch.randint(len(data.train), (cfg["batch"],), generator=rng)
        b = data.train[ids]
        model.train_step(b[:, :-1], b[:, 1:])
    structured = row["task"] == "structured"
    baseline = evaluate(model,data.validation,structured)["ce"]
    if baseline != row["validation"][-1]["ce"]:
        raise RuntimeError("Could not reconstruct the recorded validation endpoint")
    weights = dict(model.anchors.named_buffers()) if row["method"] == "anchor" else dict(model.net.weights.items())
    diffs = {}
    for name in weights:
        replaced = weights | {name: initial[name]}
        def predict(ids, self_only=False):
            return model.net.compute(ids, replaced, self_only)[0]
        diffs[name] = evaluate(predict,data.validation,structured)["ce"] - baseline
    return diffs


def plot_curves(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = {"dfa":"#096e69", "anchor":"#a84d12", "adamw":"#454594"}
    for ax, (task, title) in zip(axes, (("structured","Structured symbols"), ("blake","Blake characters"))):
        series = [r for r in rows if r["task"] == task]
        for m, label in (("dfa","DFA"),("anchor","Moving anchor"),("adamw","AdamW")):
            runs = [r for r in series if r["method"] == m]
            steps = [c["step"] for c in runs[0]["validation"]]
            curves = [[c["ce"] for c in r["validation"]] for r in runs]
            for curve in curves:
                ax.plot(steps, curve, color=colors[m], alpha=.20, linewidth=1)
            ax.plot(steps, [mean(v) for v in zip(*curves)], label=label, color=colors[m], linewidth=2.2)
        for name, style in (("unigram","--"),("bigram",":")):
            ax.axhline(series[0]["baselines"][name+"_ce"], color="#444444", linestyle=style,
                       linewidth=1, label=name.title())
        ax.set(title=title, xlabel="Training updates", ylabel="Validation CE (nats per token)")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.savefig(out / "validation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c02a.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = [json.loads(line) for task in ("structured","blake")
            for line in (args.folder / f"{task}_primary.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "error" not in r]
    if args.plot or args.plot_only:
        plot_curves(rows, args.folder)
    if args.plot_only:
        return
    curves, coverage = [], []
    for task in ("structured", "blake"):
        data = load_data(task, cfg, ROOT)
        for row in (r for r in rows if r["task"] == task):
            keys = {k:row[k] for k in ("task", "seed", "method")}
            curves.extend(keys | point for point in row["validation"])
            deltas = {} if row["method"] == "adamw" else reversion(row,cfg,data)
            for name, final in row["coverage"]["final_deployed"].items():
                coverage.append(keys | {"group": name, "elements": final["elements"],
                    "changed_on_first_update": row["coverage"]["step_1"][name]["changed"],
                    "deployed_changed": final["changed"], "deployed_delta_l2": final["delta_l2"],
                    "reset_val_ce_minus_trained": deltas.get(name)})
    save_csv(args.folder / "validation_curves.csv", curves)
    save_csv(args.folder / "parameter_coverage.csv", coverage)


if __name__ == "__main__":
    main()
