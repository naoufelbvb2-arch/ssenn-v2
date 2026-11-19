"""Run the C02-A decoder CPU experiment and verify uninterrupted vs resumed training."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.nn.functional import cross_entropy

from ssenn.decoder import DecoderConfig
from ssenn.lm_data import EOS, load_data
from ssenn.lm_training import checkpoint, learner, restore

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate(model, partition, structured: bool, self_only: bool = False):
    xs, ys, offsets = partition
    loss_sum = hits = tokens = move_hits = moves = 0
    for x, y, off in zip(xs.split(64), ys.split(64), offsets.split(64)):
        logits = model(x, self_only=self_only)
        valid = y != -100
        loss_sum += cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        correct = logits.argmax(-1) == y
        hits += correct.sum().item()
        tokens += valid.sum().item()
        if structured:
            move = valid & (y != EOS) & (off % 5 == 2)
            move_hits += (correct & move).sum().item()
            moves += move.sum().item()
    ce = loss_sum / tokens
    if not math.isfinite(ce):
        raise FloatingPointError("Nonfinite evaluation")
    return {"tokens": tokens, "ce": ce, "ppl": math.exp(ce), "accuracy": hits / tokens,
            **({"first_move_accuracy": move_hits / moves} if moves else {})}


def count_baselines(data, partition):
    x, y, _ = partition
    valid = y != -100
    x, y = x[valid], y[valid]
    unigram = data.counts.double() + 1
    bigram = data.transitions.double() + 1
    return {"unigram_ce": -(unigram / unigram.sum())[y].log().mean().item(),
            "bigram_ce": -(bigram / bigram.sum(1, keepdim=True))[x, y].log().mean().item()}


def coverage(weights, initial):
    result = {}
    for name, value in weights.items():
        delta = value.detach() - initial[name]
        result[name] = {"elements": delta.numel(), "changed": delta.count_nonzero().item(),
                        "delta_l2": delta.norm().item()}
    return result


def memory(model):
    weights = list(model.net.weights.values())
    extra = list(model.feedback.buffers()) if hasattr(model, "feedback") else []
    if hasattr(model, "anchors"):
        extra.extend(model.anchors.buffers())
    if hasattr(model, "optimizer"):
        extra.extend(v for state in model.optimizer.state.values() for v in state.values())
    groups = {"deployed_weights": weights, "training_only": extra, "fixed_forward": list(model.net.buffers())}
    return {k: {"elements": sum(t.numel() for t in values),
                "bytes": sum(t.numel()*t.element_size() for t in values)} for k, values in groups.items()}


def trial(task, method, seed, data, cfg, resume):
    net = DecoderConfig(vocab=len(data.vocab)+3, **cfg["model"])
    model = learner(method, net, seed, cfg)
    initial = {n: w.detach().clone() for n,w in model.net.weights.items()}
    sampler = torch.Generator().manual_seed(seed + 2)
    batches = hashlib.sha256()
    history = []
    seconds = 0.
    run_id = hashlib.sha256(json.dumps([cfg, task, method, seed, data.manifest],
                                      sort_keys=True).encode()).hexdigest()
    for step in range(cfg["steps"] + 1):
        if step:
            before = perf_counter()
            ids = torch.randint(len(data.train), (cfg["batch"],), generator=sampler)
            window = data.train[ids]
            model.train_step(window[:, :-1], window[:, 1:])
            seconds += perf_counter() - before
            batches.update(str(ids.tolist()).encode())
            if step == 1:
                first = coverage(model.net.weights, initial)
        if step % cfg["evaluate_every"] == 0 or step == cfg["steps"]:
            history.append({"step": step, **evaluate(model, data.validation, task == "structured")})
        if resume and step == cfg["steps"] // 2:
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "decoder.pt"
                checkpoint(path, model, sampler, step, run_id)
                model = learner(method, net, seed, cfg)
                sampler = torch.Generator()
                if restore(path, model, sampler, run_id) != step:
                    raise RuntimeError("Resume step mismatch")
    val = history[-1]
    base = count_baselines(data, data.validation)
    final = coverage(model.net.weights, initial)
    gate = val["ce"] < min(history[0]["ce"], base["unigram_ce"])
    return {"task": task, "method": method, "seed": seed, "status": "PASS" if gate else "FAIL",
            "validation": history, "test": evaluate(model, data.test, task == "structured"),
            "self_only_validation": evaluate(model, data.validation, task == "structured", True),
            "baselines": base, "coverage": {"step_1": first, "final_fast": final,
                "final_deployed": coverage(dict(model.anchors.named_buffers()), initial) if method=="anchor" else final},
            "batch_stream_sha256": batches.hexdigest(), "memory": memory(model), "train_seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c02a.json")
    parser.add_argument("--output", type=Path, default=Path("c02a.local"))
    opts = parser.parse_args()
    out = opts.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(opts.config.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records = {}
    for task in ("structured", "blake"):
        data = load_data(task, cfg, ROOT)
        (out / f"{task}_manifest.json").write_text(json.dumps(data.manifest, indent=2)+"\n")
        for mode in ("primary", "resumed"):
            with (out / f"{task}_{mode}.jsonl").open("w") as f:
                for seed in cfg["model_seeds"]:
                    for method in cfg["methods"]:
                        try:
                            row = trial(task, method, seed, data, cfg, mode == "resumed")
                        except (FloatingPointError, OverflowError) as exc:
                            row = {"task": task, "seed": seed, "method": method, "status":"FAIL", "error":str(exc)}
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                        f.flush()
                        records[task, method, seed, mode] = row
                    print(task, mode, seed, flush=True)
    matches = []
    for (task, method, seed, mode), r in records.items():
        if mode == "primary":
            other = records[task, method, seed, "resumed"]
            matches.append(all(v == other.get(k) for k,v in r.items() if k != "train_seconds"))
    audit = {"cpu": platform.processor(), "python": platform.python_version(), "torch": str(torch.__version__),
             "threads": torch.get_num_threads(), "trajectories": len(matches), "exact_resumed": sum(matches),
             "primary_science": dict(Counter(r["status"] for k,r in records.items() if k[-1] == "primary"))}
    (out / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(audit)
    return int(not all(matches) or any("error" in r for r in records.values()))


if __name__ == "__main__":
    raise SystemExit(main())
