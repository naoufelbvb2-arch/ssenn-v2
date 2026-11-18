"""Reproduce the locked CPU diagnosis with two independent passes and a CPU profile."""

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import torch
from torch.profiler import profile, ProfilerActivity

from ssenn.credit import DirectFeedback
from ssenn.network import initial_weights
from .r01_metrics import breakdown, digest
from .r01_trials import Corpus, make_model, predictor, train, trial_config

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")


def manifest(corpora, cfg, seeds):
    documents = [doc for corpus in corpora.values() for split in corpus.doc_ids for doc in split]
    if len(set(documents)) != len(documents):
        raise ValueError("Document collision across registered splits or corpora")
    data = next(iter(corpora.values())).data
    rngs = {}
    for seed in seeds:
        weights = initial_weights(data.train[0].shape[1], cfg.hidden, data.classes, seed)
        feedback = DirectFeedback.random(data.classes, cfg.hidden, seed + 1)
        sampler = torch.Generator().manual_seed(seed + 2)
        ids = [torch.randint(len(data.train[0]), (cfg.batch_size,), generator=sampler).tolist()
               for _ in range(cfg.steps)]
        rngs[seed] = {"weights": digest([w.tolist() for w in weights]),
                      "feedback": digest(feedback.matrix.tolist()), "batch_indices": digest(ids)}
    return {"distinct_documents": len(documents), "model_rngs": rngs,
            "corpora": {s: {"doc_hashes_by_split": c.doc_ids, "vocabulary": c.vocab} for s, c in corpora.items()}}


def run_one(corpus, seed, method, scale, cfg, probes, confirm):
    model = make_model(cfg, corpus.data, method, scale, seed)
    warm = make_model(cfg, corpus.data, method, scale, seed)
    x, y = corpus.data.train
    warm.train_step(x[:cfg.batch_size], y[:cfg.batch_size])
    del warm
    row = {"corpus": corpus.seed, "seed": seed, "method": method, "scale": scale}
    try:
        row.update(train(model, method, seed, cfg, corpus.data.train, corpus.data.validation,
                         corpus.phases[1], corpus.vocab, probes))
        if confirm:
            row["views"] = {}
            for split, batch, phase in (("validation", corpus.data.validation, corpus.phases[1]),
                                        ("test", corpus.data.test, corpus.phases[2])):
                bx, by = batch
                row["views"][split] = {
                    "deployed": breakdown(predictor(model, method)(bx), by, phase),
                    "fast": breakdown(model.logits(bx), by, phase)}
    except FloatingPointError as e:
        row.update(status="FAIL", error=str(e))
    return row


def select(search, methods, grid, pairs):
    decisions = {}
    for method in methods:
        losses = []
        for scale in grid:
            rows = [r for r in search if (r["method"], r["scale"]) == (method, scale)]
            if len(rows) != pairs:
                raise ValueError("Incomplete candidate")
            loss = None if any("error" in r for r in rows) else statistics.mean(
                r["validation"]["overall"]["ce"] for r in rows)
            losses.append(loss)
        valid = [i for i, v in enumerate(losses) if v is not None and math.isfinite(v)]
        if not valid:
            raise ValueError(f"No finite candidate for {method}")
        best = min(valid, key=lambda i: losses[i])
        decisions[method] = {"scale": grid[best], "mean_validation_ce_in_grid_order": losses}
    return decisions


def plan(methods, decisions):
    settings = {(m, s) for m in methods for s in (1., decisions[m]["scale"])}
    for s in (1., decisions["spring"]["scale"]):
        settings.update((m, s) for m in ("ablation", "anchor_only", "restore_only", "spring"))
    return sorted(settings)


def profile_settings(corpus, seed, settings, cfg):
    rows = []
    x, y = (t[:cfg.batch_size] for t in corpus.data.train)
    for method, scale in settings:
        model = make_model(cfg, corpus.data, method, scale, seed)
        for _ in range(3):
            model.train_step(x, y)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            for _ in range(16):
                model.train_step(x, y)
        ops = prof.key_averages()
        matrix = [op for op in ops if op.key in {"aten::mm", "aten::bmm", "aten::addmm"}]
        rows.append({"method": method, "scale": scale,
                     "matrix_flops_per_step": sum(op.flops for op in matrix) / 16,
                     "matrix_ops_per_step": sum(op.count for op in matrix) / 16,
                     "self_cpu_us_per_step": sum(op.self_cpu_time_total for op in ops) / 16,
                     "top_self_cpu_us_per_step": {op.key: op.self_cpu_time_total / 16
                         for op in sorted(ops, key=lambda op: op.self_cpu_time_total, reverse=True)[:8]}})
    return rows


def learning(row):
    out = {k: v for k, v in row.items() if k not in ("training_seconds", "tokens_per_second")}
    hit = out.get("first_hit")
    if hit is not None:
        out["first_hit"] = hit["step"]
    return out


def run_pass(out, protocol, corpora, history_seed, cfg):
    out.mkdir(parents=True, exist_ok=True)
    seeds, methods = protocol["model_seeds"], protocol["methods"]
    search = []
    for method in methods:
        for scale in protocol["rate_multipliers"]:
            for corpus in protocol["corpus_seeds"]:
                for seed in seeds:
                    search.append(run_one(corpora[corpus], seed, method, scale, cfg, [], False))
        print(out.name, "search", method, flush=True)
    write(out / "search.json", search)
    decisions = select(search, methods, protocol["rate_multipliers"], len(seeds) * len(protocol["corpus_seeds"]))
    write(out / "selection.json", decisions)
    settings = plan(methods, decisions)
    confirms = []
    for corpus in corpora.values():
        variants = [(m, 1.) for m in methods] if corpus.seed == history_seed else settings
        for method, scale in variants:
            for seed in seeds:
                confirms.append(run_one(corpus, seed, method, scale, cfg, protocol["credit_probe_steps"], True))
        print(out.name, "confirmation corpus", corpus.seed, flush=True)
    write(out / "confirmation.json", confirms)
    return search, decisions, confirms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("r01.local"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / "configs/c01_r01.json").read_text())
    base = json.loads((ROOT / protocol["base_config"]).read_text())["tasks"][protocol["task"]]
    cfg = trial_config(base)
    history_seed = base["data"]["seed"]
    def make_corpora():
        return {s: Corpus.make(s, base["data"]) for s in [*protocol["corpus_seeds"], history_seed]}
    corpora = make_corpora()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = manifest(corpora, cfg, protocol["model_seeds"])
    write(args.output / "manifest.json", meta)
    first = run_pass(args.output / "primary", protocol, corpora, history_seed, cfg)
    replay_corpora = make_corpora()
    replay_manifest = manifest(replay_corpora, cfg, protocol["model_seeds"])
    second = run_pass(args.output / "replay", protocol, replay_corpora, history_seed, cfg)
    profiles = profile_settings(corpora[protocol["corpus_seeds"][0]], protocol["model_seeds"][0],
                                plan(protocol["methods"], first[1]), cfg)
    write(args.output / "profile.json", profiles)
    archived = json.loads((ROOT / "results/c01.json").read_text())["runs"]
    historical = {(r["method"], r["seed"]): r for r in archived if r["task"] == "T02"}
    matches = []
    for r in first[2]:
        if r["corpus"] == history_seed and "error" not in r:
            old = historical[r["method"], r["seed"]]
            test = r["views"]["test"]["deployed"]
            matches.append(all(test["overall"][k] == old["test"][k] for k in ("ce", "accuracy")) and
                           [p["accuracy"] for p in test["positions"]] == old["test_accuracy_by_clause_position"] and
                           all(a[0:3] == [b["step"], b["ce"], b["accuracy"]] for a,b in
                               zip(r["validation_history"], old["history"], strict=True)))
    audit = {"python": sys.version, "torch": str(torch.__version__), "os": platform.platform(),
             "cpu": platform.processor(), "threads": torch.get_num_threads(), "device": "cpu",
             "manifest_matches": meta == replay_manifest, "selection_matches": first[1] == second[1],
             "historical_matches": sum(matches), "historical_expected": len(historical)}
    for name, index in (("search", 0), ("confirmation", 2)):
        audit[name + "_count"] = len(first[index])
        audit[name + "_exact_matches"] = sum(learning(a) == learning(b)
            for a, b in zip(first[index], second[index], strict=True))
    write(args.output / "reproducibility.json", audit)
    print(json.dumps(audit, indent=2))
    return int(not (audit["manifest_matches"] and audit["selection_matches"] and
                    sum(matches) == len(historical) and all(audit[k + "_count"] == audit[k + "_exact_matches"]
                    for k in ("search", "confirmation"))))


if __name__ == "__main__":
    raise SystemExit(main())
