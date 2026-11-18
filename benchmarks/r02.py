"""Run the registered R02 diagnosis, lock the choices, and confirm them on fresh corpora."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter

import torch

from ssenn.conventional import AdamNetwork
from ssenn.credit import DirectFeedback
from ssenn.dynamics import Rates
from ssenn.light_dfa import LightDFA
from ssenn.local import LocalNetwork
from ssenn.network import initial_weights
from .r01 import ROOT
from .r01_metrics import score
from .r01_trials import Corpus
from .r02_measure import allocator_peak, force_probe, persistent_bytes


@dataclass(frozen=True)
class Setting:
    name: str
    method: str
    eta0: float = 0.
    beta: float = 0.
    gamma: float = 0.


def settings(p):
    springs = [Setting(name, "spring", eta0, beta, gamma) for name, eta0, beta, gamma in p["spring_grid"]]
    anchors = [Setting(f"anchor_{r:g}", "anchor", eta0=r) for r in p["anchor_rates"]]
    return [Setting("light", "light"), *anchors, *springs, Setting("adamw", "adamw")]


def make_model(s, seed, data, p, base):
    w = initial_weights(data.train[0].shape[1], base["hidden"], data.classes, seed)
    if s.method == "adamw":
        return AdamNetwork(w, p["adam_lr"], base["weight_decay"])
    credit = DirectFeedback.random(data.classes, base["hidden"], seed + 1)
    if s.method == "spring":
        return LocalNetwork(w, credit, Rates(p["eta_w"], s.eta0, s.gamma, s.beta))
    return LightDFA(w, credit, p["eta_w"], s.eta0 if s.method == "anchor" else None)


def metrics(model, batch, phase):
    x, y = batch
    scores = model.deployed_logits(x)
    return {**score(scores, y), "first_move_accuracy": score(scores[phase == 2], y[phase == 2])["accuracy"]}


def trial(s, corpus, seed, p, base, confirm):
    data = corpus.data
    x, y = data.train
    model = make_model(s, seed, data, p, base)
    warm = make_model(s, seed, data, p, base)
    size = base["batch_size"]
    warm.train_step(x[:size], y[:size])
    del warm
    sampler = torch.Generator().manual_seed(seed + 2)
    history, probes, endpoints, hits = [], [], {}, {}
    update_time = train_time = 0.
    steps = p["confirmation_extension"] if confirm else p["primary_steps"]
    for step in range(steps + 1):
        if step:
            t0 = perf_counter()
            ids = torch.randint(len(x), (size,), generator=sampler)
            batch = x[ids], y[ids]
            t1 = perf_counter()
            model.train_step(*batch)
            t2 = perf_counter()
            train_time += t2 - t0
            update_time += t2 - t1
        if step % base["eval_every"] == 0:
            val = metrics(model, data.validation, corpus.phases[1])
            history.append([step, val["ce"], val["accuracy"], val["first_move_accuracy"]])
            for key, hit in (("accuracy", val["accuracy"] >= base["target_accuracy"]),
                             ("ce", val["ce"] <= p["ce_target"])):
                if hit and key not in hits:
                    hits[key] = {"step": step, "seconds": train_time}
        if s.method == "spring" and step in p["probe_steps"]:
            peek = torch.Generator().set_state(sampler.get_state())
            ids = torch.randint(len(x), (size,), generator=peek)
            probes.append({"step": step, **force_probe(model, (x[ids], y[ids]), data.validation)})
        if step in (p["primary_steps"], steps):
            passed = val["accuracy"] >= base["target_accuracy"] and val["ce"] < history[0][1]
            end = {"validation": val, "status": "PASS" if passed else "FAIL",
                   "hits": dict(hits), "train_seconds": train_time, "update_seconds": update_time,
                   "memory_bytes": persistent_bytes(model)}
            if confirm:
                end["test"] = metrics(model, data.test, corpus.phases[2])
            endpoints[str(step)] = end
    return {"history": history, "probes": probes, "endpoints": endpoints}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sweep(path, corpora, variants, p, base, confirm):
    rows = []
    with path.open("w", encoding="utf-8") as f:
        for c in corpora:
            for i, seed in enumerate(p["model_seeds"]):
                # Rotate execution order within a corpus to distribute host warm-up effects.
                for s in variants[i:] + variants[:i]:
                    r = {"corpus": c.seed, "seed": seed, "setting": s.name}
                    try:
                        r.update(trial(s, c, seed, p, base, confirm))
                    except FloatingPointError as e:
                        r.update(status="FAIL", error=str(e))
                    f.write(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n")
                    rows.append(r)
            f.flush()
            print(path.parent.name, path.stem, c.seed, flush=True)
    return rows


def choose(rows, variants, p):
    losses = {}
    for s in variants:
        rs = [r for r in rows if r["setting"] == s.name]
        if len(rs) != len(p["model_seeds"]) * len(p["diagnosis_corpora"]):
            raise ValueError(f"Incomplete setting {s.name}")
        losses[s.name] = None if any("error" in r for r in rs) else mean(
            r["endpoints"][str(p["primary_steps"])]["validation"]["ce"] for r in rs)
    def best(eligible):
        return min((s.name for s in eligible if losses[s.name] is not None), key=losses.get)
    return {"spring": best(s for s in variants if s.beta > 0 and s.gamma > 0),
            "anchor": best(s for s in variants if s.method == "anchor"), "mean_validation_ce": losses}


def run(folder, p, base):
    folder.mkdir(parents=True, exist_ok=True)
    variants = settings(p)
    old = [Corpus.make(s, base["data"]) for s in p["diagnosis_corpora"]]
    search = sweep(folder / "diagnosis.jsonl", old, variants, p, base, False)
    lock = choose(search, variants, p)
    save(folder / "selection.json", lock)
    fresh = [Corpus.make(s, base["data"]) for s in p["confirmation_corpora"]]
    historical = json.loads((ROOT / "results/r01/manifest.json").read_text())["corpora"]
    for c in old:
        if historical[str(c.seed)]["doc_hashes_by_split"] != c.doc_ids:
            raise ValueError("R01 dataset fingerprint mismatch")
    ids = [h for c in historical.values() for split in c["doc_hashes_by_split"] for h in split]
    ids.extend(h for c in fresh for split in c.doc_ids for h in split)
    if len(ids) != len(set(ids)):
        raise ValueError("Document overlap")
    save(folder / "new_corpus_manifest.json", {c.seed: c.doc_ids for c in fresh})
    names = {"reference", "light", "adamw", lock["anchor"], lock["spring"]}
    selected = [s for s in variants if s.name in names]
    confirm = sweep(folder / "confirmation.jsonl", fresh, selected, p, base, True)
    return search, confirm, lock, selected, fresh[0]


def scientific(row):
    endpoints = {}
    for step, end in row.get("endpoints", {}).items():
        endpoints[step] = {k: v for k, v in end.items() if k not in ("train_seconds", "update_seconds", "hits")}
        endpoints[step]["hits"] = {k: v["step"] for k, v in end["hits"].items()}
    return row | {"endpoints": endpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("r02.local"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p = json.loads((ROOT / "configs/c01_r02.json").read_text())
    base = json.loads((ROOT / "configs/c01.json").read_text())["tasks"]["T02"]
    a = run(args.output / "primary", p, base)
    b = run(args.output / "replay", p, base)
    audit = {"lock_matches": a[2] == b[2], "python": platform.python_version(), "torch": str(torch.__version__),
             "cpu": platform.processor(), "platform": platform.platform(), "threads": torch.get_num_threads()}
    for i, key in enumerate(("diagnosis", "confirmation")):
        audit[key] = {"count": len(a[i]), "exact": sum(scientific(x) == scientific(y)
            for x,y in zip(a[i], b[i], strict=True))}
    save(args.output / "audit.json", audit)
    profile = {}
    for s in a[3]:
        profile[s.name] = [allocator_peak(make_model(s, seed, a[4].data, p, base),
            tuple(t[:base["batch_size"]] for t in a[4].data.train)) for seed in p["model_seeds"]]
    save(args.output / "memory.json", profile)
    print(audit)
    return int(not audit["lock_matches"] or any(audit[k]["count"] != audit[k]["exact"]
                                              for k in ("diagnosis", "confirmation")))


if __name__ == "__main__":
    raise SystemExit(main())
