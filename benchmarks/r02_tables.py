"""Summarize the registered R02 measurements and verify parity with R01."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from .r01 import ROOT


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, cells)) + " |" for cells in
                     (headers, ["---"] * len(headers), *rows))


def summarize(out):
    p = json.loads((ROOT / "configs/c01_r02.json").read_text())
    lock = json.loads((out / "primary/selection.json").read_text())
    mem = json.loads((out / "memory.json").read_text())
    diag = read_rows(out / "primary/diagnosis.jsonl")
    confirm = read_rows(out / "primary/confirmation.jsonl")
    main = str(p["primary_steps"])
    old = json.loads((ROOT / "results/r01/primary/confirmation.json").read_text())
    historical = {(r["method"], r["corpus"], r["seed"]): r for r in old if r["scale"] == 4}
    checks = {"light": "ablation", "reference": "spring", "anchor_0.05": "anchor_only"}
    for r in diag:
        if r["setting"] in checks:
            curve = historical[checks[r["setting"]], r["corpus"], r["seed"]]["validation_history"]
            if r["history"] != [[c[0], c[1], c[2], c[4]] for c in curve]:
                raise ValueError(f"R01 trajectory mismatch for {r['setting']}")
    exported = []
    for cohort, rows in (("diagnosis", diag), ("confirmation", confirm)):
        for r in rows:
            key = {"cohort": cohort, **{k: r[k] for k in ("corpus", "seed", "setting")}}
            if "error" in r:
                exported.append(key | {"status": "FAIL", "error": r["error"]})
                continue
            for step, e in r["endpoints"].items():
                entry = key | {"step": step, "status": e["status"],
                               "train_seconds": e["train_seconds"], "update_seconds": e["update_seconds"]}
                for target in ("accuracy", "ce"):
                    for k in ("step", "seconds"):
                        entry[f"{target}_hit_{k}"] = e["hits"].get(target, {}).get(k)
                for split in ("validation", "test"):
                    entry.update({f"{split}_{k}": v for k,v in e.get(split, {}).items() if k != "n"})
                exported.append(entry)
    with (out / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in exported for k in r)))
        writer.writeheader()
        writer.writerows(exported)

    sections = ["# R02 measurement tables",
                "All quality and force summaries use the primary pass. Costs describe unprofiled single-thread CPU runs unless labeled allocator measurements. All 27 matched historical validation trajectories reproduced exactly, including the full learning curve for lightweight DFA versus the R01 ablation."]
    for cohort, rows in (("R01 diagnosis", diag), ("Fresh confirmation", confirm)):
        scores = []
        for name in dict.fromkeys(r["setting"] for r in rows):
            group = [r for r in rows if r["setting"] == name]
            for step in (main, str(p["confirmation_extension"])) if cohort == "Fresh confirmation" else (main,):
                ends = [r["endpoints"][step] for r in group if "error" not in r]
                if not ends:
                    continue
                hits = [e["hits"]["ce"] for e in ends if "ce" in e["hits"]]
                scores.append([name, step, f'{sum(e["status"] == "PASS" for e in ends)}/{len(group)}',
                    f'{mean(e["validation"]["ce"] for e in ends):.5f}',
                    f'{100*mean(e["validation"]["first_move_accuracy"] for e in ends):.2f}',
                    f'{mean(e["test"]["ce"] for e in ends):.5f}' if "test" in ends[0] else "—",
                    f'{len(hits)}/{len(group)}', median(h["step"] for h in hits) if hits else "—",
                    f'{median(h["seconds"] for h in hits):.4f}' if hits else "—",
                    f'{1e6*mean(e["update_seconds"] for e in ends)/int(step):.1f}'])
        sections += [f"## {cohort}", table(["Setting","Steps","Final PASS","Val CE","Val first move %",
            "Test CE","CE ≤ .60 hits","Median hit step¹","Median hit s¹","Mean update µs"], scores)]
    sections += ["¹ Hit medians are conditional on reaching the target by that horizon. The historical accuracy gate and cost of each hit appear in the per-seed CSV."]

    def key(r):
        return r["corpus"], r["seed"]
    spring = {key(r): r["endpoints"][main] for r in confirm if r["setting"] == lock["spring"] and "error" not in r}
    anchor = {key(r): r["endpoints"][main] for r in confirm if r["setting"] == lock["anchor"] and "error" not in r}
    contrasts = []
    for target, units in (("validation","ce"), ("ce","step"), ("ce","seconds"), ("accuracy","step")):
        deltas = []
        outcomes = []
        corpus_deltas = []
        for c in p["confirmation_corpora"]:
            dc = []
            for s in p["model_seeds"]:
                a, b = spring.get((c,s)), anchor.get((c,s))
                def value(e):
                    if e is None:
                        return None
                    return e["validation"]["ce"] if target == "validation" else e["hits"].get(target, {}).get(units)
                va, vb = value(a), value(b)
                if va is not None and vb is not None:
                    dc.append(va-vb)
                u, v = (float("inf") if n is None else n for n in (va,vb))
                outcomes.append((u>v)-(u<v))
            deltas.extend(dc)
            corpus_deltas.append(f"{mean(dc):+.5f} ({len(dc)}/3)" if dc else "—")
        contrasts.append([target, units, outcomes.count(-1), outcomes.count(0), outcomes.count(1),
                          f"{mean(deltas):+.5f}" if deltas else "—", *corpus_deltas])
    sections += ["## Locked spring minus moving anchor at 1,000 steps",
                 "A method missing a target loses to one that hits. Both missing is a tie. Numerical differences use only pairs with both values, with coverage shown per corpus.",
                 table(["Target","Measure","Spring wins","Ties","Losses","Mean paired Δ", *p["confirmation_corpora"]], contrasts)]
    forces = []
    for name in dict.fromkeys(("reference", lock["spring"])):
        rows = [r for r in confirm if r["setting"] == name and "error" not in r]
        for t in p["probe_steps"]:
            cells = [c for r in rows for c in r["probes"] if c["step"] == t]
            def avg(k):
                vals = [c[k] for c in cells if c[k] is not None]
                return f"{mean(vals):.5g}" if vals else "—"
            helpful = [c for c in cells if c["val_ce_drive"] < c["val_ce_before"]]
            harmed = sum(c["val_ce_coupled"] > c["val_ce_drive"] for c in helpful)
            forces.append([name,t,avg("mean_k"),avg("mean_abs_displacement"),avg("mean_abs_elastic"),
                avg("mean_abs_drive"),avg("elastic_to_drive"),avg("elastic_drive_cosine"),
                f"{mean(c['val_ce_coupled']-c['val_ce_drive'] for c in cells):+.6f}",f"{harmed}/{len(helpful)}"])
    sections += ["## Force evolution on fresh corpora",
                "Magnitudes are mean absolute coefficient updates. Ratios and cosines use flattened L2 vectors. Positive Δ CE means restoration worsened fast-weight validation CE versus the same drive-only step. The last column restricts this comparison to drive-only steps that improved CE from the current state.",
                table(["Setting","Step","Mean k","Displacement","Elastic magnitude","Drive magnitude",
                       "Ratio","Cosine","Δ CE","Helpful drive attenuated"],forces)]
    sections += ["## Memory per warmed CPU update",
                 table(["Setting","Deployed bytes","Training-only bytes","Peak above entry (min–max bytes)"],
                       [[name,ms[0]["deployed"],ms[0]["training_only"],
                         f'{min(m["peak_above_entry"] for m in ms)}–{max(m["peak_above_entry"] for m in ms)}']
                        for name,ms in mem.items()])]
    (out / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
