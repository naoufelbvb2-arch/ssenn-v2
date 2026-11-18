"""Build review tables from recorded trials without rerunning training."""

import argparse
import csv
from itertools import groupby
import json
from pathlib import Path
from statistics import mean, median


def md_table(header, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in
                     [header, ["---"] * len(header), *rows])


def main(folder: Path):
    def load(path):
        return json.loads((folder / path).read_text())
    rows = load("primary/confirmation.json")
    selection = load("primary/selection.json")
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/c01_r01.json").read_text())
    new = protocol["corpus_seeds"]
    def group(r):
        return ("new" if r["corpus"] in new else "historical", r["method"], r["scale"])
    overview, by_position, paired, by_corpus = [], [], [], []
    exports = []
    for r in rows:
        entry = {k: r[k] for k in ("corpus", "seed", "method", "scale", "status")}
        entry["error"] = r.get("error")
        if "error" not in r:
            entry["first_hit_step"] = (r["first_hit"] or {}).get("step")
            entry["first_hit_seconds"] = (r["first_hit"] or {}).get("training_seconds")
            entry["training_seconds"] = r["training_seconds"]
            entry["additional_state_bytes"] = r["state_bytes"]["additional_training"]
            for split, views in r["views"].items():
                for view, scores in views.items():
                    for name, cell in [("all", scores["overall"]), *enumerate(scores["positions"])]:
                        for metric in ("accuracy", "ce"):
                            entry[f"{split}_{view}_{name}_{metric}"] = cell[metric]
        exports.append(entry)
    with (folder / "per_seed.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(dict.fromkeys(k for e in exports for k in e)))
        writer.writeheader()
        writer.writerows(exports)

    for key, members in groupby(sorted(rows, key=group), key=group):
        scope, method, scale = key
        all_runs = list(members)
        rs = [r for r in all_runs if "error" not in r]
        if not rs:
            continue
        prefix = [scope, method, f"{scale:g}"]
        hits = [r["first_hit"]["step"] for r in rs if r["first_hit"]]
        def avg(split, name):
            return mean(r["views"][split]["deployed"]["overall"][name] for r in rs)
        overview.append([*prefix, f'{sum(r["status"] == "PASS" for r in rs)}/{len(all_runs)}',
            len(hits), median(hits) if hits else "—", f'{avg("validation","ce"):.4f}',
            f'{100*avg("test","accuracy"):.2f}', f'{avg("test","ce"):.4f}',
            f'{mean(r["training_seconds"] for r in rs):.3f}'])
        for view in ("deployed", "fast"):
            for split in ("validation", "test"):
                cells = []
                for p in range(5):
                    scores = [r["views"][split][view]["positions"][p] for r in rs]
                    cells.append(f'{100*mean(s["accuracy"] for s in scores):.2f}% / {mean(s["ce"] for s in scores):.3f}')
                by_position.append([*prefix, split, view, *cells])
        if scope == "new" and method in selection:
            for seed in new:
                subset = [r for r in rs if r["corpus"] == seed]
                if not subset:
                    continue
                by_corpus.append([seed, method, scale, sum(r["status"] == "PASS" for r in subset),
                    f'{mean(r["validation"]["overall"]["ce"] for r in subset):.4f}',
                    f'{100*mean(r["views"]["test"]["deployed"]["positions"][2]["accuracy"] for r in subset):.2f}'])

    lookup = {(r["corpus"],r["seed"],r["method"],r["scale"]): r for r in rows if "error" not in r}
    contrasts = [("anchor_only","deployed","ablation","fast"), ("anchor_only","fast","ablation","fast"),
                 ("restore_only","fast","ablation","fast"), ("spring","fast","anchor_only","fast"),
                 ("spring","deployed","anchor_only","deployed"), ("spring","deployed","spring","fast"),
                 ("spring","deployed","ablation","fast")]
    for scale in sorted({1., selection["spring"]["scale"]}):
        for a, av, b, bv in contrasts:
            ces, accs = [], []
            corpus_deltas = []
            for c in new:
                per_c = []
                for s in protocol["model_seeds"]:
                    if (c,s,a,scale) not in lookup or (c,s,b,scale) not in lookup:
                        continue
                    left = lookup[c,s,a,scale]["views"]["test"][av]
                    right = lookup[c,s,b,scale]["views"]["test"][bv]
                    per_c.append(left["overall"]["ce"] - right["overall"]["ce"])
                    accs.append(100*(left["positions"][2]["accuracy"]-right["positions"][2]["accuracy"]))
                ces.extend(per_c)
                corpus_deltas.append(f"{mean(per_c):+.4f}" if per_c else "—")
            if ces:
                paired.append([scale, f"{a} {av} − {b} {bv}", f"{mean(ces):+.4f}",
                        f"{sum(v<0 for v in ces)}/{sum(v==0 for v in ces)}/{sum(v>0 for v in ces)}",
                        f"{mean(accs):+.2f}", *corpus_deltas])

    sections = [
        "# R01 measurement tables",
        "Quality and cost are arithmetic means across corpus/model pairs. Time to target is the median among observed hits. A run can hit transiently and finish below target.",
        md_table(["Corpora","Method","Rate ×","Final PASS","Ever hit","Hit step","Val CE",
                  "Test acc %","Test CE","Train s"], overview),
        "## Per-corpus outcomes for the original methods",
        md_table(["Corpus","Method","Rate ×","Final PASS / 3","Val CE","Direction test acc %"], by_corpus),
        "## Paired spring component contrasts",
        "Each difference is left minus right. A negative CE difference favors the left state. Counts indicate lower/equal/higher CE. The last three columns average over model seeds within each corpus.",
        md_table(["Rate ×","Test comparison","Δ CE","Lower/equal/higher","Direction Δ pp", *new], paired),
        "## All clause positions",
        "Each cell gives mean accuracy and cross-entropy in nats. Positions are zero-based. Historical means cover three model seeds. New-corpus means cover nine paired runs.",
        md_table(["Corpora","Method","Rate ×","Split","Readout","0 direction","1 start","2 first move",
                  "3 second move","4 third move"], by_position)]
    (folder / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)
