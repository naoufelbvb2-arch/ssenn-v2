"""Document-disjoint CPU next-token windows with training-only vocabulary fitting."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import Tensor

from .data import documents

BOS, EOS, UNK = 0, 1, 2


@dataclass
class LanguageData:
    vocab: dict[str, int]
    train: Tensor
    validation: tuple[Tensor, Tensor, Tensor]
    test: tuple[Tensor, Tensor, Tensor]
    counts: Tensor
    transitions: Tensor
    manifest: dict


def evaluation_windows(docs: list[Tensor], context: int):
    xs, ys, offsets = [], [], []
    for doc in docs:
        for i in range(0, len(doc) - 1, context):
            part = doc[i:i + context + 1]
            n = len(part) - 1
            x = torch.full((context,), BOS)
            y = torch.full((context,), -100)
            x[:n], y[:n] = part[:-1], part[1:]
            xs.append(x)
            ys.append(y)
            offsets.append(torch.arange(i, i + context))
    return torch.stack(xs), torch.stack(ys), torch.stack(offsets)


def build_data(splits, context: int, unit: str) -> LanguageData:
    symbols = (lambda s: list(s.lower())) if unit == "character" else str.split
    vocab = {s: i for i, s in enumerate(sorted({s for _, text in splits[0] for s in symbols(text)}), 3)}
    batches, ids, fingerprints, counts = [], [], [], []
    for split in splits:
        docs = [torch.tensor([BOS, *(vocab.get(s, UNK) for s in symbols(text)), EOS]) for _, text in split]
        batches.append(docs)
        ids.append([name for name, _ in split])
        fingerprints.append([hashlib.sha256(text.lower().encode()).hexdigest() for _, text in split])
        counts.append(sum(len(d) - 1 for d in docs))
    for groups in (ids, fingerprints):
        if len(set().union(*groups)) != sum(map(len, groups)):
            raise ValueError("Duplicate document ID or content across corpus")
    train = torch.cat([d.unfold(0, context + 1, 1) for d in batches[0] if len(d) > context])
    if any(len(d) <= context for d in batches[0]):
        raise ValueError("A training document is shorter than a full sampled window")
    v = len(vocab) + 3
    pairs = torch.cat([d[:-1] * v + d[1:] for d in batches[0]])
    transitions = torch.bincount(pairs, minlength=v*v).reshape(v, v)
    manifest = {"ids": ids, "body_sha256": fingerprints, "target_tokens": counts,
                "unknown_targets": [sum(int((d == UNK).sum()) for d in split) for split in batches],
                "training_windows": len(train), "tokens": vocab, "unit": unit}
    return LanguageData(vocab, train, evaluation_windows(batches[1], context),
                        evaluation_windows(batches[2], context), transitions.sum(0), transitions, manifest)


def load_data(task: str, cfg: dict, root: Path) -> LanguageData:
    if task == "structured":
        raw = documents(**cfg[task])
        splits = [[(f"ring-{j}-{i}", text) for i, text in enumerate(docs)] for j, docs in enumerate(raw)]
        return build_data(splits, cfg["model"]["context"], "symbol")
    if task != "blake":
        raise ValueError(task)
    poems = json.loads((root / "data/c02a/blake.json").read_text())
    random.Random(cfg[task]["split_seed"]).shuffle(poems)
    a, b, c = cfg[task]["sizes"]
    if a+b+c != len(poems):
        raise ValueError("Corpus size mismatch")
    splits = [[(d["id"], d["text"]) for d in ds] for ds in (poems[:a], poems[a:a+b], poems[a+b:])]
    return build_data(splits, cfg["model"]["context"], "character")
