"""R01 trial construction, timing and validation. The training loop cannot score test data."""

from dataclasses import dataclass, replace
from time import perf_counter

import torch

from ssenn.conventional import AdamNetwork
from ssenn.data import Examples, TaskData, documents, examples, vocabulary
from ssenn.dynamics import Rates
from ssenn.experiment import Trial, build
from ssenn.local import LocalNetwork

from .r01_metrics import breakdown, credit_cosine, digest, flip_accuracy, flip_examples, positions, score


@dataclass
class Corpus:
    seed: int
    data: TaskData
    vocab: dict[str, int]
    phases: tuple[torch.Tensor, ...]
    doc_ids: list[list[str]]

    @classmethod
    def make(cls, seed: int, settings: dict) -> "Corpus":
        args = settings | {"seed": seed}
        splits = documents(seed, args["sizes"], args["clauses"])
        vocab = vocabulary(splits[0])
        batches = [examples(docs, vocab, args["context"]) for docs in splits]
        phases = tuple(positions(len(docs), args["clauses"], args["context"]) for docs in splits)
        return cls(seed, TaskData(*batches, len(vocab) + 1), vocab, phases,
                   [[digest(doc) for doc in docs] for docs in splits])


def make_model(cfg: Trial, data: TaskData, method: str, multiplier: float,
               seed: int) -> LocalNetwork | AdamNetwork:
    rates = replace(cfg.rates, eta_w=cfg.rates.eta_w * multiplier)
    if method == "anchor_only":
        rates = replace(rates, beta_k=0.)
    if method == "restore_only":
        rates = replace(rates, eta_0=0.)
    variant = "spring" if method in ("anchor_only", "restore_only") else method
    settings = replace(cfg, rates=rates, adam_lr=cfg.adam_lr * multiplier)
    return build(settings, data, variant, seed)


def predictor(model: LocalNetwork | AdamNetwork, method: str):
    return model.logits if method == "restore_only" else model.deployed_logits


def state_bytes(model: LocalNetwork | AdamNetwork) -> dict:
    if isinstance(model, LocalNetwork):
        weights = [model.first.w, model.second.w]
        extra = [model.first.w0, model.first.k, model.second.w0, model.second.k, model.credit.matrix]
    else:
        weights = model.weights
        extra = [v for state in model.optimizer.state.values() for v in state.values()]
    return {key: sum(t.numel() * t.element_size() for t in tensors)
            for key, tensors in (("deployed", weights), ("additional_training", extra))}


def train(model: LocalNetwork | AdamNetwork, method: str, seed: int, cfg: Trial,
          training: Examples, validation: Examples, phase: torch.Tensor,
          vocab: dict[str, int], probe_steps: list[int]) -> dict:
    x, y = training
    vx, vy = validation
    predict = predictor(model, method)
    subset = vx[phase == 2], vy[phase == 2]
    flipped = flip_examples(*subset, vocab)
    sampler = torch.Generator().manual_seed(seed + 2)
    elapsed = 0.
    history, probes = [], []
    first_hit = None
    for step in range(cfg.steps + 1):
        if step:
            start = perf_counter()
            indices = torch.randint(len(x), (cfg.batch_size,), generator=sampler)
            model.train_step(x[indices], y[indices])
            elapsed += perf_counter() - start
        if step % cfg.eval_every == 0 or step == cfg.steps:
            logits = predict(vx)
            overall = score(logits, vy)
            direction = score(logits[phase == 2], vy[phase == 2])
            history.append([step, overall["ce"], overall["accuracy"], direction["ce"], direction["accuracy"]])
            if first_hit is None and overall["accuracy"] >= cfg.target_accuracy:
                first_hit = {"step": step, "training_seconds": elapsed}
        if step in probe_steps:
            probe = {"step": step, "both_directions_correct": flip_accuracy(predict, subset, flipped)}
            if isinstance(model, LocalNetwork):
                probe["dfa_cosine_fast"] = credit_cosine(model, subset)
            probes.append(probe)
    passed = history[-1][2] >= cfg.target_accuracy and history[-1][1] < history[0][1]
    return {"status": "PASS" if passed else "FAIL", "first_hit": first_hit,
            "training_seconds": elapsed, "tokens_per_second": cfg.batch_size * cfg.steps / elapsed,
            "state_bytes": state_bytes(model), "validation_history": history, "validation_probes": probes,
            "validation": breakdown(logits, vy, phase)}


def trial_config(raw: dict) -> Trial:
    return Trial(**{k: v for k, v in raw.items() if k not in ("rates", "data")}, rates=Rates(**raw["rates"]))
