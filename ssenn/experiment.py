"""CPU timing loop and matched evaluation for the three learners."""

from collections.abc import Callable
from dataclasses import dataclass
import math
import sys
from time import perf_counter
from typing import Protocol

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from .credit import DirectFeedback
from .data import CLAUSE_LENGTH, Examples, TaskData, token_task, xor
from .dynamics import Rates
from .local import LocalNetwork
from .network import initial_weights


class Learner(Protocol):
    def train_step(self, x: Tensor, y: Tensor) -> None: ...
    def logits(self, x: Tensor) -> Tensor: ...
    def deployed_logits(self, x: Tensor) -> Tensor: ...
    def counts(self) -> tuple[int, int]: ...


@dataclass(frozen=True)
class Trial:
    hidden: int
    steps: int
    batch_size: int
    eval_every: int
    target_accuracy: float
    rates: Rates
    adam_lr: float
    weight_decay: float


def build(cfg: Trial, data: TaskData, method: str, seed: int) -> Learner:
    weights = initial_weights(data.train[0].shape[1], cfg.hidden, data.classes, seed)
    if method == "adamw":
        from .conventional import AdamNetwork
        return AdamNetwork(weights, cfg.adam_lr, cfg.weight_decay)
    if method not in ("spring", "ablation"):
        raise ValueError(f"Unknown method {method}")
    credit = DirectFeedback.random(data.classes, cfg.hidden, seed + 1)
    return LocalNetwork(weights, credit, cfg.rates, ablated=method == "ablation")


@torch.no_grad()
def evaluate(predict: Callable[[Tensor], Tensor], batch: Examples) -> dict[str, float]:
    x, y = batch
    scores = predict(x)
    loss = cross_entropy(scores, y).item()
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite cross-entropy")
    return {"ce": loss, "ppl": math.exp(loss), "accuracy": (scores.argmax(1) == y).float().mean().item()}


def peak_process_bytes() -> int | None:
    """Lifetime high-water mark including imports, setup, warmup and evaluation."""
    if sys.platform == "win32":
        try:
            import psutil
        except ImportError:
            return None
        return psutil.Process().memory_info().peak_wset
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def run_trial(task: str, raw: dict, method: str, seed: int) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    values = raw.copy()
    if task == "T01":
        data = xor()
    elif task == "T02":
        data = token_task(**values.pop("data"))
    else:
        raise ValueError(f"Unknown task {task}")
    cfg = Trial(**(values | {"rates": Rates(**values["rates"])}))
    x, y = data.train
    generator = torch.Generator().manual_seed(seed + 2)
    full_batch = cfg.batch_size == len(x)
    model = build(cfg, data, method, seed)
    warmup = build(cfg, data, method, seed)
    warmup.train_step(x[:cfg.batch_size], y[:cfg.batch_size])
    del warmup
    history = []
    training_seconds = 0.
    reached = None
    initial = evaluate(model.deployed_logits, data.validation)
    for step in range(cfg.steps + 1):
        if step:
            start = perf_counter()
            if full_batch:
                model.train_step(x, y)
            else:
                ids = torch.randint(len(x), (cfg.batch_size,), generator=generator)
                model.train_step(x[ids], y[ids])
            training_seconds += perf_counter() - start
        if step % cfg.eval_every == 0 or step == cfg.steps:
            scores = evaluate(model.deployed_logits, data.validation)
            history.append({"step": step, **scores})
            if reached is None and scores["accuracy"] >= cfg.target_accuracy:
                reached = {"step": step, "training_seconds": training_seconds}
    final = history[-1]
    deployed, extra = model.counts()
    test = evaluate(model.deployed_logits, data.test)
    diagnostics = {}
    if task == "T02":
        # Exact document positions after dropping each document's warm-up context.
        corpus = raw["data"]
        positions = torch.arange(corpus["context"], corpus["clauses"] * CLAUSE_LENGTH)
        phase = positions.repeat(corpus["sizes"][2]) % CLAUSE_LENGTH
        correct = model.deployed_logits(data.test[0]).argmax(1) == data.test[1]
        diagnostics["test_accuracy_by_clause_position"] = [
            correct[phase == pos].float().mean().item() for pos in range(CLAUSE_LENGTH)
        ]
    return {
        "task": task, "method": method, "seed": seed, "initial": initial,
        "validation": final, "test": test, "fast_validation": evaluate(model.logits, data.validation),
        "history": history, "first_target": reached,
        "status": "PASS" if (final["accuracy"] >= cfg.target_accuracy and final["ce"] < initial["ce"])
                  else "FAIL",
        "training_seconds": training_seconds, "items_per_second": cfg.steps * cfg.batch_size / training_seconds,
        "deployed_elements": deployed, "training_only_elements": extra,
        "split_items": [len(batch[1]) for batch in (data.train, data.validation, data.test)],
        "peak_process_bytes": peak_process_bytes(), **diagnostics,
    }
