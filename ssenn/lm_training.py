"""C02-A learner selection and checkpoints taken at completed update boundaries."""

from dataclasses import asdict
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .decoder import DecoderConfig
from .decoder_local import LocalDecoder


def learner(method: str, net: DecoderConfig, seed: int, cfg: dict):
    if method == "adamw":
        from .decoder_adam import AdamDecoder
        return AdamDecoder(net, seed, cfg["adam_lr"], cfg["weight_decay"])
    if method not in ("dfa", "anchor"):
        raise ValueError(method)
    return LocalDecoder(net, seed, cfg["local_rate"], cfg["anchor_rate"] if method == "anchor" else None)


def checkpoint(path: Path, model, sampler: torch.Generator, step: int, run_id: str) -> None:
    state = {"version": 1, "run_id": run_id, "architecture": asdict(model.net.cfg),
             "step": step, "sampler": sampler.get_state(), "weights": model.state_dict()}
    if hasattr(model, "optimizer"):
        state["optimizer"] = model.optimizer.state_dict()
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def restore(path: Path, model, sampler: torch.Generator, run_id: str) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["run_id"] != run_id or state["architecture"] != asdict(model.net.cfg):
        raise ValueError("Checkpoint does not match the run")
    model.load_state_dict(state["weights"], strict=True)
    if hasattr(model, "optimizer"):
        model.optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler"])
    return state["step"]
