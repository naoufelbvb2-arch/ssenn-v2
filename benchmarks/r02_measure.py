"""Read-only force and allocator measurements outside timed training."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn.functional import cross_entropy, cosine_similarity
from torch.profiler import ProfilerActivity, profile

from ssenn.conventional import AdamNetwork
from ssenn.light_dfa import LightDFA, local_drives
from ssenn.local import LocalNetwork
from ssenn.network import forward


@torch.no_grad()
def force_probe(model: LocalNetwork, batch, validation) -> dict:
    layers = model.first, model.second
    weights = tuple(s.w for s in layers)
    drives = local_drives(*batch, weights, model.credit)
    drive = tuple(model.rates.eta_w * d for d in drives)
    elastic = tuple(-model.rates.gamma * s.k * (s.w - s.w0) for s in layers)
    alone = tuple(w + d for w, d in zip(weights, drive))
    coupled = tuple(w + (d + f) for w, d, f in zip(weights, drive, elastic))
    def flatten(tensors):
        return torch.cat([t.flatten() for t in tensors])
    d, f = flatten(drive), flatten(elastic)
    dn, fn = d.norm().item(), f.norm().item()
    vx, vy = validation
    def loss(ws):
        return cross_entropy(forward(vx, *ws)[2], vy).item()
    return {
        "mean_k": flatten(s.k for s in layers).mean().item(),
        "mean_abs_displacement": flatten(s.w - s.w0 for s in layers).abs().mean().item(),
        "mean_abs_drive": d.abs().mean().item(), "mean_abs_elastic": f.abs().mean().item(),
        "drive_l2": dn, "elastic_l2": fn, "elastic_to_drive": fn / dn if dn else None,
        "elastic_drive_cosine": cosine_similarity(d, f, dim=0).item() if dn and fn else None,
        "val_ce_before": loss(weights), "val_ce_drive": loss(alone), "val_ce_coupled": loss(coupled),
    }


def persistent_bytes(model) -> dict:
    if isinstance(model, LightDFA):
        deployed = model.anchors or model.weights
        extra = [model.credit.matrix, *(model.weights if model.anchors else ())]
    elif isinstance(model, LocalNetwork):
        deployed = [model.first.w0, model.second.w0]
        extra = [model.credit.matrix, model.first.w, model.second.w, model.first.k, model.second.k]
    elif isinstance(model, AdamNetwork):
        deployed = model.weights
        extra = [v for state in model.optimizer.state.values() for v in state.values()]
    else:
        raise TypeError(type(model))
    return {key: sum(t.numel() * t.element_size() for t in tensors)
            for key, tensors in (("deployed", deployed), ("training_only", extra))}


def allocator_peak(model, batch) -> dict:
    """Peak live CPU tensor allocation over the steady-state footprint at entry."""
    for _ in range(3):
        model.train_step(*batch)
    state = persistent_bytes(model)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "allocation.json"
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model.train_step(*batch)
            with torch.profiler.record_function("r02_measured_update"):
                model.train_step(*batch)
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text(encoding="utf8"))["traceEvents"]
    window, = [e for e in events if e["name"] == "r02_measured_update" and e["ph"] == "X"]
    end = window["ts"] + window["dur"]
    allocs = sorted((e for e in events if e["name"] == "[memory]" and window["ts"] <= e["ts"] < end and
                     e["args"].get("Device Type") == 0), key=lambda e: e["ts"])
    if not allocs:
        raise RuntimeError("Trace has no CPU tensor allocation events")
    live = peak = 0
    for e in allocs:
        live += int(e["args"]["Bytes"])
        peak = max(peak, live)
    if live != 0:
        raise RuntimeError(f"Unbalanced steady-state CPU trace: {live} bytes")
    return {**state, "peak_above_entry": peak, "net_change": live, "events": len(allocs)}
