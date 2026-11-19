import json
import unittest
from unittest.mock import patch

import torch
from torch.nn.functional import cross_entropy

from benchmarks.r02 import ROOT, Setting, make_model
from benchmarks.r02_measure import force_probe, persistent_bytes
from benchmarks.r01_trials import Corpus


class SpringDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads((ROOT / "configs/c01.json").read_text())["tasks"]["T02"]
        cls.p = json.loads((ROOT / "configs/c01_r02.json").read_text())
        cls.data = Corpus.make(3109, cls.base["data"]).data

    def model(self, setting):
        return make_model(setting, 11, self.data, self.p, self.base)

    def test_light_and_moving_anchor_match_legacy_over_a_full_training_run(self):
        light = self.model(Setting("light", "light"))
        anchor = self.model(Setting("anchor", "anchor", .05))
        reference = self.model(Setting("legacy", "spring", .05, 0., .1))
        x, y = self.data.train
        rng = torch.Generator().manual_seed(13)
        with patch("torch.autograd.grad", side_effect=AssertionError), \
             patch("torch.autograd.backward", side_effect=AssertionError):
            for _ in range(1000):
                ids = torch.randint(len(x), (64,), generator=rng)
                for model in (reference, light, anchor):
                    model.train_step(x[ids], y[ids])
            for lw, aw, aa, ref in zip(light.weights, anchor.weights, anchor.anchors,
                                       (reference.first, reference.second)):
                self.assertTrue(torch.equal(lw, ref.w) and torch.equal(aw, ref.w))
                self.assertTrue(torch.equal(aa, ref.w0))
            self.assertTrue(torch.equal(light.logits(x[:64].requires_grad_()), reference.logits(x[:64])))
            self.assertFalse(light.logits(x[:64].requires_grad_()).requires_grad)

    def test_light_persistent_tensor_inventory(self):
        light = self.model(Setting("light", "light"))
        anchor = self.model(Setting("anchor", "anchor", .05))
        self.assertIsNone(light.anchors)
        self.assertEqual(set(vars(light)), {"weights", "anchors", "credit", "eta_w", "anchor_rate"})
        self.assertEqual(persistent_bytes(light), {"deployed": 8868, "training_only": 1728})
        self.assertEqual(persistent_bytes(anchor), {"deployed": 8868, "training_only": 10596})

    def test_force_probe_matches_real_and_drive_only_fast_updates(self):
        model = self.model(Setting("reference", "spring", .05, .02, .1))
        x, y = self.data.train
        for start in range(0, 640, 64):
            model.train_step(x[start:start+64], y[start:start+64])
        batch = x[640:704], y[640:704]
        snapshot = tuple(t.clone() for l in (model.first, model.second) for t in vars(l).values())
        probe = force_probe(model, batch, self.data.validation)
        for before, now in zip(snapshot, (t for l in (model.first, model.second) for t in vars(l).values())):
            self.assertTrue(torch.equal(before, now))
        light = self.model(Setting("light", "light"))
        light.weights = model.first.w.clone(), model.second.w.clone()
        light.train_step(*batch)
        model.train_step(*batch)
        vx, vy = self.data.validation
        self.assertEqual(cross_entropy(light.logits(vx), vy).item(), probe["val_ce_drive"])
        self.assertEqual(cross_entropy(model.logits(vx), vy).item(), probe["val_ce_coupled"])
        self.assertGreater(probe["elastic_l2"], 0)

    def test_zero_force_still_allows_stiffness_to_gate_consolidation(self):
        off = self.model(Setting("no_growth", "spring", .05, 0., .1))
        gated = self.model(Setting("no_restore", "spring", .05, .02, 0.))
        batch = tuple(t[:64] for t in self.data.train)
        for _ in range(20):
            for model in (off, gated):
                model.train_step(*batch)
        self.assertTrue(torch.equal(off.first.w, gated.first.w))
        self.assertFalse(torch.equal(off.first.w0, gated.first.w0))
        for model in (off, gated):
            probe = force_probe(model, batch, self.data.validation)
            self.assertEqual(probe["elastic_l2"], 0)
            self.assertEqual(probe["val_ce_drive"], probe["val_ce_coupled"])
