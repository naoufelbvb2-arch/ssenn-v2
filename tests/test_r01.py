import json
from dataclasses import replace
import unittest

import torch
from torch.nn.functional import cross_entropy, cosine_similarity, one_hot

from benchmarks.r01 import ROOT, manifest, run_one, select
from benchmarks.r01_metrics import breakdown, credit_cosine, flip_examples
from benchmarks.r01_trials import Corpus, make_model, predictor, trial_config
from ssenn.network import forward


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw = json.loads((ROOT / "configs/c01.json").read_text())["tasks"]["T02"]
        cls.cfg = trial_config(cls.raw)
        cls.corp = Corpus.make(3109, cls.raw["data"])

    def test_independent_corpus_and_model_rngs_in_crossed_design(self):
        second = Corpus.make(5471, self.raw["data"])
        cfg = replace(self.cfg, steps=8)
        pairs = {}
        for corpus in (self.corp, second):
            audit = manifest({corpus.seed: corpus}, cfg, [11, 29])
            pairs[corpus.seed] = audit["model_rngs"]
            a, b = (make_model(cfg, corpus.data, "spring", 1., s) for s in (11, 29))
            self.assertFalse(torch.equal(a.first.w, b.first.w))
            self.assertFalse(torch.equal(a.credit.matrix, b.credit.matrix))
        self.assertEqual(pairs[3109], pairs[5471])
        self.assertNotEqual(pairs[3109][11]["batch_indices"], pairs[3109][29]["batch_indices"])
        self.assertNotEqual(self.corp.doc_ids, second.doc_ids)
        self.assertFalse(torch.equal(self.corp.data.train[0], second.data.train[0]))
        both = manifest({3109: self.corp, 5471: second}, cfg, [11])
        self.assertEqual(both["distinct_documents"], 288)

    def test_phase_scoring_and_counterfactual_are_document_local(self):
        x, y = self.corp.data.validation
        phase = self.corp.phases[1]
        sx, sy = x[phase == 2], y[phase == 2]
        flipped, fy = flip_examples(sx, sy, self.corp.vocab)
        self.assertTrue((fy != sy).all())
        self.assertTrue(torch.equal(flipped.reshape(-1, 4, 9)[:, [0, 1, 3]],
                                    sx.reshape(-1, 4, 9)[:, [0, 1, 3]]))
        restored, ry = flip_examples(flipped, fy, self.corp.vocab)
        self.assertTrue(torch.equal(restored, sx) and torch.equal(ry, sy))
        logits = one_hot(y, 9).float()
        logits[phase == 2] *= -1
        scores = breakdown(logits, y, phase)
        self.assertEqual([p["n"] for p in scores["positions"]], [168, 168, 168, 168, 192])
        self.assertEqual([p["accuracy"] for p in scores["positions"]], [1., 1., 0., 1., 1.])
        self.assertAlmostEqual(scores["overall"]["ce"], sum(p["n"] * p["ce"] for p in
                               scores["positions"]) / len(y), places=5)

    def test_factorial_controls_and_fast_weight_identity(self):
        x, y = (t[:64] for t in self.corp.data.train)
        names = ("ablation", "anchor_only", "restore_only", "spring")
        models = {m: make_model(self.cfg, self.corp.data, m, 1., 11) for m in names}
        original = models["spring"].first.w0.clone()
        for _ in range(15):
            for model in models.values():
                model.train_step(x, y)
        for m in names:
            layer = models[m].first
            self.assertEqual(torch.equal(original, layer.w0), m in ("ablation", "restore_only"))
            self.assertEqual(bool(layer.k.count_nonzero()), m in ("restore_only", "spring"))
        self.assertTrue(torch.equal(models["ablation"].logits(x), models["anchor_only"].logits(x)))
        self.assertTrue(torch.equal(predictor(models["restore_only"], "restore_only")(x),
                                    models["restore_only"].logits(x)))
        self.assertFalse(torch.equal(models["anchor_only"].logits(x), models["anchor_only"].deployed_logits(x)))

    def test_read_only_credit_probe_against_independent_ce_derivative(self):
        model = make_model(self.cfg, self.corp.data, "spring", 1., 29)
        x, y = (t[:39] for t in self.corp.data.train)
        probe = credit_cosine(model, (x, y))
        w = model.first.w.clone().requires_grad_()
        inputs, h, logits = forward(x, w, model.second.w)
        exact, = torch.autograd.grad(-cross_entropy(logits, y), w)
        with torch.no_grad():
            err = one_hot(y, 9) - logits.softmax(1)
            dfa = inputs.T @ ((err @ model.credit.matrix) * (1 - h[:, :-1].square())) / len(y)
            self.assertAlmostEqual(probe, cosine_similarity(dfa.flatten(), exact.flatten(), dim=0).item(), places=5)
        self.assertIsNone(model.first.w.grad)

    def test_search_and_selection_have_no_test_dependency(self):
        data = replace(self.corp.data, test=None)
        sealed = replace(self.corp, data=data)
        short = replace(self.cfg, steps=2)
        row = run_one(sealed, 11, "spring", 1., short, [], False)
        self.assertNotIn("views", row)
        rows = [row, row | {"scale": .5, "test": {"ce": -1000},
                           "validation": {"overall": {"ce": row["validation"]["overall"]["ce"] + 1}}}]
        self.assertEqual(select(rows, ["spring"], [1., .5], 1)["spring"]["scale"], 1.)
