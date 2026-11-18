import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from ssenn.conventional import AdamNetwork
from ssenn.credit import DirectFeedback
from ssenn.data import xor
from ssenn.dynamics import Rates
from ssenn.experiment import run_trial
from ssenn.local import LocalNetwork
from ssenn.network import initial_weights


class ExperimentTests(unittest.TestCase):
    def test_matched_initial_logits_and_counted_training_state(self):
        weights = initial_weights(2, 16, 2, 11)
        adam = AdamNetwork(weights, .01, .01)
        spring = LocalNetwork(weights, DirectFeedback.random(2, 16, 12), Rates(.3, .05, .1, .02))
        x, y = xor().train
        self.assertTrue(torch.equal(adam.logits(x), spring.logits(x)))
        adam.train_step(x, y)
        self.assertEqual(adam.counts(), (82, 166))
        self.assertEqual(spring.counts(), (82, 196))

    def test_minibatch_replay_through_benchmark_entry_point(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "c01.json"
        config = json.loads(path.read_text())["tasks"]["T02"] | {"steps": 8, "eval_every": 2}
        for method in ("spring", "ablation", "adamw"):
            with self.subTest(method=method):
                a, b = (run_trial("T02", config, method, 29) for _ in range(2))
                self.assertEqual(a["history"], b["history"])
                self.assertEqual(a["test"], b["test"])
                self.assertEqual(a["fast_validation"], b["fast_validation"])

    def test_normal_local_runner_never_calls_autograd(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "c01.json"
        config = json.loads(path.read_text())["tasks"]["T01"] | {"steps": 3}
        with patch("torch.autograd.grad", side_effect=AssertionError), \
             patch("torch.autograd.backward", side_effect=AssertionError), \
             patch("torch.Tensor.backward", side_effect=AssertionError):
            result = run_trial("T01", config, "spring", 11)
        self.assertEqual([v["step"] for v in result["history"]], [0, 3])
