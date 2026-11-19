import ast
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.nn.functional import one_hot
from torch.utils._python_dispatch import TorchDispatchMode

from benchmarks.c02a import ROOT, evaluate, memory
from ssenn.decoder import DecoderConfig, join_heads, split_heads
from ssenn.lm_data import BOS, EOS, UNK, build_data, evaluation_windows, load_data
from ssenn.lm_training import checkpoint, learner, restore


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = json.loads((ROOT / "configs/c02a.json").read_text())
        cls.net = DecoderConfig(vocab=11, context=8, width=12, heads=3, ff=16)

    def model(self, method):
        return learner(method, self.net, 11, self.cfg)

    def test_identical_initial_logits_and_full_forward_implementation(self):
        x = torch.tensor([[0, 3, 6, 4], [0, 7, 5, 1]])
        dfa = self.model("dfa")
        for m in ("anchor", "adamw"):
            model = self.model(m)
            self.assertIs(type(model.net), type(dfa.net))
            for key, value in dfa.net.weights.items():
                self.assertTrue(torch.equal(value, model.net.weights[key]))
            self.assertTrue(torch.equal(model(x), dfa(x)))

    def test_causality_for_all_positions_and_modes_after_update(self):
        x = torch.tensor([[0, 3, 5, 4, 9, 3, 8, 6]])
        for m in ("dfa", "anchor", "adamw"):
            model = self.model(m)
            model.train_step(x[:, :-1], x[:, 1:])
            expected = model(x)
            _, trace = model.net.compute(x, model.net.weights, capture=True)
            prob = trace["prob"]
            self.assertEqual(prob.triu(1).count_nonzero().item(), 0)
            torch.testing.assert_close(prob.sum(-1), torch.ones_like(prob.sum(-1)))
            for t in range(1, x.size(1)+1):
                changed = x.clone()
                changed[:, t:] = (x[:, t:] + 3) % self.net.vocab
                self.assertTrue(torch.equal(model(changed)[:, :t], expected[:, :t]))
                torch.testing.assert_close(model(x[:, :t]), expected[:, :t], atol=2e-6, rtol=1e-5)

    def test_local_entry_points_do_not_record_or_call_autograd(self):
        class RequireNoGrad(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if torch.is_grad_enabled():
                    raise AssertionError(str(func))
                return func(*args, **(kwargs or {}))

        x = torch.tensor([[0, 3, 4, 3, 9]])
        for m in ("dfa", "anchor"):
            model = self.model(m)
            a, b = x[:, :-1], x[:, 1:]
            with ExitStack() as stack:
                for fn in ("torch.autograd.backward", "torch.autograd.grad", "torch.Tensor.backward"):
                    stack.enter_context(patch(fn, side_effect=AssertionError(fn)))
                stack.enter_context(RequireNoGrad())
                model.train_step(a, b)
                pred = model(a)
            self.assertFalse(pred.requires_grad)
            for w in model.net.parameters():
                self.assertIsNone(w.grad_fn)
                self.assertIsNone(w.grad)
                self.assertFalse(w.requires_grad)
        for source in ("decoder", "decoder_local"):
            tree = ast.parse((ROOT / f"ssenn/{source}.py").read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.Attribute):
                    self.assertNotIn(n.attr, {"autograd", "backward", "grad", "func", "enable_grad",
                                              "set_grad_enabled", "requires_grad_", "jvp", "vjp"})

    def test_attention_update_matches_local_finite_difference(self):
        model = self.model("dfa")
        x = torch.tensor([[0, 4, 3, 4], [0, 3, 7, 7]])
        y = torch.tensor([[4, 3, 4, 1], [3, 7, 7, 1]])
        old = {k: w.detach().double().clone() for k,w in model.net.weights.items()}
        logits, c = model.net.compute(x, model.net.weights, capture=True)
        e = (one_hot(y, self.net.vocab) - logits.softmax(-1)) / y.numel()
        routed = (e @ model.feedback.attention_mix).double()
        a, mask = c["a"].double(), model.net.mask[:4, :4]
        def objective():
            q,k,v = (split_heads(a @ old[n], self.net.heads) for n in ("q","k","v"))
            prob = ((q @ k.transpose(-1,-2)) / (q.size(-1)**0.5)).masked_fill(~mask, -torch.inf).softmax(-1)
            return (join_heads(prob @ v) * routed).sum().item()
        model.train_step(x, y)
        for n in ("q","k","v"):
            # Check a direction involving the whole projection, including every head.
            delta = (model.net.weights[n].double() - old[n]) / model.rate
            direction = delta / delta.norm()
            eps = 1e-5
            old[n] += eps*direction
            plus = objective()
            old[n] -= 2*eps*direction
            minus = objective()
            old[n] += eps*direction
            self.assertAlmostEqual((plus-minus)/(2*eps), (delta*direction).sum().item(), places=5)

    def test_every_group_changes_and_nonfinite_step_is_atomic(self):
        model = self.model("anchor")
        x = torch.tensor([[0, 4, 2, 6], [0, 3, 4, 2]])
        y = torch.tensor([[4, 2, 6, 1], [3, 4, 2, 1]])
        initial = {k:w.clone() for k,w in model.net.weights.items()}
        model.train_step(x, y)
        self.assertEqual(len(initial), 11)
        for k,w in model.net.weights.items():
            self.assertFalse(torch.equal(w, initial[k]), k)
            self.assertFalse(torch.equal(getattr(model.anchors, k), initial[k]), k)
        model.feedback.attention_mix.fill_(float("nan"))
        snapshot = {k:w.clone() for k,w in model.net.weights.items()}
        with self.assertRaises(FloatingPointError):
            model.train_step(x,y)
        self.assertTrue(all(torch.equal(w, snapshot[k]) for k,w in model.net.weights.items()))

    def test_checkpoint_replays_next_update_for_every_method(self):
        for m in ("dfa", "anchor", "adamw"):
            model = self.model(m)
            rng = torch.Generator().manual_seed(7)
            def step(model, rng):
                ids = torch.randint(11, (3, 8), generator=rng)
                model.train_step(ids[:, :-1], ids[:, 1:])
            step(model,rng)
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                checkpoint(path, model, rng, 1, m)
                other, rng2 = self.model(m), torch.Generator()
                with self.assertRaises(ValueError):
                    restore(path, other, rng2, "wrong corpus")
                self.assertEqual(restore(path, other, rng2, m), 1)
            step(model,rng)
            step(other,rng2)
            for k,v in model.state_dict().items():
                self.assertTrue(torch.equal(v, other.state_dict()[k]), (m,k))

    def test_tokenizer_split_shift_and_partial_evaluation_chunk(self):
        data = build_data([[("a", "abeab")], [("b", "cbaa")], [("c", "z")]], 3, "character")
        a,b = data.vocab["a"], data.vocab["b"]
        x,y,_ = data.validation
        self.assertEqual(x.tolist(), [[BOS,UNK,b],[a,a,BOS]])
        self.assertEqual(y.tolist(), [[UNK,b,a],[a,EOS,-100]])
        self.assertNotIn("z",data.vocab)
        self.assertEqual(data.train[0, :3].tolist(), [BOS,a,b])
        self.assertEqual(data.manifest["unknown_targets"], [0,1,1])
        with self.assertRaises(ValueError):
            build_data([[("a","abcdef")],[("b","ABCDEF")],[("c","xy")]], 3, "character")

    def test_real_data_and_accounting(self):
        data = load_data("blake", self.cfg, ROOT)
        self.assertEqual(list(map(len,data.manifest["ids"])), [33,6,6])
        self.assertEqual(len(set().union(*data.manifest["body_sha256"])),45)
        net = DecoderConfig(len(data.vocab)+3, **self.cfg["model"])
        model = learner("anchor", net, 11, self.cfg)
        metrics = evaluate(model, data.validation, False)
        self.assertEqual(metrics["tokens"],data.manifest["target_tokens"][1])
        state = memory(model)
        expected = sum(w.numel() for w in model.feedback.buffers()) + sum(w.numel() for w in model.net.parameters())
        self.assertEqual(state["training_only"]["elements"],expected)
