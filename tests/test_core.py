import ast
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ssenn.credit import DirectFeedback
from ssenn.data import documents, examples, token_task, vocabulary, xor
from ssenn.dynamics import Rates, Spring
from ssenn.local import LocalNetwork
from ssenn.network import initial_weights


def learner(ablated: bool = False) -> LocalNetwork:
    return LocalNetwork(initial_weights(2, 8, 2, 11), DirectFeedback.random(2, 8, 12),
                        Rates(.2, .1, .2, .1), ablated)


class DynamicsTests(unittest.TestCase):
    def test_all_three_laws_use_reference_order(self):
        # Two scalar oracles include opposing drive, restoring force and an old anchor offset.
        state = Spring(*(torch.tensor([v]) for v in ([2., -1.], [1., .5], [.25, .5])))
        after = state.advance(torch.tensor([[2., -3.]]), Rates(.2, .5, .4, .4))
        for actual, expected in zip((after.w, after.w0, after.k),
                                    ([2.3, -1.3], [1.4875, .05], [.64, .86])):
            torch.testing.assert_close(actual, torch.tensor([expected]))

    def test_stiffness_saturates_and_full_stiffness_locks_anchor(self):
        state = Spring.initialize(torch.tensor([[0.]]))
        for _ in range(500):
            updated = state.advance(torch.tensor([[8.]]), Rates(.5, .25, .1, 2.))
            self.assertEqual(updated.k.item(), 1.)
            if state.k.item() == 1.:
                self.assertTrue(torch.equal(updated.w0, state.w0))
            state = updated
        self.assertLess(abs(state.w.item()), 50.)

    def test_bad_updates_do_not_partially_mutate_state(self):
        state = Spring.initialize(torch.ones(2, 2))
        for value in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
            with self.assertRaises(FloatingPointError):
                state.advance(torch.full_like(state.w, value), Rates(2., .1, .1, .1))
        self.assertTrue(torch.equal(state.w, torch.ones(2, 2)))
        with self.assertRaises(ValueError):
            state.advance(torch.zeros(2), Rates(.2, .1, .1, .1))
        with self.assertRaises(ValueError):
            Rates(.1, float("nan"), .1, .1)

    def test_ablation_matches_drive_only_and_keeps_initial_anchor(self):
        model = learner(ablated=True)
        x, y = xor().train
        w = model.second.w.clone()
        h = torch.cat((x, torch.ones(4, 1)), 1) @ model.first.w
        h = torch.cat((h.tanh(), torch.ones(4, 1)), 1)
        e = torch.eye(2)[y] - (h @ w).softmax(1)
        model.train_step(x, y)
        torch.testing.assert_close(model.second.w, w + .2 * h.T @ e / 4)
        self.assertTrue(torch.equal(model.second.w0, w))
        self.assertEqual(model.second.k.count_nonzero().item(), 0)
        torch.testing.assert_close(model.logits(x), model.freeze().logits(x))


class LearningPathTests(unittest.TestCase):
    def test_dfa_hidden_drive_is_direct_and_projection_fixed(self):
        model = learner()
        x, y = xor().train
        matrix = model.credit.matrix.clone()
        inputs = torch.cat((x, torch.ones(4, 1)), 1)
        h = (inputs @ model.first.w).tanh()
        residual = torch.eye(2)[y] - model.logits(x).softmax(1)
        delta = (residual @ matrix) * (1 - h.square())
        expected = model.first.w + .2 * (inputs.T @ delta / len(y))
        model.train_step(x, y)
        torch.testing.assert_close(expected, model.first.w)
        self.assertTrue(torch.equal(matrix, model.credit.matrix))

    def test_frozen_snapshot_uses_only_anchors_and_stays_fixed(self):
        x, y = xor().train
        model = learner()
        model.train_step(x, y)
        frozen = model.freeze()
        self.assertEqual(set(vars(frozen)), {"first", "second"})
        self.assertTrue(torch.equal(frozen.first, model.first.w0))
        self.assertTrue(torch.equal(frozen.second, model.second.w0))
        pred = frozen.logits(x)
        model.train_step(x, y)
        self.assertTrue(torch.equal(frozen.logits(x), pred))
        self.assertFalse(torch.equal(model.deployed_logits(x), pred))

    def test_no_autograd_in_actual_local_updates(self):
        class NoGradOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if torch.is_grad_enabled():
                    raise AssertionError(f"Grad mode enabled at {func}")
                return func(*args, **(kwargs or {}))

        x, y = xor().train
        x.requires_grad_()
        model = learner()
        with ExitStack() as stack:
            for name in ("torch.autograd.grad", "torch.autograd.backward", "torch.Tensor.backward"):
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
            stack.enter_context(NoGradOps())
            model.train_step(x, y)
            outputs = (model.logits(x), model.deployed_logits(x), model.freeze().logits(x))
        for value in (*outputs, *vars(model.first).values(), *vars(model.second).values()):
            self.assertIsNone(value.grad_fn)
            self.assertFalse(value.requires_grad)

    def test_local_source_has_no_autodiff_entry_points(self):
        root = Path(__file__).resolve().parents[1] / "ssenn"
        forbidden = {"autograd", "backward", "requires_grad_", "enable_grad", "set_grad_enabled",
                     "func", "jvp", "vjp", "jacrev", "jacfwd", "grad", "conventional"}
        for module in ("local", "network", "credit", "dynamics"):
            tree = ast.parse((root / f"{module}.py").read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, forbidden)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names] + [getattr(node, "module", "") or ""]
                    for name in names:
                        self.assertTrue(forbidden.isdisjoint(name.split(".")), name)

    def test_seeded_replay_and_independent_initialization(self):
        x, y = xor().train
        a, b = learner(), learner()
        self.assertFalse(torch.equal(a.first.w, initial_weights(2, 8, 2, 29)[0]))
        for _ in range(12):
            a.train_step(x, y)
            b.train_step(x, y)
        for original, replay in zip((a.first, a.second), (b.first, b.second)):
            for name, value in vars(original).items():
                self.assertTrue(torch.equal(value, getattr(replay, name)))


class DataTests(unittest.TestCase):
    def test_split_unique_documents_and_repeatable_examples(self):
        splits = documents(2026, (96, 24, 24), 8)
        self.assertEqual(list(map(len, splits)), [96, 24, 24])
        self.assertEqual(len(set().union(*splits)), 144)
        a, b = (token_task(2026, (96, 24, 24), 8, 4) for _ in range(2))
        for left, right in zip((a.train, a.validation, a.test), (b.train, b.validation, b.test)):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(left, right)))
        self.assertEqual([len(batch[1]) for batch in (a.train, a.validation, a.test)], [3456, 864, 864])

    def test_vocab_is_fit_on_train_and_windows_do_not_cross_documents(self):
        with patch("ssenn.data.documents", return_value=[["a b c"], ["d b c"], ["a b d"]]):
            data = token_task(1, (1, 1, 1), 1, 2)
        self.assertEqual(data.classes, 4)
        self.assertEqual(data.validation[0].reshape(2, 4).argmax(1).tolist(), [0, 2])
        self.assertEqual(data.test[1].tolist(), [0])
        vocab = vocabulary(["a b c"])
        x, y = examples(["a b c", "c b a"], vocab, 2)
        self.assertEqual(x.reshape(2, 2, 4).argmax(2).tolist(), [[1, 2], [3, 2]])
        self.assertEqual(y.tolist(), [3, 1])


if __name__ == "__main__":
    unittest.main()
