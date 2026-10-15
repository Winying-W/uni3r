"""Fast boundary failure tests; real numerics are checked by the exporter."""
import unittest
import torch
from torch import nn
from uni3r.instance.frozen_base import FrozenUni3R


class FailingBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.vggt = nn.Linear(2, 2)
        self.dropout = nn.Dropout()

    def forward(self, views):
        raise RuntimeError("deliberate base failure")


class FrozenBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.base = FailingBase()
        self.wrapper = FrozenUni3R(self.base, {})
        self.images = torch.zeros(1, 2, 3, 256, 256)
        self.K = torch.eye(3).expand(1, 2, 3, 3).clone()

    def test_parent_train_keeps_entire_base_eval_and_frozen(self):
        nn.Sequential(self.wrapper).train()
        self.assertTrue(self.wrapper.training)
        self.assertTrue(all(not m.training for m in self.base.modules()))
        self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))

    def test_hook_removed_after_original_forward_failure(self):
        original_hooks = dict(self.base.vggt._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "deliberate base failure"):
            self.wrapper(self.images, self.K, {})
        self.assertEqual(dict(self.base.vggt._forward_hooks), original_hooks)

    def test_reenabled_base_gradient_fails(self):
        self.base.vggt.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "must remain frozen"):
            self.wrapper(self.images, self.K, {})

    def test_reject_wrong_resolution_and_normalization(self):
        with self.assertRaises(ValueError):
            self.wrapper(self.images[..., :128], self.K, {})
        with self.assertRaises(ValueError):
            self.wrapper(self.images + 2, self.K, {})


if __name__ == "__main__":
    unittest.main()
