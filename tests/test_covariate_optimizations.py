"""Numerical and gradient regressions for TimeFilter normalization and compact UniCA."""
import contextlib
import io
import math
import unittest

import torch
import torch.nn.functional as F

from models.timefilter_attention import TimeFilterAttentionBias
from models.unica_adapter import PEMSUniCACovariateAdapter


class TimeFilterNormalizationTests(unittest.TestCase):
    def test_similarity_ignores_residual_stream_scale_and_offset(self):
        torch.manual_seed(1)
        layer = TimeFilterAttentionBias(24, 4, 5).double()
        tokens = torch.randn(2, 5, 24, dtype=torch.float64)
        expected = layer._dynamic_similarity(tokens)
        # Changes per-node magnitude and mean without changing its normalized
        # feature pattern, as can happen along the GPT residual stream.
        scale = torch.linspace(10, 1000, 5, dtype=tokens.dtype)[None, :, None]
        actual = layer._dynamic_similarity(tokens * scale + 500)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

        tokens = (tokens * 1000).requires_grad_()
        bias, auxiliary = layer(tokens, torch.eye(5), is_training=True)
        allowed = bias > torch.finfo(bias.dtype).min / 2
        (bias.masked_fill(~allowed, 0).square().mean() + auxiliary).backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_normalization_can_be_disabled_without_changing_checkpoint_keys(self):
        layer = TimeFilterAttentionBias(24, 4, 5, input_norm=False)
        normalized = TimeFilterAttentionBias(24, 4, 5)
        normalized.load_state_dict(layer.state_dict(), strict=True)
        tokens = torch.randn(2, 5, 24)
        q = layer.proj_q(tokens).reshape(2, 5, 4, 6).transpose(1, 2)
        k = layer.proj_k(tokens).reshape(2, 5, 4, 6).transpose(1, 2)
        legacy = F.gelu(q @ k.transpose(-1, -2) / math.sqrt(6))
        torch.testing.assert_close(layer._dynamic_similarity(tokens), legacy, rtol=0, atol=0)


class CompactUniCATests(unittest.TestCase):
    @staticmethod
    def covariates():
        return {
            "past_dynamic_real": torch.randn(2, 4, 3, 2),
            "future_dynamic_real": torch.randn(2, 3, 1, 4),
            "static_real": torch.randn(3, 3),
        }

    def test_gated_initialization_is_identity_and_both_fusions_learn(self):
        for gated in (False, True):
            with self.subTest(gated=gated):
                torch.manual_seed(2)
                adapter = PEMSUniCACovariateAdapter(
                    48, 4, 3, 3, past_dynamic_dim=2, future_dynamic_dim=4,
                    static_dim=3, num_heads=3, bottleneck_dim=12,
                    dropout=0, with_gate=gated,
                )
                nodes = torch.randn(2, 3, 48)
                cov = self.covariates()
                if gated:
                    torch.testing.assert_close(adapter.fuse_past(nodes, cov), nodes, rtol=0, atol=0)
                    torch.testing.assert_close(adapter.fuse_future(nodes, cov), nodes, rtol=0, atol=0)
                optimizer = torch.optim.SGD(adapter.parameters(), lr=.05)
                target = torch.randn_like(nodes)
                for _ in range(2):
                    optimizer.zero_grad()
                    output = adapter.fuse_future(adapter.fuse_past(nodes, cov), cov)
                    self.assertEqual(output.shape, nodes.shape)
                    (output - target).square().mean().backward()
                    optimizer.step()
                # The second update must reach through the initially closed gates
                # and the new projections to all three covariate families.
                parameters = dict(adapter.named_parameters())
                for name in ("node_projection.weight", "output_projection.weight",
                             "past_projectors.0.weight", "future_projectors.0.weight",
                             "static_projectors.0.weight", "future_attention.in_proj_weight"):
                    grad = parameters[name].grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_nyc_parameter_budget_and_legacy_layout(self):
        kwargs = dict(hidden_dim=768, input_len=24, output_len=24, num_nodes=326,
                      past_dynamic_dim=7, future_dynamic_dim=5, static_dim=10,
                      num_heads=4, with_gate=True)
        legacy = PEMSUniCACovariateAdapter(**kwargs, bottleneck_dim=768)
        compact = PEMSUniCACovariateAdapter(**kwargs)
        old_count = sum(p.numel() for p in legacy.parameters())
        new_count = sum(p.numel() for p in compact.parameters())
        self.assertEqual(old_count, 19798579)
        self.assertEqual(compact.bottleneck_dim, 128)
        self.assertLess(new_count, old_count // 10)
        self.assertFalse(any(k.startswith(("node_projection.", "output_projection."))
                             for k in legacy.state_dict()))

    def test_configuration_reaches_parser_and_rejects_invalid_head_width(self):
        from train_tfp import parse_args
        args = parse_args(["--unica_enabled"])
        self.assertTrue(args.tfp_input_norm)
        self.assertEqual(args.unica_bottleneck_dim, 128)
        args = parse_args(["--unica_enabled", "--no-tfp_input_norm",
                           "--unica_bottleneck_dim", "768"])
        self.assertFalse(args.tfp_input_norm)
        self.assertEqual(args.unica_bottleneck_dim, 768)
        for width in (0, 7, 769):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--unica_enabled", "--unica_bottleneck_dim", str(width)])
        with self.assertRaisesRegex(ValueError, "divisible"):
            PEMSUniCACovariateAdapter(48, 4, 3, 3, num_heads=5, bottleneck_dim=12)


if __name__ == "__main__":
    unittest.main()
