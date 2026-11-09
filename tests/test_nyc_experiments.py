"""Offline regressions for the five NYC-Bike experiment options."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2Model

from models.st_llm_tfp import ST_LLM_TFP, TimeFilteredPFA
from models.timefilter_attention import TimeFilterAttentionBias
from models.unica_adapter import PEMSUniCACovariateAdapter
from ranger21 import Ranger
from train_tfp import parse_args
from utils.level_prior import level_prior_arrays

WEEK = 168


def backbone():
    return GPT2Model(GPT2Config(n_embd=768, n_head=12, n_layer=2, n_positions=8, vocab_size=16,
                                attn_pdrop=0., resid_pdrop=0., embd_pdrop=0., attn_implementation="eager"))


class LoraOnlyTests(unittest.TestCase):
    def test_graph_layers_train_only_lora_and_layernorm(self):
        def trainable(**kwargs):
            with patch("models.st_llm_tfp.GPT2Model.from_pretrained", return_value=backbone()):
                model = TimeFilteredPFA(np.eye(4), device="cpu", gpt_layers=2, U=2, tfp_enabled=False, **kwargs)
            return {name for name, p in model.transformer.h.named_parameters() if p.requires_grad}

        full, lora = trainable(), trainable(lora_only=True)
        self.assertTrue(any("c_attn.base_layer" in name for name in full))
        self.assertTrue(any(".c_proj." in name for name in full))
        self.assertTrue(lora)
        self.assertTrue(all("lora_" in name or ".ln_" in name for name in lora), lora)
        self.assertTrue(any("lora_" in name for name in lora))


class LearningRateDecayTests(unittest.TestCase):
    def test_ranger_follows_the_schedule(self):
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = Ranger([torch.nn.Parameter(torch.ones(2))], lr=1.0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3], gamma=.5)
        rates = []
        for _ in range(4):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertEqual(rates, [1.0, .5, .5, .25])


class LevelPriorTests(unittest.TestCase):
    @staticmethod
    def dataset(hours=WEEK * 8):
        slots = np.arange(hours) % WEEK
        flow = np.stack([1 + slots % 24, 2 * (1 + slots % 24)], axis=1).astype(np.float32)
        origins = dict(train=np.arange(24, WEEK * 4), val=np.arange(WEEK * 5 + 24, WEEK * 6),
                       test=np.arange(WEEK * 7, hours - 24))
        return flow, slots, origins

    def test_ratio_and_prior_follow_the_last_week(self):
        flow, slots, origins = self.dataset()
        flow[WEEK * 4 + 24:] *= 2
        arrays = level_prior_arrays(flow, slots, origins)
        np.testing.assert_allclose(arrays["val"]["ratio"], 2, rtol=1e-5)
        first = origins["val"][0]
        np.testing.assert_allclose(arrays["val"]["prior"][0], flow[first:first + 24], rtol=1e-5)

    def test_prior_cannot_see_hours_at_or_after_its_origin(self):
        flow, slots, origins = self.dataset()
        origin = int(origins["val"][10])
        changed = flow.copy()
        changed[origin:] = changed[origin:] * 7 + 3
        before = level_prior_arrays(flow, slots, origins)["val"]
        after = level_prior_arrays(changed, slots, origins)["val"]
        np.testing.assert_array_equal(before["ratio"][:11], after["ratio"][:11])
        np.testing.assert_array_equal(before["prior"][:11], after["prior"][:11])
        self.assertFalse(np.array_equal(before["ratio"][11:], after["ratio"][11:]))

    def test_closure_gives_zero_then_recovers_the_previous_level(self):
        flow, slots, origins = self.dataset()
        flow[WEEK * 4 + 24:] *= 2
        start, end = WEEK * 5 + 20, WEEK * 5 + 100
        flow[start:end] = 0
        ratio = level_prior_arrays(flow, slots, origins)["val"]["ratio"]
        index = {int(origin): i for i, origin in enumerate(origins["val"])}
        self.assertTrue((ratio[index[end - 1]] == 0).all())
        np.testing.assert_allclose(ratio[index[end + 1]], 2, rtol=1e-5)


class ModelOptionTests(unittest.TestCase):
    def test_compact_timefilter_keeps_the_attention_shape(self):
        layer = TimeFilterAttentionBias(768, 12, 5, proj_dim=120)
        bias, _ = layer(torch.randn(2, 5, 768), torch.eye(5))
        self.assertEqual(tuple(bias.shape), (2, 12, 5, 5))
        self.assertEqual(layer.proj_q.out_features, 120)
        self.assertEqual(TimeFilterAttentionBias(768, 12, 5).proj_q.out_features, 768)
        with self.assertRaisesRegex(ValueError, "proj_dim"):
            TimeFilterAttentionBias(768, 12, 5, proj_dim=100)

    def test_adapter_without_static_path_learns(self):
        adapter = PEMSUniCACovariateAdapter(48, 4, 3, 3, past_dynamic_dim=2, future_dynamic_dim=2, static_dim=0,
                                            num_heads=3, bottleneck_dim=12, dropout=0)
        self.assertFalse(any("static" in name for name in adapter.state_dict()))
        covariates = dict(past_dynamic_real=torch.randn(2, 4, 3, 2), future_dynamic_real=torch.randn(2, 3, 1, 2))
        output = adapter.fuse_future(adapter.fuse_past(torch.randn(2, 3, 48), covariates), covariates)
        output.square().mean().backward()
        self.assertGreater(adapter.past_projectors[0].weight.grad.abs().sum().item(), 0)

    def test_residual_model_starts_exactly_at_the_prior(self):
        config = dict(past_dynamic_dim=1, future_dynamic_dim=2, static_dim=0, num_heads=2, bottleneck_dim=16, with_gate=True)
        with patch("models.st_llm_tfp.GPT2Model.from_pretrained", return_value=backbone()):
            model = ST_LLM_TFP("cpu", np.eye(4), num_nodes=4, input_len=3, output_len=2, llm_layer=2, U=2,
                               tfp_config=dict(proj_dim=120), unica_enabled=True, unica_config=config,
                               steps_per_day=24, level_prior="residual")
        prior = torch.randn(2, 2, 4)
        covariates = dict(past_dynamic_real=torch.randn(2, 3, 1, 1), future_dynamic_real=torch.randn(2, 2, 1, 2),
                          level_prior=prior, level_ratio=torch.randn(2, 4))
        model.eval()
        output = model(torch.rand(2, 3, 4, 3), covariates)
        torch.testing.assert_close(output.squeeze(-1), prior, rtol=0, atol=0)
        (output - 1).abs().mean().backward()
        self.assertGreater(model.regression_layer.weight.grad.abs().sum().item(), 0)
        self.assertEqual(len(model.unica_adapter.future_projectors), 4)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_leave_every_experiment_off(self):
        args = parse_args(["--data", "nyc_bike"])
        self.assertEqual(args.es_patience, 20)
        self.assertEqual(args.lr_milestones, [])
        self.assertFalse(args.lora_only)
        self.assertEqual(args.level_prior, "none")
        self.assertTrue(args.unica_static)
        self.assertIsNone(args.tfp_proj_dim)

    def test_invalid_combinations_are_rejected(self):
        for argv in (["--data", "nyc_bike", "--level_prior", "residual"],
                     ["--data", "pems08", "--unica_enabled", "--level_prior", "covariate"],
                     ["--tfp_proj_dim", "100"],
                     ["--lr_milestones", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
