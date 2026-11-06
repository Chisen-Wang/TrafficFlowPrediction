"""Offline regression for the GPT-2 blocks invoked outside GPT2Model.forward."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2Model

from models.st_llm_tfp import TimeFilteredPFA


class ModelAttentionTests(unittest.TestCase):
    def test_prefix_layer_cannot_attend_to_later_node_tokens(self):
        torch.manual_seed(0)
        backbone = GPT2Model(GPT2Config(
            n_embd=24, n_head=4, n_layer=2, n_positions=8,
            vocab_size=16, attn_pdrop=0., resid_pdrop=0., embd_pdrop=0.,
            attn_implementation="eager",
        ))
        with patch("models.st_llm_tfp.GPT2Model.from_pretrained", return_value=backbone):
            model = TimeFilteredPFA(np.eye(5), device="cpu", gpt_layers=2, U=1, dropout_rate=0.)
        model.eval()
        prefix_outputs = []

        def capture_prefix(module, inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            prefix_outputs.append(hidden.detach().clone())

        handle = model.transformer.h[0].register_forward_hook(capture_prefix)
        tokens = torch.randn(1, 5, 24)
        perturbed = tokens.clone()
        perturbed[:, -1] += torch.arange(24, dtype=tokens.dtype)
        try:
            with torch.no_grad():
                model(tokens)
                model(perturbed)
        finally:
            handle.remove()

        torch.testing.assert_close(prefix_outputs[0][:, :-1], prefix_outputs[1][:, :-1], rtol=0., atol=1e-6)
        self.assertFalse(torch.allclose(prefix_outputs[0][:, -1], prefix_outputs[1][:, -1]))


if __name__ == "__main__":
    unittest.main()
