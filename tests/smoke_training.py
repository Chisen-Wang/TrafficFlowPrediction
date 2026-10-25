"""Run the actual training entrypoint with additional integration assertions.

Example: python -B tests/smoke_training.py --data pems07 --epochs 1
  --batch_size 1 --max_train_batches 2 --max_eval_batches 1 --save logs/smoke-tfp-
All arguments are passed to train_tfp; use small batch/epoch limits explicitly.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import torch
import train_tfp


def main():
    evidence = {"forward_calls": 0, "optimizer_steps": 0, "finite_gradients": True}
    captured = {}
    original_init = train_tfp.Trainer.__init__

    def checked_init(self, scaler, adj_mx, args, device):
        original_init(self, scaler, adj_mx, args, device)
        captured.update(engine=self, args=args)
        if args.unica_enabled and args.unica_training_policy == "adapter_only":
            captured["initial_state"] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            assert all(p.requires_grad == name.startswith("unica_adapter.")
                       for name, p in self.model.named_parameters())

        def forward_check(model, inputs, output):
            result = output[0] if isinstance(output, tuple) else output
            batch = inputs[0].shape[0]
            assert result.shape == (batch, args.output_len, args.num_nodes, 1), result.shape
            assert torch.isfinite(result).all(), "Nonfinite model output"
            assert model.last_attention_bias_shape == (batch, 12, args.num_nodes, args.num_nodes)
            evidence["output_shape"] = list(result.shape)
            evidence["attention_shape"] = list(model.last_attention_bias_shape)
            evidence["forward_calls"] += 1

        self.model.register_forward_hook(forward_check)
        original_step = self.optimizer.step

        def checked_step(*step_args, **kwargs):
            grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), "Missing or nonfinite gradients"
            evidence["optimizer_steps"] += 1
            return original_step(*step_args, **kwargs)

        self.optimizer.step = checked_step

    train_tfp.Trainer.__init__ = checked_init
    train_tfp.main()
    engine, args = captured["engine"], captured["args"]
    run_prefix = Path(args.save).expanduser()
    if not run_prefix.is_absolute():
        run_prefix = train_tfp.SCRIPT_DIR / run_prefix
    run_dir = Path(str(run_prefix) + args.data)
    for filename in ("train.csv", "test.csv"):
        frame = pd.read_csv(run_dir / filename, index_col=0)
        assert np.isfinite(frame.to_numpy()).all(), f"Nonfinite metrics in {filename}"
        if filename == "test.csv":
            assert len(frame) == args.output_len + 1
    if "initial_state" in captured:
        changed = [k for k, v in engine.model.state_dict().items()
                   if not torch.equal(v.cpu(), captured["initial_state"][k])]
        assert changed and all(k.startswith("unica_adapter.") for k in changed), changed
        evidence["adapter_only_changed_tensors"] = len(changed)
        evidence["backbone_unchanged"] = True
    evidence["dataset"] = args.data
    evidence["past_dynamic_dim"] = args.unica_past_dynamic_dim
    evidence["checkpoint_reloaded"] = True
    evidence["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    (run_dir / "smoke_checks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("SMOKE CHECKS PASSED", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
