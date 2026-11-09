"""Average the test predictions of several finished train_tfp.py runs and score the average.

Usage (repo root, conda env dl):
    python -B experiments/ensemble_eval.py logs/tune-A-.../ logs/tune-B-.../ [--out ensemble.json]

Each run dir needs run_config.json and best_model.pth. All runs must use the same dataset.
Predictions are averaged in raw flow units; metrics use the same masked functions as training.
With --split train the training windows are kept in their file order, and --save_pred writes the
averaged prediction as [samples, horizon, nodes] float32, the layout of y[..., 0] (distillation targets).
"""
import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_tfp  # noqa: E402
from train_tfp import Trainer, load_dataset, load_graph_data, prepare_batch, resolve_data_dir  # noqa: E402
from utils import metrics  # noqa: E402


def run_args(run_dir: Path):
    args = train_tfp.parse_args([])
    for key, value in json.loads((run_dir / "run_config.json").read_text()).items():
        setattr(args, key, value)
    return args


def predict(run_dir: Path, dataloader, adj_mx, device, split="test"):
    args = run_args(run_dir)
    if args.unica_enabled:
        args.unica_past_dynamic_dim = dataloader["covariate_dims"]["past_dynamic_dim"]
    engine = Trainer(dataloader["scaler"], adj_mx, args, device)
    state = torch.load(run_dir / "best_model.pth", map_location=device)
    if engine.model.temporal_encoder is not None:
        # Runs trained 2026-11-09 03:15-05:50 also saved an unused start_conv (a since-fixed bug); it never fed the output.
        state = {k: v for k, v in state.items() if not k.startswith("start_conv.")}
    engine.model.load_state_dict(state)
    engine.model.eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader[f"{split}_loader"].get_iterator():
            testx, _, covariates = prepare_batch(batch, device)
            outputs.append(engine.model(testx, covariates, is_training=False).transpose(1, 3).squeeze(1))
    yhat = torch.cat(outputs)[: dataloader[f"y_{split}"].shape[0]]
    pred = dataloader["scaler"].inverse_transform(yhat)  # [S, N, T]
    del engine
    torch.cuda.empty_cache()
    return pred


def score(pred, realy):
    rows = []
    for h in range(pred.shape[-1]):
        mae, mape, rmse, wmape = metrics.metric(pred[..., h], realy[..., h])
        rows.append(dict(horizon=h + 1, mae=mae, rmse=rmse, mape=mape, wmape=wmape))
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ("mae", "rmse", "mape", "wmape")}
    return rows, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--save_pred", type=Path, default=None)
    cli = parser.parse_args()

    device = torch.device(cli.device)
    first = run_args(cli.runs[0])
    data_dir = resolve_data_dir(ROOT, first.data_dir)
    adj_mx = np.asarray(load_graph_data(data_dir / "adj_mx.pkl"))
    covariates_dir = resolve_data_dir(ROOT, first.unica_covariates) if first.unica_covariates else None
    # load_dataset shuffles train/val with np.random.permutation; identity keeps file order for --split train.
    keep_order = mock.patch.object(np.random, "permutation", np.arange) if cli.split == "train" else mock.MagicMock()
    with keep_order:
        dataloader = load_dataset(
            data_dir, first.batch_size, first.batch_size, first.batch_size,
            load_covariates=first.unica_enabled, covariates_dir=covariates_dir,
            level_prior=first.level_prior != "none",
        )
    realy = torch.as_tensor(dataloader[f"y_{cli.split}"], dtype=torch.float32, device=device).transpose(1, 3)[:, 0]

    total = None
    singles = {}
    for i, run_dir in enumerate(cli.runs, 1):
        pred = predict(run_dir, dataloader, adj_mx, device, cli.split)
        singles[run_dir.name] = score(pred, realy)[1]["mae"]
        total = pred if total is None else total + pred
        _, avg = score(total / i, realy)
        print(f"[{i}] {run_dir.name}: single MAE {singles[run_dir.name]:.4f} | ensemble of {i} MAE {avg['mae']:.4f}")

    rows, avg = score(total / len(cli.runs), realy)
    for r in rows:
        print(cli.split + " horizon {horizon:d}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}".format(**r))
    print(("Ensemble of {} runs on " + cli.split + ", Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}, Test MAPE: {mape:.4f}, Test WMAPE: {wmape:.4f}"
          ).format(len(cli.runs), **avg))
    if cli.save_pred:
        np.save(cli.save_pred, (total / len(cli.runs)).transpose(1, 2).float().cpu().numpy())
    if cli.out:
        cli.out.write_text(json.dumps(dict(runs=[str(r) for r in cli.runs], singles=singles,
                                           horizons=rows, average=avg), indent=2) + "\n")


if __name__ == "__main__":
    main()
