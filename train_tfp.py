import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.st_llm_tfp import ST_LLM_TFP
from ranger21 import Ranger
from utils.pems import PEMS_DATASETS
from utils import metrics
from utils.data import (
    load_dataset,
    load_graph_data,
    prepare_batch,
    resolve_data_dir,
    validate_window_lengths,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAVE_PREFIX = SCRIPT_DIR / "logs" / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}-"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TFP: ST-LLM+ with TimeFilter attention filtration")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str.lower, choices=tuple(PEMS_DATASETS) + ("nyc_bike",), default="pems04")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--input_len", type=int, default=None)
    parser.add_argument("--output_len", type=int, default=None)
    parser.add_argument("--llm_layer", type=int, default=2)
    parser.add_argument("--U", type=int, default=2)
    parser.add_argument("--wdecay", type=float, default=0.0001)
    parser.add_argument("--lr_milestones", type=int, nargs="*", default=[],
                        help="Epochs after which the learning rate is multiplied by --lr_gamma; empty keeps it constant")
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--cosine_epochs", type=int, default=0,
                        help="Cosine-anneal the learning rate to --min_lr over this many epochs, then hold; 0 disables")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear warm-up epochs before the cosine schedule")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--loss", choices=("mae", "huber"), default="mae",
                        help="Training loss on the original flow scale; evaluation always reports MAE")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--llm_lr_scale", type=float, default=1.0,
                        help="learning-rate multiplier for the trainable GPT-2 parameters")
    parser.add_argument("--input_noise", type=float, default=0.0,
                        help="std of Gaussian noise added to the normalized input flow during training")
    parser.add_argument("--input_drop", type=float, default=0.0,
                        help="fraction of input flow readings replaced by the missing value during training")
    parser.add_argument("--ema_decay", type=float, default=0.0,
                        help="Keep an exponential moving average of trainable weights and evaluate/save it; 0 disables")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Model dropout (embedding output, LoRA, final hidden state, MLP head)")
    parser.add_argument("--head_hidden", type=int, default=0,
                        help="Width of a two-layer prediction head; 0 keeps the original linear head")
    parser.add_argument("--input_skip", action="store_true",
                        help="Add a zero-initialized linear shortcut from the input flows to the outputs")
    parser.add_argument("--embed_layers", type=int, default=0,
                        help="Residual MLP blocks applied to node tokens before GPT-2; 0 keeps the original embedding")
    parser.add_argument("--temporal_layers", type=int, default=-1,
                        help="Per-step temporal transformer layers before GPT-2; -1 keeps the original flattening conv")
    parser.add_argument("--temporal_skip", action="store_true",
                        help="Add a zero-initialized linear output head on the temporal encoder tokens")
    parser.add_argument("--distill_targets", type=str, default=None,
                        help="npy of teacher flow predictions on the training windows in file order, [samples, horizon, nodes]")
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="Training target = (1 - alpha) * truth + alpha * teacher; missing (zero) truths stay zero")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--subset_seed", type=int, default=6666)
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffle training windows together with targets and covariates each epoch")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--es_patience", type=int, default=20)
    parser.add_argument("--model_selection_split", choices=("train", "val", "test"), default="val",
                        help="Split whose evaluation-mode MAE selects checkpoints and controls early stopping")
    parser.add_argument("--save", type=str, default=str(DEFAULT_SAVE_PREFIX))
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    parser.add_argument("--tfp_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tfp_alpha", type=float, default=0.1)
    parser.add_argument("--tfp_top_p", type=float, default=0.5)
    parser.add_argument("--tfp_static_weight", type=float, default=1.0)
    parser.add_argument("--tfp_dynamic_weight", type=float, default=1.0)
    parser.add_argument("--tfp_moe_loss_weight", type=float, default=0.05)
    parser.add_argument("--tfp_input_norm", action=argparse.BooleanOptionalAction, default=True,
                        help="Layer-normalize node tokens before TimeFilter similarity projection")
    parser.add_argument("--tfp_proj_dim", type=int, default=None,
                        help="TimeFilter Q/K width, a multiple of the 12 GPT-2 heads; default keeps the full 768")
    parser.add_argument("--lora_only", action="store_true",
                        help="Freeze base attention weights in graph layers so only LoRA and LayerNorm train there")

    parser.add_argument("--unica_enabled", action="store_true")
    parser.add_argument("--unica_static", action=argparse.BooleanOptionalAction, default=True,
                        help="Use static covariates in the UniCA adapter")
    parser.add_argument("--level_prior", choices=("none", "covariate", "residual"), default="none",
                        help="NYC-Bike only. covariate adds the rescaled weekly profile and level ratio as future "
                             "covariates; residual also predicts the error of that profile")
    parser.add_argument("--unica_covariates", type=str, default=None)
    parser.add_argument("--unica_with_gate", action="store_true")
    parser.add_argument("--unica_with_future", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unica_past_pos", type=str, default="pre", choices=["pre", "post"])
    parser.add_argument("--unica_future_pos", type=str, default="post", choices=["pre", "post"])
    parser.add_argument("--unica_heads", type=int, default=8)
    parser.add_argument("--unica_dropout", type=float, default=0.1)
    parser.add_argument("--unica_bottleneck_dim", type=int, default=128,
                        help="Internal UniCA width; use 768 for the legacy full-width adapter")
    parser.add_argument("--unica_past_dynamic_dim", type=int, default=None)
    parser.add_argument("--unica_future_dynamic_dim", type=int, default=None)
    parser.add_argument("--unica_static_dim", type=int, default=None)
    parser.add_argument(
        "--unica_training_policy",
        type=str,
        default="adapter_lora",
        choices=["adapter_lora", "adapter_only"],
    )
    parser.add_argument("--unica_pretrained_path", type=str, default=None)
    parser.add_argument("--steps_per_day", type=int, default=None)
    args = parser.parse_args(argv)
    if not 0 < args.train_fraction <= 1:
        parser.error("--train_fraction must be in (0, 1]")
    nyc = args.data == "nyc_bike"
    if args.input_len is None:
        args.input_len = 24 if nyc else 12
    if args.steps_per_day is None:
        args.steps_per_day = 24 if nyc else 288
    config = {"num_nodes": 326} if nyc else PEMS_DATASETS[args.data]
    metadata = {}
    if nyc:
        meta_path = resolve_data_dir(SCRIPT_DIR, args.data_dir or "data/nyc_bike_24_24") / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            config = {"num_nodes": metadata["num_nodes"]}
    if args.unica_future_dynamic_dim is None:
        args.unica_future_dynamic_dim = metadata.get("future_dynamic_dim", 5 if nyc else 4)
    if args.unica_static_dim is None:
        args.unica_static_dim = metadata.get("static_dim", 10 if nyc else 3)
    if args.num_nodes is None:
        args.num_nodes = config["num_nodes"]
    if args.num_nodes != config["num_nodes"]:
        parser.error(f"{args.data} expects --num_nodes {config['num_nodes']}, got {args.num_nodes}")
    if args.output_len is None:
        args.output_len = 24 if nyc else 12
    if args.data_dir is None:
        args.data_dir = f"data/{args.data}_{args.input_len}_{args.output_len}"
    for key in ("batch_size", "epochs", "input_len", "output_len", "steps_per_day", "llm_layer", "es_patience"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    for key in ("max_train_batches", "max_eval_batches"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.unica_enabled and args.unica_training_policy == "adapter_only" and not args.unica_pretrained_path:
        parser.error("--unica_pretrained_path is required for adapter_only training")
    if args.unica_enabled and (not 1 <= args.unica_bottleneck_dim <= 768
                              or args.unica_heads < 1
                              or args.unica_bottleneck_dim % args.unica_heads):
        parser.error("--unica_bottleneck_dim must be in [1, 768] and divisible by positive --unica_heads")
    if args.tfp_proj_dim is not None and (args.tfp_proj_dim <= 0 or args.tfp_proj_dim % 12):
        parser.error("--tfp_proj_dim must be a positive multiple of 12")
    if args.lr_milestones and (min(args.lr_milestones) <= 0 or not 0 < args.lr_gamma <= 1):
        parser.error("--lr_milestones must be positive and --lr_gamma in (0, 1]")
    if args.cosine_epochs and args.lr_milestones:
        parser.error("--cosine_epochs and --lr_milestones are mutually exclusive")
    if args.cosine_epochs < 0 or args.warmup_epochs < 0 or (args.warmup_epochs and not args.cosine_epochs):
        parser.error("--warmup_epochs needs a positive --cosine_epochs")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema_decay must be in [0, 1)")
    if args.level_prior != "none" and not (nyc and args.unica_enabled and args.unica_with_future):
        parser.error("--level_prior needs --data nyc_bike, --unica_enabled and --unica_with_future")
    return args


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


class Trainer:
    def __init__(
        self,
        scaler,
        adj_mx,
        args,
        device,
    ):
        tfp_config = {
            "alpha": args.tfp_alpha,
            "top_p": args.tfp_top_p,
            "static_weight": args.tfp_static_weight,
            "dynamic_weight": args.tfp_dynamic_weight,
            "input_norm": args.tfp_input_norm,
            "proj_dim": args.tfp_proj_dim,
        }
        unica_config = {
            "past_pos": args.unica_past_pos,
            "future_pos": args.unica_future_pos,
            "with_gate": args.unica_with_gate,
            "with_future": args.unica_with_future,
            "num_heads": args.unica_heads,
            "dropout": args.unica_dropout,
            "past_dynamic_dim": args.unica_past_dynamic_dim,
            "future_dynamic_dim": args.unica_future_dynamic_dim,
            "static_dim": args.unica_static_dim if args.unica_static else 0,
            "bottleneck_dim": args.unica_bottleneck_dim,
        }
        self.model = ST_LLM_TFP(
            device=str(device),
            adj_mx=adj_mx,
            steps_per_day=args.steps_per_day,
            input_dim=args.input_dim,
            num_nodes=args.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            llm_layer=args.llm_layer,
            U=args.U,
            tfp_enabled=args.tfp_enabled,
            tfp_config=tfp_config,
            unica_enabled=args.unica_enabled,
            unica_config=unica_config,
            lora_only=args.lora_only,
            level_prior=args.level_prior,
            dropout=args.dropout,
            head_hidden=args.head_hidden,
            input_skip=args.input_skip,
            embed_layers=args.embed_layers,
            temporal_layers=args.temporal_layers,
            temporal_skip=args.temporal_skip,
        ).to(device)

        if args.unica_enabled and args.unica_training_policy == "adapter_only":
            if not args.unica_pretrained_path:
                raise ValueError("--unica_pretrained_path is required for adapter_only training")
            checkpoint = torch.load(args.unica_pretrained_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            config_path = Path(args.unica_pretrained_path).parent / "run_config.json"
            if config_path.exists():
                saved_config = json.loads(config_path.read_text())
                for key in ("data", "input_len", "output_len", "num_nodes", "steps_per_day"):
                    if key in saved_config and saved_config[key] != getattr(args, key):
                        raise ValueError(f"Checkpoint {key} mismatch: saved {saved_config[key]}, requested {getattr(args, key)}")
            current_state = self.model.state_dict()
            mismatches = [f"{name}: saved {tuple(value.shape)}, expected {tuple(current_state[name].shape)}"
                          for name, value in checkpoint.items()
                          if name in current_state and value.shape != current_state[name].shape]
            if mismatches:
                raise ValueError("Checkpoint shape mismatch; match dataset, window and UniCA width "
                                 "(--unica_bottleneck_dim 768 for a legacy UniCA checkpoint): "
                                 + "; ".join(mismatches[:5]))
            if "gpt.static_adj" in checkpoint and not torch.equal(checkpoint["gpt.static_adj"], current_state["gpt.static_adj"]):
                raise ValueError("Checkpoint adjacency differs from this dataset; use its matching TFP checkpoint")
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            allowed_missing_prefixes = ("unica_adapter.", "gpt.tfp_bias.")
            non_adapter_missing = [
                name for name in missing if not name.startswith(allowed_missing_prefixes)
            ]
            if non_adapter_missing:
                raise ValueError(
                    "Checkpoint is missing non-adapter parameters: "
                    + ", ".join(non_adapter_missing[:10])
                )
            if unexpected:
                print("Ignoring unexpected checkpoint keys:", unexpected[:10])
            self.model.freeze_backbone_for_adapter_only()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found.")
        if args.llm_lr_scale != 1.0:
            # Pretrained GPT-2 weights (attention, layer norms, position embeddings) learn at a scaled rate;
            # the newly initialized LoRA adapters keep the base rate.
            llm = [p for n, p in self.model.named_parameters()
                   if p.requires_grad and n.startswith("gpt.gpt2.") and "lora_" not in n]
            llm_ids = {id(p) for p in llm}
            rest = [p for p in trainable_params if id(p) not in llm_ids]
            print(f"LLM params at lr x{args.llm_lr_scale}: {sum(p.numel() for p in llm)}; others: {sum(p.numel() for p in rest)}")
            trainable_params = [dict(params=rest), dict(params=llm, lr=args.lrate * args.llm_lr_scale)]
        self.optimizer = Ranger(trainable_params, lr=args.lrate, weight_decay=args.wdecay)
        self.lr_scheduler = (torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma) if args.lr_milestones else None)
        if args.cosine_epochs:
            warmup, total, floor = args.warmup_epochs, args.cosine_epochs, args.min_lr / args.lrate

            def lr_factor(epoch):  # epoch = number of finished epochs
                if epoch < warmup:
                    return (epoch + 1) / (warmup + 1)
                progress = min(1.0, (epoch - warmup) / max(1, total - warmup))
                return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        self.loss_name = args.loss
        self.huber_delta = args.huber_delta
        self.input_noise = args.input_noise
        self.input_drop = args.input_drop
        self.ema_decay = args.ema_decay
        self.ema_params = None
        self._backup = None
        if self.ema_decay > 0:
            self.ema_named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            self.ema_params = [p.detach().clone() for _, p in self.ema_named]
            self.ema_steps = 0
        self.scaler = scaler
        self.clip = args.clip
        self.count_targets = args.data == "nyc_bike"
        self.moe_loss_weight = args.tfp_moe_loss_weight if args.tfp_enabled else 0.0

        print(f"The number of parameters: {self.model.param_num()}")
        print(f"The number of trainable parameters: {self.model.count_trainable_params()}")
        print(f"TFP enabled: {args.tfp_enabled}; UniCA enabled: {args.unica_enabled}")

    def train(self, input_data, real_val, covariates=None):
        self.model.train()
        self.optimizer.zero_grad()
        if self.input_noise > 0 or self.input_drop > 0:
            # Training-only augmentation of the normalized flow channel ([B, C, N, T], channel 0).
            input_data = input_data.clone()
            flow = input_data[:, 0]
            if self.input_noise > 0:
                flow.add_(self.input_noise * torch.randn_like(flow))
            if self.input_drop > 0:
                # A dropped reading looks like a missing sensor value (raw 0 after normalization).
                missing = float(self.scaler.transform(0.0))
                flow.masked_fill_(torch.rand_like(flow) < self.input_drop, missing)

        output, aux = self.model(
            input_data,
            covariates,
            is_training=True,
            return_aux=True,
        )
        output = output.transpose(1, 3)
        real = real_val.unsqueeze(1)
        predict = self.scaler.inverse_transform(output)
        mae_loss = metrics.MAE_torch(predict, real, None if self.count_targets else 0.0)
        moe_loss = aux["moe_loss"]
        if self.loss_name == "huber":
            if self.count_targets:
                fit_loss = torch.nn.functional.huber_loss(predict, real, delta=self.huber_delta)
            else:
                mask = real > 0.0
                fit_loss = torch.nn.functional.huber_loss(
                    torch.masked_select(predict, mask), torch.masked_select(real, mask), delta=self.huber_delta)
        else:
            fit_loss = mae_loss
        objective = fit_loss + self.moe_loss_weight * moe_loss
        objective.backward()
        if self.clip is not None and self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.update_ema()

        if self.count_targets:
            stats = metrics.count_statistics(predict, real)
            return dict(metrics.count_metrics(stats), **stats, _samples=input_data.shape[0],
                        objective=objective.item(), moe_loss=moe_loss.item())
        return {
            "loss": mae_loss.item(),
            "objective": objective.item(),
            "moe_loss": moe_loss.item(),
            "mape": metrics.MAPE_torch(predict, real, 0.0).item(),
            "rmse": metrics.RMSE_torch(predict, real, 0.0).item(),
            "wmape": metrics.WMAPE_torch(predict, real, 0.0).item(),
        }

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params is None:
            return
        self.ema_steps += 1
        # Short warm-up so early epochs are not dominated by the random initialization.
        decay = min(self.ema_decay, (1 + self.ema_steps) / (10 + self.ema_steps))
        torch._foreach_lerp_(self.ema_params, [p.detach() for _, p in self.ema_named], 1 - decay)

    @torch.no_grad()
    def swap_in_ema(self):
        """Load EMA weights into the model (keeping the live weights) until swap_out_ema()."""
        if self.ema_params is None or self._backup is not None:
            return
        self._backup = [p.detach().clone() for _, p in self.ema_named]
        for (_, p), e in zip(self.ema_named, self.ema_params):
            p.copy_(e)

    @torch.no_grad()
    def swap_out_ema(self):
        if self._backup is None:
            return
        for (_, p), b in zip(self.ema_named, self._backup):
            p.copy_(b)
        self._backup = None

    def eval(self, input_data, real_val, covariates=None):
        self.model.eval()
        with torch.no_grad():
            output, aux = self.model(
                input_data,
                covariates,
                is_training=False,
                return_aux=True,
            )
            output = output.transpose(1, 3)
            real = real_val.unsqueeze(1)
            predict = self.scaler.inverse_transform(output)
            loss = metrics.MAE_torch(predict, real, None if self.count_targets else 0.0)

        if self.count_targets:
            stats = metrics.count_statistics(predict, real)
            return dict(metrics.count_metrics(stats), **stats, _samples=input_data.shape[0],
                        moe_loss=aux["moe_loss"].item())
        return {
            "loss": loss.item(),
            "moe_loss": aux["moe_loss"].item(),
            "mape": metrics.MAPE_torch(predict, real, 0.0).item(),
            "rmse": metrics.RMSE_torch(predict, real, 0.0).item(),
            "wmape": metrics.WMAPE_torch(predict, real, 0.0).item(),
        }


def run_loader(engine, loader, device, max_batches=None, training=False, print_every=0, shuffle=False):
    records = []
    if training and shuffle:
        loader.shuffle()
    iterator = loader.get_iterator()
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y, covariates = prepare_batch(batch, device)
        if training:
            record = engine.train(x, y[:, 0, :, :], covariates)
        else:
            record = engine.eval(x, y[:, 0, :, :], covariates)
        records.append(record)
        if training and print_every > 0 and (batch_idx + 1) % print_every == 0:
            print(f"Train batch {batch_idx + 1}/{loader.num_batch}, MAE={record['loss']:.4f}", flush=True)
    if not records:
        raise ValueError("No batches were processed.")
    if "_count" in records[0]:
        totals = {key: sum(r[key] for r in records) for key in records[0] if key.startswith("_")}
        result = metrics.count_metrics(totals)
        for key in ("objective", "moe_loss"):
            if key in records[0]:
                result[key] = sum(r[key] * r["_samples"] for r in records) / totals["_samples"]
        return result
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0].keys()
    }


def evaluate_horizons(engine, dataloader, scaler, device, output_len, max_batches=None, quiet=False):
    engine.model.eval()
    outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader["test_loader"].get_iterator()):
            if max_batches is not None and batch_idx >= max_batches:
                break
            testx, _, test_covariates = prepare_batch(batch, device)
            preds = engine.model(testx, test_covariates, is_training=False).transpose(1, 3)
            outputs.append(preds.squeeze(1))

    yhat = torch.cat(outputs, dim=0)
    sample_limit = yhat.size(0) if max_batches is not None else dataloader["y_test"].shape[0]
    sample_limit = min(sample_limit, yhat.size(0), dataloader["y_test"].shape[0])
    yhat = yhat[:sample_limit, ...]
    realy = torch.as_tensor(dataloader["y_test"][:sample_limit], dtype=torch.float32, device=device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    result_rows = []
    amae, amape, armse, awmape = [], [], [], []
    for horizon in range(output_len):
        pred = scaler.inverse_transform(yhat[:, :, horizon])
        real = realy[:, :, horizon]
        if getattr(engine, "count_targets", False):
            values = metrics.count_metrics(metrics.count_statistics(pred, real))
            mae, mape, rmse, wmape = (values[k] for k in ("loss", "mape", "rmse", "wmape"))
        else:
            mae, mape, rmse, wmape = metrics.metric(pred, real)
        if not quiet:
            print(
                "Evaluate best model on test data for horizon {:d}, "
                "Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                    horizon + 1, mae, rmse, mape, wmape
                )
            )
        result_rows.append(
            {
                "test_loss": mae,
                "test_rmse": rmse,
                "test_mape": mape,
                "test_wmape": wmape,
            }
        )
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)
        awmape.append(wmape)

    avg_row = {
        "test_loss": float(np.mean(amae)),
        "test_rmse": float(np.mean(armse)),
        "test_mape": float(np.mean(amape)),
        "test_wmape": float(np.mean(awmape)),
    }
    result_rows.append(avg_row)
    if not quiet:
        print(
            "On average over {:d} horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, "
            "Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                output_len,
                avg_row["test_loss"],
                avg_row["test_rmse"],
                avg_row["test_mape"],
                avg_row["test_wmape"],
            )
        )
    return result_rows


def main():
    args = parse_args()
    seed_it(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    device = torch.device(args.device)
    data_dir = resolve_data_dir(SCRIPT_DIR, args.data_dir)
    adj_path = data_dir / "adj_mx.pkl"
    adj_mx = np.asarray(load_graph_data(adj_path))
    if adj_mx.shape != (args.num_nodes, args.num_nodes) or not np.isfinite(adj_mx).all():
        raise ValueError(f"Expected finite adjacency [{args.num_nodes},{args.num_nodes}], got {adj_mx.shape}")

    covariates_dir = None
    if args.unica_covariates is not None:
        covariates_dir = resolve_data_dir(SCRIPT_DIR, args.unica_covariates)

    train_flow_targets = None
    if args.distill_targets:
        if not 0.0 < args.distill_alpha <= 1.0:
            raise ValueError("--distill_alpha must be in (0, 1]")
        with np.load(data_dir / "train.npz") as raw:
            truth = raw["y"][..., 0]
        teacher = np.load(os.path.expanduser(args.distill_targets)).astype(truth.dtype)
        if teacher.shape != truth.shape:
            raise ValueError(f"--distill_targets shape {teacher.shape} != training targets {truth.shape}")
        train_flow_targets = np.where(truth > 0, (1 - args.distill_alpha) * truth + args.distill_alpha * teacher, truth)

    dataloader = load_dataset(
        data_dir,
        args.batch_size,
        args.batch_size,
        args.batch_size,
        load_covariates=args.unica_enabled,
        covariates_dir=covariates_dir,
        train_fraction=args.train_fraction,
        subset_seed=args.subset_seed,
        level_prior=args.level_prior != "none",
        train_flow_targets=train_flow_targets,
        expected={"dataset": args.data, "num_nodes": args.num_nodes,
                  "input_dim": args.input_dim, "input_len": args.input_len,
                  "output_len": args.output_len, "steps_per_day": args.steps_per_day,
                  "past_dynamic_dim": args.unica_past_dynamic_dim,
                  "future_dynamic_dim": args.unica_future_dynamic_dim,
                  "static_dim": args.unica_static_dim},
    )
    if args.unica_enabled:
        args.unica_past_dynamic_dim = dataloader["covariate_dims"]["past_dynamic_dim"]
    validate_window_lengths(dataloader, args.input_len, args.output_len, data_dir)
    scaler = dataloader["scaler"]

    save_prefix = Path(os.path.expanduser(args.save))
    if not save_prefix.is_absolute():
        save_prefix = SCRIPT_DIR / save_prefix
    run_dir = Path(str(save_prefix) + args.data)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(args)
    print(f"data_dir={data_dir}")
    print(f"run_dir={run_dir}")
    sampling = dict(dataloader["sampling"], scaler_mean=float(scaler.mean), scaler_std=float(scaler.std))
    (run_dir / "sampling.json").write_text(json.dumps(sampling, indent=2) + "\n")
    np.save(run_dir / "train_indices.npy", dataloader["train_indices"])
    print(f"sampling={sampling}", flush=True)

    engine = Trainer(scaler, adj_mx, args, device)
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    train_rows = []
    best_valid = float("inf")
    best_selection = float("inf")
    epoch_test_rows = []
    best_epoch = 0
    epochs_since_best = 0
    train_time, val_time = [], []

    print("start training...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t1 = time.time()
        train_metrics = run_loader(
            engine,
            dataloader["train_loader"],
            device,
            max_batches=args.max_train_batches,
            training=True,
            print_every=args.print_every,
            shuffle=args.shuffle_train,
        )
        t2 = time.time()
        train_time.append(t2 - t1)
        engine.swap_in_ema()  # evaluate and save the averaged weights when --ema_decay is set

        s1 = time.time()
        valid_metrics = run_loader(
            engine,
            dataloader["val_loader"],
            device,
            max_batches=args.max_eval_batches,
            training=False,
        )
        s2 = time.time()
        val_time.append(s2 - s1)

        row = {
            "train_loss": train_metrics["loss"],
            "train_objective": train_metrics["objective"],
            "train_moe_loss": train_metrics["moe_loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "train_wmape": train_metrics["wmape"],
            "valid_loss": valid_metrics["loss"],
            "valid_moe_loss": valid_metrics["moe_loss"],
            "valid_rmse": valid_metrics["rmse"],
            "valid_mape": valid_metrics["mape"],
            "valid_wmape": valid_metrics["wmape"],
            "lr": engine.optimizer.param_groups[0]["lr"],
        }
        if engine.lr_scheduler is not None:
            engine.lr_scheduler.step()
        if args.model_selection_split == "train":
            training_evaluation = run_loader(
                engine, dataloader["train_loader"], device, max_batches=args.max_eval_batches)
            row["train_eval_mae"] = training_evaluation["loss"]
        if args.model_selection_split == "test":
            epoch_tests = evaluate_horizons(engine, dataloader, scaler, device,
                                            args.output_len, args.max_eval_batches, quiet=True)
            row.update(epoch_tests[-1])
            epoch_test_rows.extend(dict(epoch=epoch, horizon=i + 1 if i < args.output_len else "average", **metrics_row)
                                   for i, metrics_row in enumerate(epoch_tests))
            pd.DataFrame(epoch_test_rows).round(8).to_csv(run_dir / "epoch_test.csv", index=False)
            print(f"Epoch: {epoch:03d}, Test MAE: {row['test_loss']:.4f}, Test RMSE: {row['test_rmse']:.4f}, Test MAPE: {row['test_mape']:.4f}")
        train_rows.append(row)
        pd.DataFrame(train_rows).round(8).to_csv(run_dir / "train.csv")

        print(f"Epoch: {epoch:03d}, Training Time: {t2 - t1:.4f} secs")
        print(f"Epoch: {epoch:03d}, Inference Time: {s2 - s1:.4f} secs")
        print(
            "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, "
            "Train MAPE: {:.4f}, Train WMAPE: {:.4f}, MoE: {:.6f}".format(
                epoch,
                row["train_loss"],
                row["train_rmse"],
                row["train_mape"],
                row["train_wmape"],
                row["train_moe_loss"],
            )
        )
        print(
            "Epoch: {:03d}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, "
            "Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}".format(
                epoch,
                row["valid_loss"],
                row["valid_rmse"],
                row["valid_mape"],
                row["valid_wmape"],
            )
        )

        selection_mae = (row["train_eval_mae"] if args.model_selection_split == "train" else
                         row["test_loss"] if args.model_selection_split == "test" else valid_metrics["loss"])
        if selection_mae < best_selection:
            best_selection = selection_mae
            best_valid = valid_metrics["loss"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(engine.model.state_dict(), run_dir / "best_model.pth")
            label = args.model_selection_split
            print(f"Updated best model at epoch {epoch}; {label} MAE={best_selection:.4f}")
            (run_dir / "model_selection.json").write_text(json.dumps(dict(
                split=args.model_selection_split, metric="mae", best_epoch=best_epoch,
                best_mae=best_selection, valid_mae_at_best=best_valid), indent=2) + "\n")
        else:
            epochs_since_best += 1
            print("No update")
        engine.swap_out_ema()

        if epochs_since_best >= args.es_patience:
            print(f"Early stopping after {epochs_since_best} epochs without {args.model_selection_split} improvement.")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Training ends")
    print("The epoch of the best result:", best_epoch)
    print("The valid loss of the best model:", round(best_valid, 4))

    engine.model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    test_rows = evaluate_horizons(
        engine,
        dataloader,
        scaler,
        device,
        args.output_len,
        max_batches=args.max_eval_batches,
    )
    pd.DataFrame(test_rows).round(8).to_csv(run_dir / "test.csv")
    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    start = time.time()
    main()
    print("Total time spent: {:.4f}".format(time.time() - start))
