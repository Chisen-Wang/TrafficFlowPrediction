import torch


def MAE_torch(pred: torch.Tensor, true: torch.Tensor, mask_value=None) -> torch.Tensor:
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred: torch.Tensor, true: torch.Tensor, mask_value=None) -> torch.Tensor:
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def RMSE_torch(pred: torch.Tensor, true: torch.Tensor, mask_value=None) -> torch.Tensor:
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred: torch.Tensor, true: torch.Tensor, mask_value=None) -> torch.Tensor:
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))


def metric(pred: torch.Tensor, real: torch.Tensor):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape


def count_statistics(pred, true):
    """Sufficient statistics for real count targets, including valid zero demand."""
    error = (pred.detach().double() - true.detach().double()).abs()
    positive = true > 0
    return {
        "_count": true.numel(), "_sae": error.sum().item(), "_sse": error.square().sum().item(),
        "_target_sum": true.double().sum().item(), "_positive_count": positive.sum().item(),
        "_ape": (error[positive] / true.double()[positive]).sum().item(),
    }


def count_metrics(stats):
    return {
        "loss": stats["_sae"] / stats["_count"],
        "rmse": (stats["_sse"] / stats["_count"]) ** .5,
        # Zero-denominator batches have no percentage information; sums retain this fact.
        "mape": stats["_ape"] / stats["_positive_count"] if stats["_positive_count"] else 0.,
        "wmape": stats["_sae"] / stats["_target_sum"] if stats["_target_sum"] else 0.,
    }
