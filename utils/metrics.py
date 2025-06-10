import torch

def concordance_corr(preds, targets):
    """Compute CCC metric (not as loss, just value)."""
    preds_mean = preds.mean()
    targets_mean = targets.mean()
    preds_var = preds.var(unbiased=False)
    targets_var = targets.var(unbiased=False)
    cov = ((preds - preds_mean) * (targets - targets_mean)).mean()
    ccc = 2 * cov / (preds_var + targets_var + (preds_mean - targets_mean) ** 2)
    return ccc.item()

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def mse(preds, targets):
    return torch.mean((preds - targets)**2).item()