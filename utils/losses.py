# Concordance Correlation Coefficient (CCC)
# CCC is a specialized loss often used in affect regression tasks; it measures how well predictions correlate with ground truth, taking into account both precision and accuracy. The CCC between predictions p and ground truth y
import torch
import torch.nn as nn

class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient Loss. Returns 1-CCC (to minimize)."""
    def forward(self, preds, targets):
        # preds and targets are 1D tensors (batch,)
        preds_mean = preds.mean()
        targets_mean = targets.mean()
        preds_var = preds.var(unbiased=False)  # population variance
        targets_var = targets.var(unbiased=False)
        # For covariance, use dot product formula:
        cov = ((preds - preds_mean) * (targets - targets_mean)).mean()
        # Compute CCC numerator and denominator
        denom = preds_var + targets_var + (preds_mean - targets_mean) ** 2
        if denom.item() == 0:
            return torch.tensor(0.0)  # if variance is zero (unlikely in practice for PHQ)
        ccc = 2 * cov / denom
        # loss is 1 - ccc (higher ccc => lower loss)
        return 1 - ccc