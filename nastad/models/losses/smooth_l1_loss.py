import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "sum",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(inputs, targets, reduction="none")

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class TemporalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        positive = - (inputs - targets)**2 /2./logvar.exp()

        prediction_1 = inputs.unsqueeze(1)
        targets_1 = targets.unsqueeze(0)

        negative = - ((targets_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp()

        mi = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        loglikelihood= - (inputs - targets)**2 /logvar.exp()-logvar
        loss = - loglikelihood.sum(dim=1).mean()
        return  loss, mi


# @LOSSES.register_module()
# class TemporalLoss_old(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(
#         self,
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         logvar: torch.Tensor,
#     ) -> torch.Tensor:
#
#         # positive = - (inputs - targets)**2 /2./logvar.exp()
#         #
#         # prediction_1 = inputs.unsqueeze(1)          # shape [nsample,1,dim]
#         # targets_1 = targets.unsqueeze(0)
#         #
#         # negative = - ((targets_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp()
#
#         # mi = 1./2.*(inputs**2 + logvar.exp() - 1. - logvar).mean()
#         sample_size = inputs.shape[0]
#         random_index = torch.randperm(sample_size)
#         positive = - (inputs - targets)**2 / logvar.exp()
#         negative = - (inputs - targets[random_index])**2 / logvar.exp()
#         mi = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
#         loglikelihood= - (inputs - targets)**2 /logvar.exp()-logvar
#         loss = -loglikelihood.sum(dim=1).mean()
#         return  loss, mi
