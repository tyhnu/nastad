import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class BalancedCELoss(object):
    """Used in VSGN"""

    def __init__(self) -> None:
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def __call__(self, cls_pred, cls_labels):
        pmask = (cls_labels > 0).float()
        nmask = (cls_labels == 0).float()
        num_pos = torch.sum(pmask)
        num_neg = torch.sum(nmask)

        loss = self.loss(cls_pred, cls_labels.to(torch.long))

        pos_loss = torch.sum(loss * pmask) / num_pos
        neg_loss = torch.sum(loss * nmask) / num_neg

        total_loss = pos_loss + neg_loss
        return total_loss



@LOSSES.register_module()
class SpacialMILoss(object):
    """Used in VSGN"""

    def __init__(self) -> None:
        self.pos_thresh = 0.5
    def __call__(self, spacial_variational, spacial_logvar, cls_labels):
        inputs = spacial_variational.float()
        targets = cls_labels.float()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        mi = 1. / 2. * (spacial_variational ** 2 + spacial_logvar.exp() - 1. - spacial_logvar).mean()
        return  mi, ce_loss.mean()

@LOSSES.register_module()
class InputMILoss(object):
    """Used in VSGN"""

    def __init__(self):
        super().__init__()

    def __call__(self, input_variational, input_logvar, feature):
        channels=input_variational.shape[1]
        # print(spacial_variational.shape)
        positive = - (input_variational - feature)**2 /2./input_logvar.exp()

        prediction_1 = input_variational.unsqueeze(1)          # shape [nsample,1,dim]
        targets_1 = feature.unsqueeze(0)

        negative = - ((targets_1 - prediction_1)**2).mean(dim=1)/2./input_logvar.exp()


        mi = positive.sum(dim=-1).mean() - negative.sum(dim=-1).mean()
        loglikelihood= - (input_variational - feature)**2 /input_logvar.exp()-input_logvar
        loss = -loglikelihood.sum(dim=1).mean()/channels
        # print(loss)
        return  mi, loss
