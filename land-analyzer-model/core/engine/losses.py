import torch
from torch import nn


class LogLoss(nn.Module):
    def __init__(self, reduction=False):
        super(LogLoss, self).__init__()
        self.reduce = reduction

    def forward(self, inputs, targets, masks): # BCHW
        # Clamp data
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)

        # Calculate loss
        ALPHA, BETA = 1.0, 1.0
        fn_loss = - (ALPHA * targets * torch.log(inputs))
        fp_loss = - (BETA * (1.0 - targets) * torch.log(1.0 - inputs))
        losses = fn_loss + fp_loss

        # Apply mask
        # losses = losses * masks TODO: implement and test

        if self.reduce:
            return torch.mean(losses)
        else:
            return losses


class JaccardIndex(nn.Module):
    def __init__(self, reduction=False, eps=1e-6):
        super(JaccardIndex, self).__init__()
        self.reduce = reduction
        self.eps = eps

    def forward(self, inputs, targets): # BCHW
        intersection = torch.sum(torch.multiply(inputs, targets), dim=[2, 3])
        union = torch.sum(inputs, dim=[2, 3]) + torch.sum(targets, dim=[2, 3]) - intersection
        metric = torch.add(intersection, self.eps) / torch.add(union, self.eps)

        if self.reduce:
            return torch.mean(metric)
        else:
            return metric


class DiceLoss(nn.Module):
    def __init__(self, reduction=False, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.reduce = reduction
        self.eps = eps

    def forward(self, inputs, targets): # BCHW
        intersection = torch.sum(torch.multiply(inputs, targets), dim=[2, 3])
        metric = (torch.multiply(intersection, 2.0) + self.eps) / (torch.sum(inputs, dim=[2, 3]) + torch.sum(targets, dim=[2, 3]) + self.eps)

        if self.reduce:
            return torch.mean(metric)
        else:
            return metric