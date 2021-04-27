import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 classes,
                 alpha=None,
                 gamma=2,
                 size_average=True,
                 device=None):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(classes, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.classes = classes
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * mask).sum(1).view(-1, 1)
        logp = probs.log()
        batchloss = -alpha * (torch.pow(1 - probs, self.gamma)) * logp
        if self.size_average:
            loss = batchloss.mean()
        else:
            loss = batchloss.sum()
        return loss
