import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .matcher import HungarianMatcher


class masks_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.output_channels
        self.matcher = HungarianMatcher(cost_masks=1, cost_probs=10000)
        self.loss_mask = FocalLoss(gamma=2, alpha=None, reduction='mean') # nn.BCELoss()  # FocalLoss(gamma=2, alpha=None, reduction='mean')
        self.loss_prob = nn.CrossEntropyLoss()

    def forward(self, masks_gt, masks_labels, masks_pred, masks_probs):
        indices = self.matcher(masks_gt, masks_labels, masks_pred, masks_probs)

        loss = 0.0
        lm = 0.0
        lp = 0.0
        for i, idx in enumerate(indices):
            gm = masks_gt[i, idx[1], :]
            pm = masks_pred[i, idx[0], :]
            loss_m = self.loss_mask(pm, gm)

            prob = masks_probs[i]
            gtc = masks_labels[i, idx[1]]
            # back as zero
            gclass = torch.zeros(prob.shape[1], dtype=torch.long).to(prob.device)
            gclass[idx[0]] = gtc

            # cross entropy loss
            one_hot = F.one_hot(gclass, num_classes=17).float()
            loss_p = self.loss_prob(prob.T, one_hot)

            loss = loss + loss_m * 4 + loss_p
            lm += loss_m
            lp += loss_p

        return loss, lm, lp


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # input: (B, C, N), target: (B, N)
        ce_loss = F.binary_cross_entropy(input, target, reduction='none')

        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t)**self.gamma * ce_loss

        if self.alpha is not None:
            alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

