import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F


class HungarianMatcher(nn.Module):
    def __init__(self, cost_masks: float = 1, cost_probs: float = 1):
        super().__init__()
        self.cost_masks = cost_masks
        self.cost_probs = cost_probs
        assert cost_masks != 0 or cost_probs != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, gt_mask, labels, pred_masks, probs):
        bs, num_query = probs.shape[:2]

        indices = []
        for i in range(bs):
            idx = torch.nonzero(labels[i] != -1)[:, 0]
            gm = gt_mask[i, idx]
            gc = labels[i, idx]
            pm = pred_masks[i]
            pc = probs[i].T

            cost_bce = batch_bce_loss(pm, gm)
            cost_dice = batch_dice_loss(pm, gm)
            cost_masks = cost_dice + cost_bce
            cost_probs = 1 - pc[:, gc.to(dtype=torch.long)]

            C = self.cost_masks * cost_masks + self.cost_probs * cost_probs
            indice = linear_sum_assignment(C.cpu())
            indices.append(indice)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@torch.jit.script
def batch_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy(inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss

