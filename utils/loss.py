import torch
from torch import nn
import torch.nn.functional as F
from .matcher import HungarianMatcher


class CPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.output_channels
        self.loss_prob = nn.CrossEntropyLoss()
        self.loss_mask = nn.BCELoss()
        self.loss_score = nn.MSELoss()

    def forward(self, gt_masks, gt_labels, pred_masks, pred_labels, score, all_idx):
        bs = gt_masks.shape[0]
        loss = 0.0
        loss_class = 0.0
        loss_mask = 0.0
        loss_score = 0.0
        for i in range(bs):
            idx = torch.nonzero(gt_labels[i] != -1)[:, 0]
            g_masks = gt_masks[i, idx]
            g_labels = gt_labels[i, idx]
            labels = torch.sum(g_masks * g_labels.unsqueeze(1), dim=0)
            sampl_idx = all_idx[i]
            sampl_labels = labels[sampl_idx]
            all_labels = torch.cat([torch.zeros(1).to(g_labels.device), g_labels], dim=0)
            all_masks = torch.cat([torch.zeros(1, g_masks.shape[1]).to(g_masks.device), g_masks], dim=0)
            masks_idx = torch.tensor([torch.where(all_labels == item)[0][0] for item in sampl_labels])
            sampl_masks = all_masks[masks_idx]

            p_masks = pred_masks[i]
            p_labels = pred_labels[i]
            p_score = score[i]

            one_hot = F.one_hot(sampl_labels.to(pred_labels.device, dtype=torch.long), num_classes=17).float()
            loss_c = self.loss_prob(p_labels.unsqueeze(0), one_hot.T.unsqueeze(0))
            loss_bce = self.loss_mask(p_masks.unsqueeze(0), sampl_masks.unsqueeze(0))
            loss_dice = dice_loss(p_masks.unsqueeze(0), sampl_masks.unsqueeze(0))
            loss_m = loss_bce + loss_dice

            with torch.no_grad():
                tgt_score = get_iou(p_masks, sampl_masks)
            filter_id = torch.where(tgt_score > 0.5)[0]
            loss_s = 0.0
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                p_score = p_score.squeeze()[filter_id]
                loss_s = self.loss_score(p_score, tgt_score)

            loss = loss + loss_c + loss_m * 10 + loss_s * 10  # 1 10 10  # c 2.9 m 0.71 0.92 1.64 s
            loss_class += loss_c
            loss_mask += loss_m
            loss_score += loss_s

        return loss / bs, loss_class / bs, loss_mask / bs, loss_score / bs


def contra_loss_query(labels, probs, masks):
    query_num = labels.size(0)
    pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))  # [n, n]
    dists = torch.sum((probs.T.unsqueeze(1) - probs.T.unsqueeze(0)) ** 2, dim=2)
    # pos = torch.exp(-dists*pos_mask)
    # all = torch.exp(-dists)
    loss = 0.0
    for i in range(query_num):
        pos = torch.sum(torch.exp(-dists[i][pos_mask[i]]))
        all = torch.sum(torch.exp(-dists[i]))
        l = -torch.log(pos/all)
        loss += l
    return loss# / query_num


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


class match_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.output_channels
        self.loss_prob = nn.CrossEntropyLoss()
        self.loss_mask = nn.BCELoss()
        self.loss_score = nn.MSELoss()
        self.matcher = HungarianMatcher(cost_masks=1, cost_probs=1)

    def forward(self, gt_masks, gt_labels, pred_masks, pred_labels, score):
        indices = self.matcher(gt_masks, gt_labels, pred_masks, pred_labels)

        bs = gt_masks.shape[0]
        loss = 0.0
        loss_class = 0.0
        loss_mask = 0.0
        loss_score = 0.0
        for i in range(bs):
            indice_q, indice_g = indices[i]
            idx = torch.nonzero(gt_labels[i] != -1)[:, 0]
            g_masks = gt_masks[i, idx]
            g_labels = gt_labels[i, idx]
            g_masks = g_masks[indice_g]
            g_labels = g_labels[indice_g]

            p_masks = pred_masks[i, indice_q]
            p_labels = pred_labels[i, :, indice_q]
            one_hot = F.one_hot(g_labels.to(dtype=torch.long), num_classes=17).float()

            loss_c = self.loss_prob(p_labels.unsqueeze(0), one_hot.T.unsqueeze(0))
            loss_bce = self.loss_mask(p_masks.unsqueeze(0), g_masks.unsqueeze(0))
            loss_dice = dice_loss(p_masks.unsqueeze(0), g_masks.unsqueeze(0))

            p_score = score[i, :, indice_q]
            with torch.no_grad():
                tgt_score = get_iou(p_masks, g_masks).unsqueeze(1)
            loss_s = self.loss_score(p_score.T, tgt_score)

            loss_m = loss_bce + loss_dice
            loss = loss + loss_c + loss_m * 4 + loss_s * 10
            loss_class += loss_c
            loss_mask += loss_m
            loss_score += loss_s

        return loss/bs, loss_class/bs, loss_mask/bs, loss_score/bs


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    # inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def iou_gpu(output, target, K, ignore_index=-100):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


