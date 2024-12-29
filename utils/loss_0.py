import torch
import torch.nn.functional as F


def instance_loss(gt_labels, gt_xyz, pred_prob, pred_xyz):
    loss = 0.0
    b = gt_labels.size(0)
    for i in range(b):
        prob = pred_prob[i]
        p_xyz = pred_xyz[i]
        g_xyz = gt_xyz[i]
        g_labels = gt_labels[i]
        idx = find_nearest_indices(g_xyz, p_xyz)
        # dists = torch.norm(p_xyz - g_xyz, dim=-1)
        # idx = torch.argmin(dists, dim=1)
        g_labels = g_labels[idx]
        l = F.cross_entropy(prob.unsqueeze(0), g_labels.unsqueeze(0))
        loss += l
    return loss


def find_nearest_indices(gt_xyz, pred_xyz):
    # 计算 L2 范数（欧氏距离）的平方，得到距离的平方矩阵
    distances = torch.sum((pred_xyz.unsqueeze(1) - gt_xyz.unsqueeze(0))**2, dim=2)

    # 找到每行最小值的索引，即找到每个 pred_xyz 最近的 gt_xyz 的索引
    nearest_indices = torch.argmin(distances, dim=1)

    return nearest_indices
