import torch
from .dgcnn_utils import knn


def center_fps(dmap, xyz, npoint):
    device = xyz.device
    N = xyz.shape[0]
    cen_idx = []
    mask_dmap = dmap
    knn_idx = knn(xyz.T.unsqueeze(0), 20).squeeze()
    for i in range(int(npoint/2)):
        idx = torch.argmax(mask_dmap)
        d_max = mask_dmap[idx]
        xyz_max = xyz[idx].unsqueeze(0)
        if d_max < 0.5:
            break
        n_idx = knn_idx[idx]
        mask_dmap[n_idx] = -1.
        mask_dmap[idx] = -1.
        cen_idx.append(idx.unsqueeze(0))
    cur_len = len(cen_idx)
    cen_idx = torch.cat(cen_idx, dim=0)

    distance = torch.sum((xyz.unsqueeze(1) - xyz[cen_idx].unsqueeze(0)) ** 2, dim=2)
    distance = torch.min(distance, dim=-1)[0]
    far_idx = torch.argmax(distance)
    fps_idx = []
    for i in range(npoint-cur_len):
        fps_idx.append(far_idx.unsqueeze(0))
        centroid = xyz[far_idx]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        far_idx = torch.max(distance, -1)[1]
    fps_idx = torch.cat(fps_idx, dim=0)

    sampl_idx = torch.cat([cen_idx, fps_idx])
    sampl_xyz = xyz[sampl_idx]
    return sampl_xyz, sampl_idx





