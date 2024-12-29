import torch


def calc_features(vs, ts):
    """
    :param vs: (nv, 3), float
    :param ts: (nf, 3), long
    :return: fea: (15, nf), float
    """
    nf = ts.shape[0]
    fea = torch.empty((15, nf), dtype=torch.float32).to(vs.device)

    vs_in_ts = vs[ts]

    fea[:3, :] = vs_in_ts.mean(1).T[None]  # centers , 3
    fea[3:6, :] = calc_normals(vs, ts).T[None]  # normal, 3
    fea[6:15, :] = (vs_in_ts - vs_in_ts.mean(1, keepdim=True)).reshape((nf, -1)).T[None]
    return fea


def calc_normals(vs: torch.Tensor, ts: torch.Tensor):
    """
    :param vs: (n_v, 3)
    :param ts: (n_f, 3), long
    :return normals: (n_f, 3), float
    """
    normals = torch.cross(vs[ts[:, 1]] - vs[ts[:, 0]],
                          vs[ts[:, 2]] - vs[ts[:, 0]])
    normals /= torch.sum(normals ** 2, 1, keepdims=True) ** 0.5 + 1e-9
    return normals
