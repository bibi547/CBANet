import numpy
import torch
import numpy as np
import os
import trimesh
from scipy.spatial.distance import cdist
from pl_model import LitModel
from data.common import calc_features
from utils.cluster import Cluster


class Predictor:
    def __init__(self, weight_dir):
        weights = os.path.join(weight_dir, 'checkpoints/last.ckpt')
        self.teethgroup = LitModel.load_from_checkpoint(weights).cuda()
        self.args = self.teethgroup.hparams.args
        self.teethgroup.eval()
        self.cluster = Cluster()

    @torch.no_grad()
    def infer(self, vs, ts):
        fea = calc_features(vs, ts)     # (15, n)
        pred, offsets, masks, sp_probs, sp_xyz, score = self.teethgroup.infer(fea[None])

        pred = pred[0].softmax(0).cpu().detach().T.numpy()  # (n, 17)
        offsets = offsets[0].cpu().detach().T.numpy()  # (n, 3)

        sp_probs = sp_probs[0].softmax(0).cpu().detach().T.numpy()  # (m, 17)
        sp_xyz = sp_xyz[0].cpu().detach().T.numpy()  # (m, 3)
        masks = masks[0].cpu().detach().numpy()  # (m, n)
        masks = np.where(masks > 0.5, 1, 0)  # (m, n)

        return pred, offsets, masks, sp_probs, sp_xyz

    def run(self, m: trimesh.Trimesh, sample_num, ins_masks, ins_labels, ins_xyz, sample_times=0):
        """
        :param m:
        :param sample_num:
        :param sample_times:  0 indicating sampling all faces.
        :return:
        """
        num_faces = len(m.faces)
        if sample_times == 0:
            sample_times = 1000
        sample_times = min(sample_times, (num_faces+sample_num-1)//sample_num)

        perm_idx = np.random.permutation(num_faces)
        if sample_times * sample_num > num_faces:
            sampled_idx = np.concatenate([perm_idx, perm_idx[:sample_num - len(perm_idx) % sample_num]])
        else:
            sampled_idx = perm_idx[:sample_num * sample_times]

        all_vs = torch.tensor(m.vertices, dtype=torch.float).cuda()
        all_ts = torch.tensor(m.faces, dtype=torch.long).cuda()
        all_vs = all_vs - all_vs.mean(0)  # preprocess

        # 1. network inference
        all_pred = np.zeros((num_faces, 17))
        all_offsets = np.zeros((num_faces, 3))
        all_masks = np.zeros((50, num_faces))  # 50 n
        all_mask_labels = np.zeros((num_faces, 1))  # n 17

        # for i, idx in enumerate(np.split(sampled_idx, len(sampled_idx)//sample_num)):
        #     pred, offsets, masks, sp_probs, sp_xyz = self.infer(all_vs, all_ts[idx])
        #     all_pred[idx] = pred
        #     all_offsets[idx] = offsets
        #     all_masks[:, idx] = masks
        pred, offsets, masks, sp_probs, sp_xyz = self.infer(all_vs, all_ts)

        all_pred = all_pred.argmax(-1)
        # all_masks = np.concatenate(all_masks, axis=0)

        for mask in masks:
            visual(m, mask)

        cs = m.triangles_center# - np.array(all_vs.mean(0).cpu())
        labels = self.grouping(cs, all_pred, all_offsets, all_graph_labels, all_graph_xyz, ins_labels, ins_xyz)

        # 2. clustering
        # cs = m.triangles_center
        # labels = self.grouping(all_pred, cs, all_offsets, all_masks, all_masks_labels, gins_label, gins_xyz)

        # 3. Graph cut, optimization for scattered faces

        # 4. Fuzzy clustering

        # 5. Boundary smoothing: smoothing jagged tooth boundaries

        # labels = probs.argmax(-1)       # (n,

        return labels

    def grouping(self, cs, preds, offsets, graph_labels, graph_xyz, ins_labels, ins_xyz):
        shift_xyz = cs + offsets
        pt_idx = np.where(np.sum(offsets**2, 1)**0.5 > 0.01)[0]
        clustering = DBSCAN(eps=1.05, min_samples=30).fit(shift_xyz[pt_idx])
        clusters = [pt_idx[clustering.labels_ == i] for i in range(np.max(clustering.labels_) + 1)]

        clu_centers = []
        for i, indices in enumerate(clusters):
            clu_coords = shift_xyz[indices]
            clu_center = np.mean(clu_coords, axis=0)
            clu_centers.append(clu_center)
        clu_centers = np.array(clu_centers)

        # 计算距离矩阵
        dists = cdist(clu_centers[:,:2], graph_xyz[:,:2])
        # 找到每个查询点的最近邻点的索引
        idx = np.argmin(dists, axis=1)
        clu_labels = graph_labels[idx]

        num_faces = cs.shape[0]
        labels = np.zeros(num_faces)
        for i, indices in enumerate(clusters):
            labels[indices] = clu_labels[i]

        return labels


def visual(m: trimesh.Trimesh, labels):
    from mesh import TriMesh
    TriMesh(m.vertices, m.faces, labels).visualize()

