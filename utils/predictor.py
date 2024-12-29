import torch
import numpy as np
import os
import trimesh
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
        pred, offsets, _, _ = self.teethgroup(fea[None])
        pred = pred[0].softmax(0).cpu().detach().T.numpy() # (n, 17)
        offsets = offsets[0].cpu().detach().T.numpy()   # (n, 3)
        return pred, offsets

    def get_k_nearest(self, pts, pts_ref, k=5):
        """
        :param pts: (n1, 3)
        :param pts_ref: (n2, 3)
        :return: idx: (n1, k)
        """
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k).fit(pts_ref)
        _, idx = nbrs.kneighbors(pts)
        return idx

    def run(self, m:trimesh.Trimesh, sample_num, sample_times=0):
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

        for idx in np.split(sampled_idx, len(sampled_idx)//sample_num):
            pred, offsets = self.infer(all_vs, all_ts[idx])
            all_pred[idx] = pred
            all_offsets[idx] = offsets

        # 2. clustering
        face_centers = m.triangles_center
        probs = self.cluster.cluster(all_pred, all_offsets, face_centers)

        if len(sampled_idx) < num_faces:
            # knn mapping
            unsampled_idx = list(set(range(num_faces)) - set(sampled_idx.tolist()))        # (n1, )
            idx = self.get_k_nearest(face_centers[unsampled_idx], face_centers[sampled_idx], k=5)        # (n1, k)
            probs[unsampled_idx] = probs[sampled_idx][idx].mean(1)       # (n1, 17)

        # 3. Graph cut, optimization for scattered faces

        # 4. Fuzzy clustering

        # 5. Boundary smoothing: smoothing jagged tooth boundaries

        labels = probs.argmax(-1)       # (n,

        return labels

    def infer_with_sampling(self, m:trimesh.Trimesh, sample_num, sample_times=0):
        """
        :param m:
        :param sample_num:
        :param sample_times:  0 indicating sampling all faces.
        :return: probs
                 offsets
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

        for idx in np.split(sampled_idx, len(sampled_idx)//sample_num):
            pred, offsets = self.infer(all_vs, all_ts[idx])
            all_pred[idx] = pred
            all_offsets[idx] = offsets

        face_centers = m.triangles_center
        if len(sampled_idx) < num_faces:
            # knn mapping
            unsampled_idx = list(set(range(num_faces)) - set(sampled_idx.tolist()))        # (n1, )
            idx = self.get_k_nearest(face_centers[unsampled_idx], face_centers[sampled_idx], k=5)        # (n1, k)
            all_pred[unsampled_idx] = all_pred[sampled_idx][idx].mean(1)       # (n1, 17)

        return all_pred, all_offsets
