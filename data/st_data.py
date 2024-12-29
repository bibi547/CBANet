import os
import numpy as np
import random
import json

import torch
import trimesh
from torch.utils.data import Dataset

from data.common import calc_features
from utils.data_utils import augment, get_offsets, get_centroids, get_masks, get_bmap


class Teeth3DS(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        # args
        self.args = args
        self.root_dir = args.root_dir
        self.num_points = args.num_points
        self.augmentation = args.augmentation if train else False
        # files
        self.files = []
        with open(os.path.join(args.split_dir, split_file)) as f:
            for line in f:
                filename = line.strip().split('_')[0]
                category = line.strip().split('_')[1]
                root = os.path.join(self.root_dir, category, filename)
                obj_file = os.path.join(root, f'{line.strip()}_sim.off')
                json_file = os.path.join(root, f'{line.strip()}_sim_re.txt')
                dmap_file = os.path.join(self.root_dir, category + '_fdmap', f'{line.strip()}_sim.txt')
                bmap_file = os.path.join(self.root_dir, category + '_fbmap', f'{line.strip()}_sim.txt')
                if os.path.exists(obj_file) and os.path.exists(json_file):
                    self.files.append((obj_file, json_file, dmap_file, bmap_file))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj_file, json_file, dmap_file, bmap_file = self.files[idx]

        mesh = trimesh.load(obj_file)
        vs, fs = mesh.vertices, mesh.faces
        labels = np.loadtxt(json_file, dtype=np.int32)
        dmap = np.loadtxt(dmap_file, dtype=np.float32)  # [-1 1]
        bmap = np.loadtxt(bmap_file, dtype=np.float32)  # boundary 1 other 0
        # 边界点为1 牙齿点为2 牙龈点为0
        b_idx = np.argwhere(bmap == 1)[:, 0]
        bmap = np.ones(bmap.shape, dtype='int64')
        gum_idx = np.argwhere(labels == 0)[:, 0]
        bmap[gum_idx] = -1
        bmap[b_idx] = 0
        bmap = bmap + 1

        # augmentation
        if self.augmentation:
            vs, fs = augment(vs, fs)
        # sample
        _, fids = trimesh.sample.sample_surface_even(mesh, self.num_points)

        fs, labels = fs[fids], labels[fids]
        bmap, dmap = bmap[fids], dmap[np.newaxis, fids]
        # extract input features
        vs = torch.tensor(vs, dtype=torch.float32)
        vs = vs - vs.mean(0)  # preprocess
        fs = torch.tensor(fs, dtype=torch.long)
        features = calc_features(vs, fs)  # (15, nf)
        labels = np.array(labels, dtype='float64').squeeze()

        cs = np.array(features.T[:, :3])
        ins_masks, ins_labels, ins_xyz = get_masks(cs, labels)  # label:[] xyz:[]

        return features, torch.tensor(labels, dtype=torch.long), \
            torch.tensor(bmap, dtype=torch.long), torch.tensor(dmap, dtype=torch.float32), \
            torch.tensor(ins_masks, dtype=torch.float32), torch.tensor(ins_labels, dtype=torch.float32)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.root_dir = 'F:/dataset/Teeth3DS/data'
            self.split_dir = 'F:/dataset/Teeth3DS/split'
            self.num_points = 10000
            self.augmentation = True


    data = Teeth3DS(Args(), 'training_upper.txt', True)
    i = 0
    for f, l, b, d, ins_m, ins_l in data:
        print(f.shape)
        print(b.shape)
        print(d.shape)
        i += 1

    print(i)
