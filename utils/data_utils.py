import numpy as np
import trimesh
from pathlib import Path
import json


def augment(vs, ts):
    # jitter vertices
    if np.random.rand(1) > 0.5:
        sigma, clip = 0.01, 0.05
        jitter = np.clip(sigma * np.random.randn(len(vs), 3), -1 * clip, clip)
        vs += jitter

    # translate
    if np.random.rand(1) > 0.5:
        scale = np.random.uniform(low=0.9, high=1.1, size=[3])
        translate = np.random.uniform(low=-0.2, high=0.2, size=[3])
        vs = np.add(np.multiply(vs, scale), translate)

    # change the order of vertices
    ts = np.roll(ts, np.random.randint(0, 3, 1), axis=1)

    # rotation
    if np.random.rand(1) > 0.5:
        axis_xyz = np.roll(np.eye(3), np.random.randint(0, 3, 1), axis=0)
        angles = np.random.uniform(low=-5/180*np.pi, high=5/180*np.pi, size=[3])
        matrix = trimesh.transformations.concatenate_matrices(*[trimesh.transformations.rotation_matrix(angle, axis) for axis, angle in zip(axis_xyz, angles)])
        vs = trimesh.transformations.transform_points(vs, matrix)

    return vs, ts


def load_predictions_json(fname: Path):

    cases = {}

    with open(fname, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {fname}")

    for e in entries:
        # Find case name through input file name
        inputs = e["inputs"]
        name = None
        for input in inputs:
            if input["interface"]["slug"] == "3d-teeth-scan":
                name = input["file"].split('/')[-1].split('.')[0] #str(input["image"]["name"])
                break  # expecting only a single input
        if name is None:
            raise ValueError(f"No filename found for entry: {e}")

        entry = {"name": name}

        # Find output value for this case
        outputs = e["outputs"]

        for output in outputs:
            if output["interface"]["slug"] == "dental-labels":
                # cases[name] = output['value']
                # cases[name] = e["pk"]
                cases[name] = output["file"]
    return cases


def get_offsets(points, labels):
    offsets = np.zeros((points.shape[0], 3), dtype='f4')
    for i in range(1, 17):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        center = points[idx].mean(0)
        offset = center - points[idx, :3]
        offsets[idx] = offset
    return offsets


def get_centroids(points, labels):
    centroids_label = []
    centroids_xyz = []
    for i in range(1, 17):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        center = points[idx].mean(0)
        centroids_label.append(i)
        centroids_xyz.append(center)
    return np.array(centroids_label), np.array(centroids_xyz)


def get_masks(points, labels):
    masks = - np.ones((20, points.shape[0]))
    mask_labels = - np.ones(20)
    centroids_xyz = - np.ones((20, 3))
    for i in range(1, 17):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        masks[i] = 0
        masks[i, idx] = 1
        center = points[idx].mean(0)
        mask_labels[i] = i
        centroids_xyz[i] = center
    return np.array(masks), np.array(mask_labels), np.array(centroids_xyz)


def get_bmap(vertex_faces, faces, labels):
    bmap = np.zeros(faces.shape[0])
    t_idx = np.argwhere(labels != 0)
    bmap[t_idx] = 2
    for i,adj_faces in enumerate(vertex_faces):
        adj_faces = np.setdiff1d(adj_faces, -1)
        adj_labels = labels[adj_faces]
        if len(set(adj_labels)) > 1:
            bmap[adj_faces] = 1
    return bmap

def test():
    mapping_dict = load_predictions_json(Path("F:/dataset/Teeth3DS/data/upper/00OMSZGW/00OMSZGW_upper.json"))
    return mapping_dict


if __name__ == "__main__":

    mapping_dict = test()
    print()

