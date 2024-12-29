import numpy as np
from collections import defaultdict
from sklearn import metrics


def get_contour_points(faces, labels):
    contour_points = []
    for i,adj_faces in enumerate(faces):
        adj_faces = np.setdiff1d(adj_faces, -1)
        adj_labels = labels[adj_faces]
        if len(set(adj_labels)) > 1:
            contour_points.append(i)
    return contour_points


def get_iou(ori_idx, tar_idx):

    inter = np.intersect1d(ori_idx, tar_idx)
    union = np.union1d(ori_idx, tar_idx)
    iou = len(inter)/len(union)

    return iou


def get_metrics(preds, gt, pred_contour, label_contour):
    acc = metrics.accuracy_score(list(preds.squeeze()), list(gt.squeeze()))
    contour_iou = get_iou(label_contour, pred_contour)
    tooth_iou = get_tooth_iou(preds, gt)
    return acc, tooth_iou, contour_iou


def get_faces_per_classes(labels, tar_labels):
    faces_dict = defaultdict(list)
    faces_dict = faces_dict.fromkeys(tar_labels, [])
    for i, l in enumerate(labels):
        if l in tar_labels:
            faces = faces_dict[l][:]
            faces.append(i)
            faces_dict[l] = faces
    return faces_dict


def get_tooth_iou(preds, labels):
    miou = 0.0
    tooth_num = 0.0
    tooth_idx = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    tooth_preds = get_faces_per_classes(preds, np.array(tooth_idx))
    tooth_labels = get_faces_per_classes(labels, np.array(tooth_idx))
    for l in tooth_labels:
        if len(tooth_labels[l]) == 0:
            continue
        labels = tooth_labels[l]
        if len(tooth_preds[l]) != 0:
            preds = tooth_preds[l]
            iou = get_iou(preds, labels)
        else:
            iou = 0
        miou += iou
        tooth_num += 1
    miou = miou/tooth_num
    return miou


def get_vertex_faces(faces):
    # 获取每个顶点的相邻面
    vertex_faces = {}
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            if vertex_idx not in vertex_faces:
                vertex_faces[vertex_idx] = []
            # 将面的索引添加到相邻面列表
            vertex_faces[vertex_idx].append(face_idx)
    return vertex_faces



