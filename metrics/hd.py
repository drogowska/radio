import glob

import cv2
import numpy as np
import pandas as pd
import scipy

from dataset.datasets import DataSplit, split_subjects
from utils import get_affine_cropped


def hd_all():
    for data_split in [
        DataSplit.FIRST,
        DataSplit.SECOND,
    ]:
        _, test_subjects = split_subjects(data_split)
        split_name = data_split.name

        for stage, label, num_classes in [
            ['heart', 'heart', 1],
            ['chambers', 'chambers', 4],
            ['LAD', 'LAD', 1],
        ]:
            for fcn_atlas in ['fcn', 'atlas']:
                score_hd_list = []
                score_msd_list = []
                score_vr_list = []
                for s in test_subjects:
                    preds = [np.load(p)['arr_0'] for p in
                             glob.glob(f'E:/radio/data/{split_name}/{fcn_atlas}_{stage}/np/preds/{s}_*.npz')]
                    target = [np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/cropped_XY/np/{label}/{s}_*.npz')]
                    affine = get_affine_cropped(s)

                    score_hd_list.append(hd_score(preds, target, affine, num_classes))
                    score_msd_list.append(msd_score(preds, target, affine, num_classes))
                    score_vr_list.append(volume_ratio(preds, target, num_classes))

                df = pd.DataFrame({
                    'subject': test_subjects,
                    'hd': score_hd_list,
                    'msd': score_msd_list,
                    'vr': score_vr_list,
                })
                df.to_csv(f'E:/radio/data/{split_name}/{fcn_atlas}_{stage}/hd.csv', index=False)


def hd_score(preds, target, affine, num_classes):
    scale_x = affine[0, 0]
    scale_y = affine[1, 1]
    scale_z = affine[2, 2]
    scale = np.array([scale_z, scale_y, scale_x])

    s_list = []
    for c in range(num_classes):
        indices_a_list = []
        indices_b_list = []
        for i, (a, b) in enumerate(zip(preds, target)):
            a = contour(a == c + 1)
            b = contour(b == c + 1)

            indices_a = np.concatenate([np.full(shape=[len(a), 1], fill_value=i), a], axis=1)
            indices_b = np.concatenate([np.full(shape=[len(b), 1], fill_value=i), b], axis=1)

            indices_a_list.append(indices_a)
            indices_b_list.append(indices_b)

        indices_a = np.concatenate(indices_a_list)
        indices_b = np.concatenate(indices_b_list)

        indices_a_scaled = indices_a * scale
        indices_b_scaled = indices_b * scale

        s1, _, _ = scipy.spatial.distance.directed_hausdorff(indices_a_scaled, indices_b_scaled)
        s2, _, _ = scipy.spatial.distance.directed_hausdorff(indices_b_scaled, indices_a_scaled)
        s = np.max([s1, s2])
        s_list.append(s)
    return np.mean(s_list)


def msd_score(preds, target, affine, num_classes):
    scale_x = affine[0, 0]
    scale_y = affine[1, 1]
    scale_z = affine[2, 2]
    scale = np.array([scale_z, scale_y, scale_x])

    s_list = []
    for c in range(num_classes):
        indices_a_list = []
        indices_b_list = []
        for i, (a, b) in enumerate(zip(preds, target)):
            a = contour(a == c + 1)
            b = contour(b == c + 1)

            indices_a = np.concatenate([np.full(shape=[len(a), 1], fill_value=i), a], axis=1)
            indices_b = np.concatenate([np.full(shape=[len(b), 1], fill_value=i), b], axis=1)

            indices_a_list.append(indices_a)
            indices_b_list.append(indices_b)

        indices_a = np.concatenate(indices_a_list)
        indices_b = np.concatenate(indices_b_list)

        indices_a_scaled = indices_a * scale
        indices_b_scaled = indices_b * scale

        if len(indices_a_scaled) == 0 or len(indices_b_scaled) == 0:
            print(np.inf)
            s_list.append(np.inf)
            continue
        y = scipy.spatial.distance.cdist(indices_a_scaled, indices_b_scaled, 'euclidean')
        s1 = np.mean(np.min(y, axis=0))
        s2 = np.mean(np.min(y, axis=1))
        print(s1, s2)
        s_list.append(np.mean([s1, s2]))
    return np.mean(s_list)


def volume_ratio(preds, target, num_classes):
    preds = np.asarray(preds)
    target = np.asarray(target)
    res_list = []
    for c in range(num_classes):
        res_list.append(np.sum(preds == c + 1) / np.sum(target == c + 1))
    return np.mean(res_list)


def contour(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    indices, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    indices = np.concatenate(indices, axis=0)[:, 0, :] if len(indices) > 0 else np.empty((0, 2))
    return indices
