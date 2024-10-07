import nibabel
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import dice
import glob
from dataset.datasets import split_subjects, DataSplit
from utils import save_subject_np, load_from_nii, save_subject_nii


def post_greedy(data_split: DataSplit, stage: str, labels: str):
    train_subjects, test_subjects = split_subjects(data_split)
    num_classes = dict(heart=2, chambers=4 + 1, LAD=2)[labels]
    df = []
    for test_i in test_subjects:
        true_labels = load_from_nii(f'data/common/cropped_XY/nii/{labels}/{test_i:02g}.nii.gz')
        preds_list = []
        for train_i in train_subjects:
            preds_labels = load_from_nii(
                f'data/{data_split.name}/atlas_{stage}/nii/preds/{train_i:02g}_{test_i:02g}.nii.gz')
            preds_list.append(preds_labels)

        preds_voted = vote(np.stack(preds_list), num_classes)
        save_subject_np(preds_voted, f'data/{data_split.name}/atlas_{stage}/np/preds', test_i)
        # scores = _dice_multiclass(preds_voted, true_labels, num_classes=num_classes)
        scores = [_dice(preds_voted, true_labels)]
        save_subject_nii(preds_voted.astype(np.int32), f'data/{data_split.name}/atlas_{stage}/nii/res/', test_i)
        df.append([test_i, *scores])
    df = pd.DataFrame(df)
    print(df)
    df.to_csv(f'data/{data_split.name}/atlas_{stage}/scores.csv', index=False)


def vote(arr, num_classes):
    arr = np.asarray(arr)
    counts = [np.sum(arr == i, axis=0)
              for i in range(num_classes)]
    counts = np.asarray(counts)
    preds = np.argmax(counts, axis=0)
    return preds


def vote2(arr, _num_classes):
    arr = np.asarray(arr)
    counts = np.sum(arr, axis=0)
    return np.asarray(counts >= 2, dtype=int)


def _dice(preds, target):
    return dice(
        torch.tensor(preds),
        torch.tensor(target.copy()),
        ignore_index=0,
    ).numpy()


def _dice_multiclass(preds, target, num_classes):
    return dice(
        torch.tensor(preds),
        torch.tensor(target.copy()),
        ignore_index=0,
        average=None,
        num_classes=num_classes,
    ).numpy()[1:]



def test():

    true_labels = load_from_nii('data/common/cropped_XY/nii/chambers/29.nii.gz')
    test_labels = load_from_nii('C:/Users/Pc/Documents/All_in_one/mgr/greedy_test/atl_pred_00_29.nii.gz')
    preds_copy = np.array(test_labels, copy=True)  # Make a copy of the array
    preds_voted = vote(np.stack(preds_copy), 5)
    # save_subject_np(preds_voted, f'data/{data_split.name}/atlas_{stage}/np/preds', test_i)
    # scores = _dice_multiclass(preds_voted, true_labels, num_classes=num_classes)
    scores = [_dice(preds_voted, true_labels)]
    print(scores)
    # df.append([test_i, *scores])
    # save_subject_nii(preds_voted, f'data/common/tmp/{test_i}', test_i)
    dice_score = _dice(preds_copy, true_labels)
    print(dice_score)
