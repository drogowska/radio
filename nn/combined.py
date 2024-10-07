from glob import glob
import torch
from nn.UnetModel import UnetModel
from dataset.segmentationDataModule import SegmentationDataModule
from nn.utils import make_trainer
from utils import save_subject_np, save_subject_nii, save_nifti_file, load_nii, prepare_mask_for_metrics
import numpy as np
import os
from atlas.post_greedy import _dice, vote2
import pandas as pd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
)
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric, compute_dice, compute_hausdorff_distance, compute_average_surface_distance, ConfusionMatrixMetric
from metrics.np import compute_volume_ratio


def run_cnn_on_atlas(model: UnetModel, dm: SegmentationDataModule):
    dm.setup()
    trainer = make_trainer(dm.data_split, type(model))
    model.load_state_dict(torch.load(f'E:/radio/data/models/{dm.data_split.name}_{model.stage}.pth'))

    dl = dm.test_dataloader()
    preds = trainer.predict(model, dataloaders=dl)
    os.makedirs(f'E:/radio/data/{dm.data_split.name}/{model.stage}_combined/nii/img', exist_ok=True)
    os.makedirs(f'E:/radio/data/{dm.data_split.name}/{model.stage}_combined/nii/preds', exist_ok=True)
    for pred, img, true, name in preds:
        n = name[0]
        save_nifti_file(pred.squeeze().cpu().numpy().astype(np.int32), np.eye(4),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}_combined/nii/preds/{n}')
        save_nifti_file(img.squeeze().cpu().numpy(), np.eye(4),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}_combined/nii/img/{n}')


def resize(path):
    transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(256, 256, 256)),
            EnsureTyped(keys=["image"])
        ])
    images = sorted(glob(os.path.join(path, "*.nii.gz")))
    names = [os.path.basename(path) for path in images]
    data_dicts = [{"image": img, "name": name[0:5]} for img, name in zip(images, names)]
    dataset = Dataset(data=data_dicts, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    for batch in dataloader:
        transformed_image = batch["image"][0]
        save_nifti_file(transformed_image.squeeze().numpy(), np.eye(4), f'{path}/{batch["name"][0]}')




def combined(model: UnetModel, dm: SegmentationDataModule, atlas_stage: str):
    run_cnn_on_atlas(model, dm)
    resize( f'E:/radio/data/FIRST/atlas_{atlas_stage}/nii/preds/')
    os.makedirs(f'E:/radio/data/FIRST/{model.stage}_combined/nii/combined', exist_ok=True)
    os.makedirs(f'E:/radio/data/FIRST/{model.stage}_combined/nii/fusion', exist_ok=True)

    df = []
    test = range(56,62)
    atlases = range(0,50)
    tab = []
    fusion =[]
    for i in test:
        true_labels = load_nii(f'E:/radio/data/FIRST/{model.stage}/nii/true/{i:02g}.nii.gz')
        cnn_mask = load_nii( f'E:/radio/data/FIRST/{model.stage}/nii/preds/{i:02g}.nii.gz')
        preds_list = []
        combined = []
        preds_list.append(cnn_mask)
        for j in atlases:
            preds_labels = load_nii(
                f'E:/radio/data/FIRST/atlas_{atlas_stage}/nii/preds/{j:02g}_{i:02g}.nii.gz')
            preds_list.append(preds_labels)
            
            preds_labels = load_nii(
                f'E:/radio/data/FIRST/{model.stage}_combined/nii/preds/{j:02g}_{i:02g}.nii.gz')
            preds_list.append(preds_labels)
            combined.append(preds_labels)

        preds_voted = vote2(np.stack(preds_list, 2))
        img = load_nii(f'E:/radio/data/FIRST/{model.stage}/nii/img/{i:02g}.nii.gz')
        no_data_mask = (img < 0.2) | np.isnan(img)
        preds_voted = np.where(no_data_mask, 0, preds_voted)

        voted = vote2(np.stack(combined), 2)
        save_nifti_file(preds_voted.astype(np.int32), np.eye(4),
                        f'E:/radio/data/FIRST/{model.stage}_combined/nii/combined/{i:02g}')
        save_nifti_file(voted.astype(np.int32), np.eye(4),
                        f'E:/radio/data/FIRST/{model.stage}_combined/nii/fusion/{i:02g}')
        scores = [_dice(preds_voted, true_labels.astype(np.int32))]

        tab = comput_metrics(tab, preds_voted, true_labels, i)
        fusion = comput_metrics(fusion, voted, true_labels, i)

        df.append([i, *scores])
        print(scores)

    df = pd.DataFrame(df)
    df.to_csv(f'E:/radio/data/FIRST/{model.stage}_combined/voted_dsc.csv', index=False)

    tab = pd.DataFrame(tab)
    tab.to_csv(f'E:/radio/data/FIRST/{model.stage}_combined/voted_metrics.csv', index=False)
    fusion = pd.DataFrame(fusion)
    fusion.to_csv(f'E:/radio/data/FIRST/{model.stage}_combined/fusion_metrics.csv', index=False)


def comput_metrics(tab, preds_labels, true_labels, i):
    preds = prepare_mask_for_metrics(preds_labels)
    true = prepare_mask_for_metrics(true_labels)
    dsc = compute_dice(preds, true).item()
    hd = compute_hausdorff_distance(preds, true).item()
    msd = compute_average_surface_distance(preds, true).item()
    vr =  compute_volume_ratio(preds_labels, true_labels)
    precision_metric = ConfusionMatrixMetric(
    metric_name="precision", reduction="mean_batch"
    )
    recall_metric = ConfusionMatrixMetric(
    metric_name="recall", reduction="mean_batch"
    )
    ppv = precision_metric(preds, true)
    tpr = recall_metric(preds, true)
    tab.append(dict(
        test=i,
        dsc=dsc, hd=hd, vr=vr, msd=msd, ppv=ppv, tpr=tpr
    ))

    return tab

def vote10(arr):
    arr = np.asarray(arr)
    counts = np.sum(arr, axis=0)
    return np.asarray(counts >= 10, dtype=int)

