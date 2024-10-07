import pathlib
import pandas as pd
import torch
from torchmetrics.functional import dice

from dataset.datamodules import AbstractDataModule
from nn.FCNWithMask import FCNWithMask
from nn.utils import make_trainer, train_model
from utils import save_subject_np, save_subject_nii, save_nifti_file
import numpy as np
from monai.metrics import DiceMetric, compute_dice, compute_average_surface_distance, compute_hausdorff_distance
import itertools
from nn.UnetModel import UnetModel
from dataset.segmentationDataModule import SegmentationDataModule
import os
# from metrics.np import compute_average_surface_distance, compute_hausdorff_distance
def runs_segmentation(data_split, model, dm, train=True):
    # data_split = DataSplit.SECOND
    # for model, dm in [
    #     # (HeartFCNWithMask(), HeartSegmentationWithMaskDataModule(data_split)),
    #     # (ChambersFCNWithMask(), ChambersSegmentationWithMaskDataModule(data_split)),
    #     (ChambersWithHeartFCNWithMask(), ChambersWithHeartSegmentationWithMaskDataModule(data_split)),
    #     # (ChambersFCNWithMask.load_from_checkpoint(
    #     #     'logs/SECOND_ChambersFCNWithMask/version_2/checkpoints/epoch=3-step=72.ckpt'),
    #     #  ChambersSegmentationWithMaskDataModule(data_split)),
    # ]:
    #     # model = HeartFCNWithMask()
    #     # dm = HeartSegmentationWithMaskDataModule(data_split)
    # path = f'E:/radio/data/models/{dm.data_split.name}__FCN.pth'
    path = f'E:/radio/data/models/{dm.data_split.name}_{model.stage}.pth'
    if train and not pathlib.Path(path).is_file():
        train_model(model, dm)
        torch.save(model.state_dict(), path)
    else:
        model.load_state_dict(torch.load(path))
    if 'unet' in model.stage:
        predict_Unet(model, dm)
    else:
        predict_model(model, dm)


def predict_model(model: FCNWithMask, dm: AbstractDataModule):
    trainer = make_trainer(dm.data_split, type(model))
    df = []
    for i, dl in dm.iter_all_subjects():
        img = torch.concat([b[0] for b in dl])
        if len(img.shape) == 4:
            img = img[:, 0, :, :]
        preds = trainer.predict(model=model, dataloaders=dl)
        preds = torch.concat(preds)
        target = torch.concat([b[-1] for b in dl])

        dice_score = dice(preds, target, ignore_index=0).item()
        df.append(dict(
            subject=i,
            dice=dice_score,
        ))   
        print(f'{i:02g}, dice_score:  {dice_score}')
        # pred_labels = torch.argmax(preds, dim=1)

        save_subject_np(preds.numpy(), f'E:/radio/data/{dm.data_split.name}/{model.stage}/np/preds', i)

        save_subject_nii(torch.where(preds != 0, img, img * 0.5).numpy(),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/img',
                         i)
    pd.DataFrame(df).to_csv(f'E:/radio/data/{dm.data_split.name}/{model.stage}/dice.csv', index=False)

def predict_Unet(model: UnetModel, dm: SegmentationDataModule):
    dm.setup()
    trainer = make_trainer(dm.data_split, type(model))
    df = []
    dl = dm.train_dataloader()
    dv = dm.val_dataloader()
    dt = dm.test_dataloader()

    dl = itertools.chain(dl, dv, dt)
    z = trainer.predict(model, dataloaders=dl)
    # z = torch.concat(z)
    os.makedirs(f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/preds/', exist_ok=True)
    os.makedirs(f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/img/', exist_ok=True)
    os.makedirs(f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/true/', exist_ok=True)
   
    for preds, img, true, name in z:
        dsc = compute_dice(preds, true).item()
        hd = compute_hausdorff_distance(preds, true).item()
        msd = compute_average_surface_distance(preds, true).item()
        n = name.numpy()[0]

        df.append(dict(
            subject=n,
            dice=dsc,
            hd= hd,
            vr= np.mean([np.sum(p) / np.sum(l) for p, l in zip(preds, true)]),
            msd= msd
        ))   
        print(f'{n}, dice_score:  {dsc}')
        
        save_subject_np(preds.squeeze().cpu().numpy(), f'E:/radio/data/{dm.data_split.name}/{model.stage}/np/preds', n)

        save_nifti_file(preds.squeeze().cpu().numpy().astype(np.int32), np.eye(4),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/preds/{n:02g}')
        save_nifti_file(img.squeeze().cpu().numpy(), np.eye(4),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/img/{n:02g}')                   
        save_nifti_file(true.squeeze().cpu().numpy().astype(np.int32), np.eye(4),
                         f'E:/radio/data/{dm.data_split.name}/{model.stage}/nii/true/{n:02g}')            
    pd.DataFrame(df).to_csv(f'E:/radio/data/FIRST/{model.stage}/merics.csv', index=False)


    