import itertools
import monai
import torch
from torch.utils.data import DataLoader, random_split
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped
)
from monai.data import Dataset, CacheDataset
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import os
from glob import glob
import numpy as np

from dataset.datasets import DataSplit


class SegmentationDataModule(LightningDataModule):
    def __init__(self, data_dir, target, img, atlas=False, batch_size=1, val_split=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.target = target
        # self.num_workers = num_workers
        self.data_split = DataSplit.FIRST
        self.img = img
        self.atlas=atlas

        self.train_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image", "mask"], spatial_size=(256, 256, 256)),
            EnsureTyped(keys=["image", "mask"])
        ])

        self.val_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image", "mask"], spatial_size=(256, 256, 256)),
            EnsureTyped(keys=["image", "mask"])
        ])

        self.pred_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(256, 256, 256)),
            EnsureTyped(keys=["image"])
        ])

    def setup(self, stage=None):
        if self.target == 'LAD':
            names = range(0,62)
        else:
            names = range(29,62)
            # ext = '*.nii.gz'
        ext ='*.nii.gz'

        if self.atlas:
            path = f'{self.data_dir}/img/*.nii.gz'
            images = sorted(glob(path))
            names = [os.path.basename(path) for path in images]
            data_dicts = [{"image": img, "mask": "none", "name": name[0:5]} for img, name in zip(images, names)]
            self.test_ds = CacheDataset(data=data_dicts[100:], transform=self.pred_transforms, cache_rate=0.25)

        else:
            images = sorted(glob(os.path.join(self.data_dir, self.img,  ext)))
            masks = sorted(glob(os.path.join(self.data_dir, self.target, ext)))
            if self.target == 'CTV1':
                images = images[29:62]

            data_dicts = [{"image": img, "mask": mask, "name": name} for img, mask, name in zip(images, masks , names)]

            val_size = int(len(data_dicts) * self.val_split)
            train_size = len(data_dicts) - val_size
            val_size = int(val_size / 2)
            test_size = int(val_size)

            if self.target == 'LAD':
                self.train_ds, self.val_ds, self.test_ds = data_dicts[0:50], data_dicts[50:56], data_dicts[56:62]
            else:
                self.train_ds, self.val_ds, self.test_ds = data_dicts[0:27], data_dicts[27:30], data_dicts[30:33]
            # self.train_ds, self.val_ds, self.test_ds = data_dicts[0:6], data_dicts[6:12], data_dicts[12:18]
            # self.train_ds, self.val_ds, self.test_ds = random_split(data_dicts, [train_size, val_size, test_size])
    # 
            self.train_ds = CacheDataset(data=self.train_ds, transform=self.train_transforms, cache_rate=0.1)
            self.val_ds = CacheDataset(data=self.val_ds, transform=self.val_transforms, cache_rate=0.25)
            self.test_ds = CacheDataset(data=self.test_ds, transform=self.val_transforms, cache_rate=0.25 )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)
    
    def get_all_loaders(self):
        return itertools.chain( self.train_dataloader, self.val_dataloader, self.test_dataloader)

    def get_test_names(self):
        return [d.get("name") for d in self.test_ds]

class LADcubeSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir="E:/radio/data/common/total/crop", target='LAD', img='img')

class AtlasLADcubeSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir="E:/radio/data/FIRST/atlas_crop_LAD/nii/", target='LAD', img='img',  atlas=True)




class LADSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir='E:/radio/data/common/cropped_XY/nii', target='LAD', img='img')

class AtlasLADSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir="E:/radio/data/FIRST/atlas_all_LAD/nii/", target='LAD', img='img',  atlas=True)




class BreastCancerCubeSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir="E:/radio/data/common/total_breast/crop", target='cancer_breast', img='img_left')

class BreastCancerSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir='E:/radio/data/common/cropped_XY/nii', target='CTV2', img='img')
    

class CancerBedCubeSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir="E:/radio/data/common/total_breast/crop", target='bed', img='img_left')

class CancerBedSegmentationDM(SegmentationDataModule):
    def __init__(self):
        super().__init__( data_dir='E:/radio/data/common/cropped_XY/nii', target='CTV1', img='img')
      