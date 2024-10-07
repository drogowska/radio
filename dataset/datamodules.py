import itertools

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import os
from dataset.datasets import split_subjects, DataSplit, ZippedDataset, NumpyDataset, glob_subjects, \
    get_subjects_mask, NumpySliceClassDataset, ConcatenatedDataset, UnmergedLabelsDataset, MaskedDataset, FileDataset
from PIL import Image


class AbstractDataModule(LightningDataModule):
    def __init__(self, data_split: DataSplit, breast=False):
        super().__init__()
        self.data_split = data_split
        self.train_subjects, self.val_subjects, self.test_subjects = split_subjects(data_split, breast)
        self.batch_size = 2

    def dataloader_for_subjects(self, subjects, train=False, whole=False):
        raise NotImplementedError

    def train_dataloader(self):
        return self.dataloader_for_subjects(self.train_subjects, train=True, whole=False)

    def test_dataloader(self):
        return self.dataloader_for_subjects(self.test_subjects, train=False, whole=True)

    def val_dataloader(self):
        return self.dataloader_for_subjects(self.val_subjects, train=False, whole=False)

    def iter_test_subjects(self):
        return itertools.chain(  
        (
                (subject, self.dataloader_for_subjects([subject], train=True, whole=True))
                for subject in sorted(self.val_subjects)
            ),
        (
            (subject, self.dataloader_for_subjects([subject]))
            for subject in self.test_subjects
        ))

    def iter_train_subjects(self):
        return (
            (subject, self.dataloader_for_subjects([subject]))
            for subject in self.train_subjects
        )   


    def iter_all_subjects(self):
        return itertools.chain(
            (
                (subject, self.dataloader_for_subjects([subject], train=True, whole=True))
                for subject in sorted(self.train_subjects)
            ),
            (
                (subject, self.dataloader_for_subjects([subject], train=True, whole=True))
                for subject in sorted(self.val_subjects)
            ),
            (
                (subject, self.dataloader_for_subjects([subject], train=False, whole=True))
                for subject in sorted(self.test_subjects)
            ),
        )


# segmentation
class AbstractSegmentationWithMaskDataModule(AbstractDataModule):
    def __init__(self, data_split: DataSplit, target):
        super().__init__(data_split)
        self.target = target
    
    def dataloader_for_subjects(self, subjects, train=False, whole=False):
        subjects_mask = get_subjects_mask(self.data_split, subjects)

        if train:
            mask = np.asarray(list(
                NumpySliceClassDataset(glob_subjects(f'E:/radio/data/common/cropped_XY/np/{self.target}/', subjects, '*.npz'))))
        else:
            mask = np.load(f'E:/radio/data/{self.data_split.name}/slice_classifier/mask.npz')['arr_0'][subjects_mask]

        mask = np.asarray(mask, dtype=bool)
        if whole:
            return DataLoader(
                ZippedDataset([
                    self.input_dataset(subjects, None, train),
                    mask,
                    NumpyDataset(glob_subjects(f'E:/radio/data/common/cropped_XY/np/{self.target}/', subjects, '*.npz')),
                ]),
                batch_size=self.batch_size,
                # drop_last=train,
            )
        else:
            return DataLoader(
                ZippedDataset([
                    self.input_dataset(subjects, mask, train),
                    mask[mask],
                    NumpyDataset([p for p, b in
                                  zip(glob_subjects(f'E:/radio/data/common/cropped_XY/np/{self.target}/', subjects, '*.npz'),
                                      mask)
                                  if b]),
                ]),
                batch_size=self.batch_size,
                # drop_last=train,
            )

    def input_dataset(self, subjects, mask, train):
        raise NotImplementedError


# # heart

class HeartSegmentationWithMaskDataModule(AbstractSegmentationWithMaskDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'heart')

    def input_dataset(self, subjects, mask, train):
        if mask is not None:
            return NumpyDataset([
                p for p, b in
                zip(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'), mask)
                if b
            ])
        else:
            return NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'))


# # chambers

class ChambersSegmentationWithMaskDataModule(AbstractSegmentationWithMaskDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'chambers')

    def input_dataset(self, subjects, mask, train):
        return MaskedDataset(NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz')),
                             mask=mask)


class ChambersWithHeartSegmentationWithMaskDataModule(AbstractSegmentationWithMaskDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'chambers')

    def input_dataset(self, subjects, mask, train):
        if train:
            labels_path = 'E:/radio/data/common/cropped_XY/np/heart/'
        else:
            labels_path = f'E:/radio/data/{self.data_split.name}/atlas_heart/np/preds/'
        return MaskedDataset(ConcatenatedDataset([
            NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'), expand=True),
            UnmergedLabelsDataset(glob_subjects(labels_path, subjects, '*.npz'), num_classes=2),
        ]), mask=mask)


# # LAD

class LADSegmentationWithMaskDataModule(AbstractSegmentationWithMaskDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'LAD')

    def input_dataset(self, subjects, mask, train):
        chambers = 'atlas_chambers'
        # chambers = 'atlas_chambers_with_heart'
        if train:
            heart_path = 'E:/radio/data/common/cropped_XY/np/heart/'
            chambers_path = f'E:/radio/data/common/cropped_XY/np/chambers/'
        else:
            heart_path = f'E:/radio/data/{self.data_split.name}/atlas_heart/np/preds/'
            chambers_path = f'E:/radio/data/{self.data_split.name}/{chambers}/np/preds/'
        return MaskedDataset(ConcatenatedDataset([
            NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'), expand=True),
            UnmergedLabelsDataset(glob_subjects(heart_path, subjects, '*.npz'), num_classes=2),
            UnmergedLabelsDataset(glob_subjects(chambers_path, subjects, '*.npz'), num_classes=5),
        ]), mask=mask)


class LADAloneSegmentationWithMaskDataModule(AbstractSegmentationWithMaskDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'LAD')

    def input_dataset(self, subjects, mask, train):
        return MaskedDataset(NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz')),
                             mask=mask)

class AbstractSegmentationDataModule(AbstractDataModule):
    def __init__(self, data_split: DataSplit, target, total, breast):
        super().__init__(data_split, breast)
        self.target = target
        self.total = total

    def dataloader_for_subjects(self, subjects, train=True, whole=False):
        return DataLoader(ZippedDataset([
                        self.input_dataset(subjects, train),
                        NumpyDataset(glob_subjects(f'E:/radio/data/common/{self.total}/np/{self.target}/', subjects, '*.npz')),
                    ]),
                    batch_size=self.batch_size,
                    # num_workers=15,
                    # persistent_workers=val
                    # shuffle=train
            )

    def input_dataset(self, subjects,  train):
        raise NotImplementedError



#  LAD cropped cube
class LADCubeSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'LAD', 'total/crop', False)

    def input_dataset(self, subjects, train):
         return NumpyDataset(glob_subjects('E:/radio/data/common/total/crop/np/img/', subjects, '*.npz'))

class LADCubeWithChambersSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'LAD', 'total/crop', False)

    def input_dataset(self, subjects, train):
        return ConcatenatedDataset([
            NumpyDataset(glob_subjects('E:/radio/data/common/total/crop/np/img/', subjects, '*.npz'), expand=True),
            UnmergedLabelsDataset(glob_subjects(f'E:/radio/data/common/total/crop/np/chambers/', subjects, '*.npz'), num_classes=3),
        ])
#  all
class LADSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'LAD', 'cropped_XY', False)

    def input_dataset(self, subjects, train):
        return ConcatenatedDataset([
            NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'), expand=True),
            UnmergedLabelsDataset(glob_subjects(f'E:/radio/data/common/cropped_XY/np/heart/', subjects, '*.npz'), num_classes=2),
        ])
    
    




class CancerBedCubeSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'bed', 'total_breast/crop', True)

    def input_dataset(self, subjects, train):
        return NumpyDataset(glob_subjects('E:/radio/data/common/total_breast/crop/np/img_left/', subjects, '*.npz'))

class CancerBedCubeWithBreastSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'bed', 'total_breast/crop', True)

    def input_dataset(self, subjects, train):
        return ConcatenatedDataset([
            NumpyDataset(glob_subjects('E:/radio/data/common/total_breast/crop/np/img_left/', subjects, '*.npz'), expand=True),
            UnmergedLabelsDataset(glob_subjects(f'E:/radio/data/common/total_breast/crop/np/cancer_breast/', subjects, '*.npz'), num_classes=3),
        ])
    

class CancerBreastCubeSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'cancer_breast', 'total_breast/crop', True)

    def input_dataset(self, subjects, train):
        return NumpyDataset(glob_subjects('E:/radio/data/common/total_breast/crop/np/img_left/', subjects, '*.npz'))

class CancerBreastSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'cancer_breast', 'cropped_XY', True)

    def input_dataset(self, subjects, train):
        return NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'))



class SecondBreastCubeSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'second_breast', 'total_breast/crop', True)

    def input_dataset(self, subjects, train):
        return NumpyDataset(glob_subjects('data/common/total_breast/crop/np/img_right/', subjects, '*.npz'))

class SecondBreastSegmentationModule(AbstractSegmentationDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split, 'second_breast', 'cropped_XY', True)

    def input_dataset(self, subjects, train):
        return NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz'))






# classification
# # heart

class HeartSliceClassificationDataModule(AbstractDataModule):
    def __init__(self, data_split: DataSplit):
        super().__init__(data_split)
        self.batch_size = 8

    def dataloader_for_subjects(self, subjects, train=False, whole=False):
        return DataLoader(
            ZippedDataset([
                NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz')),
                NumpySliceClassDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/heart/', subjects, '*.npz')),
            ]),
            batch_size=self.batch_size
        )
