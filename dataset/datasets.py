import glob
import itertools
from enum import Enum, auto
from typing import Optional, Sequence
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class DataSplit(Enum):
    FIRST = auto()
    SECOND = auto()


class FileDataset(Dataset):
    def __init__(self, files: list[str]):
        assert len(files) > 0
        self.files = files

    def __len__(self):
        return len(self.files)
    
class SliceDataset(Dataset):
    def __init__(self, root, mode="train", transform=None, data_split: DataSplit=None, breast=False):
        assert mode in {"train", "valid", "test"}

        self.root = '.'

        self.mode = mode
        self.transform = transform
        self.data_split = data_split
        self.train_subjects, self.val_subjects, self.test_subjects = (list(range(5)),
                            list(range(50, 57)),
                            list(range(57, 62)))

        self.ct_files = [f for f in os.listdir(f'./img') if f.endswith('.png')]
        self.mask_files = [f for f in os.listdir(f'./lad') if f.endswith('.png')]


        # self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        filename = self.ct_files[idx]
        # image_path = os.path.join('.', filename )
        # mask_path = os.path.join('.', filename)

        image = np.array(Image.open(f'./img/{filename}').convert("RGB"))
        mask = np.array(Image.open(f'./lad/{filename}'))
        sample = dict(image=image, mask=mask)

        # ct_path = os.path.join(self.root,'img', self.ct_files[idx])
        # with np.load(ct_path) as data:
        #     ct_scan = data['arr_0']

        # mask_path = os.path.join(self.root,'lad', self.mask_files[idx])
        # with np.load(mask_path) as data:
        #     mask_scan = data['arr_0']
        # sample = dict(image=ct_scan, mask=mask_scan)

        image = np.array(
            Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        )

        # image = np.array(Image.open(ct_scan).convert("RGB"))


        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)

        return sample

    def _read_split(self):
        if self.mode == "train":  # 90% for train
            filenames = self.train_subjects
        elif self.mode == "valid":  # 10% for validation
            filenames =self.val_subjects
        return filenames

class NumpyDataset(FileDataset):
    def __init__(self, files: list[str], expand=False):
        super().__init__(files)
        self.expand = expand

    def __getitem__(self, item):
        arr = np.load(self.files[item])['arr_0']

        # img = Image.fromarray(arr)

        t = arr.astype(np.uint8)

        # image = np.array(Image.open(ct_scan).convert("RGB"))
        t = torch.tensor(t)

        if self.expand:
            t = t[None]
        # t = Image.fromarray(t.tolist())  # Convert ndarray to PIL Image
        # t = self.transform(t)
        return t

def normalize_data(data):
    """Normalize the image data to the range [0, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        normalized_data = np.zeros_like(data)
    else:
        normalized_data = 255 * (data - min_val) / (max_val - min_val)
    return normalized_data

def normalise(arr):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225]) # use mean and std from ImageNet
    channels = arr.shape[1]
    mean_expanded = mean[:, :channels, :, :].expand(arr.shape[0], channels, arr.shape[2], arr.shape[3])
    mean_expanded = mean_expanded.to(arr.device)
  
    std_expanded = std[:, :channels, :, :].expand(arr.shape[0], channels, arr.shape[2], arr.shape[3])
    std_expanded = std_expanded.to(arr.device)

    arr = (arr - mean_expanded) / std_expanded
    return arr


class NumpySliceClassDataset(NumpyDataset):
    def __getitem__(self, item):
        return torch.tensor(np.any(np.load(self.files[item])['arr_0'], axis=(-2, -1)))


class ConcatenatedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item):
        return torch.cat([
            dataset[item]
            for dataset in self.datasets
        ])


class ZippedDataset(Dataset):
    def __init__(self, datasets: list[FileDataset]):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, item):
        return tuple([
            dataset[item]
            for dataset in self.datasets
        ])


class UnmergedLabelsDataset(NumpyDataset):
    def __init__(self, files: list[str], num_classes):
        super().__init__(files)
        self.num_classes = num_classes

    def __getitem__(self, item):
        t = super().__getitem__(item)
        t = torch.stack([
            torch.where(t == i, i * 1.0, 0.0)
            for i in range(1, self.num_classes)
        ])
        return t


class MaskedDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: Optional[Sequence[bool]]):
        self.dataset = dataset
        self.mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(len(dataset), dtype=bool)
        self.indices = np.arange(len(self.mask))[self.mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


def glob_subjects(base: str, subjects: list[int], ext: str) -> list[str]:
    return list(itertools.chain(*[
        glob.glob(f'{base}{subject:02g}{ext}')
        for subject in subjects
    ]))


def split_subjects(data_split: DataSplit, breast=False) -> tuple[list[int], list[int], list[int]]:
    if breast:
        return {
            DataSplit.FIRST: (list(range(29, 56)),
                            list(range(56, 59)),
                            list(range(59, 62))),
            DataSplit.SECOND: (list())
       }[data_split]
    else:
       return {
            DataSplit.FIRST: (list(range(50)),
                            list(range(50, 56)),
                            list(range(56, 62))),
            DataSplit.SECOND: (list(range(62, 65)),
                            list(range(65, 67)),
                            list()),  
            # DataSplit.FIRST: (list(range(50)),
            #                   list(range(50, 62))),
            # DataSplit.SECOND: (list(range(62, 65)),
            #                    list(range(65, 67))),        
            # DataSplit.FIRST: (list(range(4)),
            #                   list(range(4, 6))),
            # DataSplit.SECOND: (list(range(6, 9)),
            #                    list(range(9, 10))),
       }[data_split]


# @lru_cache(maxsize=4)
def get_subjects_mask(data_split: DataSplit, subjects: list[int]):
    tr, va, te = split_subjects(data_split)
    # all_subjects = glob.glob('E:/radio/data/common/cropped_XY/np/img/*.npz')
    all_subjects = glob_subjects('E:/radio/data/common/cropped_XY/np/img/', tr + va + te, '*.npz')
    select = glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz')
    return np.asarray([
        x in select
        for x in all_subjects
    ])


# ready to use datasets
# ---------------------

class HeartFCNWithMaskDataset(ZippedDataset):
    def __init__(self, data_split: DataSplit, test: bool):
        subjects = split_subjects(data_split)[1 if test else 0]
        super().__init__([
            NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/img/', subjects, '*.npz')),
            np.load(f'E:/radio/data/{data_split.name}/cropped_XY/mask.npz')['arr_0'],
            NumpyDataset(glob_subjects('E:/radio/data/common/cropped_XY/np/heart/', subjects, '*.npz')),
        ])
