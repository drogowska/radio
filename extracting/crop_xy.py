import glob

import numpy as np
import torch
from torchvision.transforms.functional import resized_crop, gaussian_blur, InterpolationMode
from tqdm import trange
from tqdm.contrib.concurrent import process_map
from utils import save_subject_np, save_subject_nii, savez_np, flip
DICOM_NUM = 68
import nibabel as nib

def crop_xy_all():
    process_map(crop_xy, range(DICOM_NUM), max_workers=4)
    # for i in trange(0,29):
    # crop_xy(0)


def crop_xy(i):
    img = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/img/{i:02g}_*.npz')])

    if i < 29 or i > 61 :
        y_list = [
            np.stack([np.load(p)['arr_0'] for p in part])
            for part in [
                glob.glob(f'E:/radio/data/common/raw/np/heart/{i:02g}_*.npz'),
                # glob.glob(f'E:/radio/data/common/raw/np/chambers/{i:02g}_*.npz'),
                # glob.glob(f'E:/radio/data/common/raw/np/LAD/{i:02g}_*.npz'),
            ]
        ]

    if i > 28 and i < 62:
        y_list = [
                np.stack([np.load(p)['arr_0'] for p in part])
                for part in [
                    glob.glob(f'E:/radio/data/common/raw/np/heart/{i:02g}_*.npz'),
                    # glob.glob(f'E:/radio/data/common/raw/np/second breast/{i:02g}_*.npz'),
                    # glob.glob(f'E:/radio/data/common/raw/np/CTV1/{i:02g}_*.npz'),
                    # glob.glob(f'E:/radio/data/common/raw/np/CTV2/{i:02g}_*.npz'),
                    # glob.glob(f'E:/radio/data/common/raw/np/LAD/{i:02g}_*.npz'),
                ]
            ]
    


    l, t, w, h = bbox_for_scan_3d(np.asarray(img))

    new_affine(i, h, w)

    img = torch.tensor(img)
    y_list = [torch.tensor(y) for y in y_list]

    img = resized_crop(img, t, l, h, w, size=[512, 512])
    y_list = [resized_crop(y, t, l, h, w, size=[512, 512], interpolation=InterpolationMode.NEAREST) for y in y_list]

    img = img.numpy()[:, :, :]
    y_list = [y.numpy()[:, :, :] for y in y_list]

    save_subject_nii(img, 'E:/radio/data/common/cropped_XY/nii/img', i)
    save_subject_np(img, 'E:/radio/data/common/cropped_XY/np/img', i)
    if i < 29 or i > 61 :
        for name, y in zip([
                # 'heart',
                'chambers',
                # 'LAD'
            ], y_list):
            save_subject_nii(y, f'E:/radio/data/common/cropped_XY/nii/{name}', i)
            # save_subject_np(y, f'E:/radio/data/common/cropped_XY/np/{name}', i)
    if i > 28 and i < 62:
        for name, y in zip([
            'heart',
            # 'second breast',
            # 'CTV1',
            # 'CTV2',
            # 'LAD'
        ], y_list):
            save_subject_nii(y, f'E:/radio/data/common/cropped_XY/nii/{name}', i)
            # save_subject_np(y, f'E:/radio/data/common/cropped_XY/np/{name}', i)



def bbox_for_scan_3d(scan3d: np.ndarray):
    """
    @param scan3d: ndarray (N, H, W)
    @return:
    """
    scan3d = gaussian_blur(torch.as_tensor(scan3d).float(), [9, 9]).numpy()
    std2d = np.std(scan3d, axis=0)
    threshold = 0.2  # chosen experimentally

    mask = std2d > threshold  

    bbox = cropping_bbox(mask)
    return bbox


def cropping_bbox(img):
    a = np.where(img != 0)
    rmin = np.min(a[0])
    cmin = np.min(a[1])
    rmax = np.max(a[0])
    cmax = np.max(a[1])
    bbox = cmin, rmin, cmax - cmin, rmax - rmin
    return bbox


def new_affine(i, h, w):
    affine = np.load(f'E:/radio/data/common/raw/affine/{i:02g}.npz')['arr_0']
    affine[0] /= 512 / w
    affine[1] /= 512 / h
    savez_np(f'E:/radio/data/common/cropped_XY/affine/{i:02g}.npz', affine)
