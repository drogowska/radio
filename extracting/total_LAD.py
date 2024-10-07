from totalsegmentator.python_api import totalsegmentator
from utils import  dcm_to_nii, save_subject_nii, save_subject_np
import pandas as pd
import torch
from torchmetrics.functional import dice
import numpy as np
import glob 
import nibabel as nib
import os
from tqdm.contrib.concurrent import process_map
from extracting.crop_xy import bbox_for_scan_3d, resized_crop, InterpolationMode
DICOM_NUM = 68

# from torchvision.transforms.functional import resized_crop, gaussian_blur, InterpolationMode


def prep_true(i):
    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/LAD/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/total/nii/true/LAD', i)
    save_subject_np(y_list, f'E:/radio/data/common/total/np/true/LAD', i)


def run_total():
    # os.makedirs(f'E:/radio/data/common/total/nii/true', exist_ok=True)
    # process_map(prep_true, range(DICOM_NUM), max_workers=4)
    os.makedirs('E:/radio/data/common/total/nii/img', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total/nii/seg', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total/nii/res', exist_ok=True)
    df = []
    process_map(total, range(0,1), max_workers=4)  
    # process_map(edit, range(30,31), max_workers=4)  

    # for i in range(DICOM_NUM):
    #   total(i)

        # save_subject_np(nii, 'E:/radio/data/common/total/np/res', i)
    #     true = nib.load(f'E:/radio/data/common/total/nii/true/chambers/{i:02g}.nii.gz').dataobj
    #     true = np.moveaxis(true, (0, 1, 2), (2, 1, 0))[::-1, ::-1, ::-1]
    #     nii = nib.load(f'E:/radio/data/common/total/nii/res/{i:02g}.nii.gz')

    #     threshold = 0.5  
    #     binary_mask = np.where(true > threshold, 1, 0)
    #     pred = nii.get_fdata()
    #     pred = np.moveaxis(pred, (0, 1, 2), (2, 1, 0))[::-1, ::-1, ::-1]
    #     scores = [dice(torch.tensor(pred.copy()),
    #                    torch.tensor(binary_mask.copy()),
    #                    ignore_index=0).numpy()]
    #     df.append([i, *scores])
    # df = pd.DataFrame(df, columns=['patient no.', 'dice score'])
    # df.to_csv(f'E:/radio/data/common/total/scores.csv', index=False)
    # print(df)

def total(i):
    totalsegmentator(f"E:/radio/data/common/total/nii/img/{i:02g}.nii.gz", f'E:/radio/data/common/total/nii/seg/{i:02g}', ml=True, task="heartchambers_highres",  device="cuda", quiet=True )
    seg_nifti = nib.load(f'E:/radio/data/common/total/nii/seg/{i:02g}.nii')
    seg_data = seg_nifti.get_fdata()
    condition = np.logical_or(seg_data == 7, seg_data == 6)
    seg_data = np.where(condition, 0, seg_data)
    seg_data = np.where(seg_data==3, 1, seg_data)
    seg_data = np.where(seg_data==4, 3, seg_data)
    seg_data = np.where(seg_data==5, 4, seg_data)

    seg2_affine = seg_nifti.affine
    rotated_data = np.flip(seg_data, axis=0)  
    rotated_data = np.flip(rotated_data, axis=1)  
    nii = nib.Nifti1Image(rotated_data, seg2_affine)
    nib.save(nii, f'E:/radio/data/common/total/nii/res/{i:02g}.nii.gz')
