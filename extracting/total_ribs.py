from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
import numpy as np
from tqdm.contrib.concurrent import process_map
import os
import glob

from utils import save_subject_nii

def prep_true(i):
    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/CTV2/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/total_breast/nii/true/CTV2', i)

    # save_subject_np(y_list, f'E:/radio/data/common/total_breast/np/true/LAD', i)

def total_ribs(i):
    # totalsegmentator(f"E:/radio/data/common/total/nii/img/{i:02g}.nii", f'E:/radio/data/common/total_breast/nii/seg/{i:02g}', ml=True, task="total",  device="cuda", quiet=False )
    # costal cartilage - chrząstka żebrowa

    #  save only ribsz
    # tissue_type - tłuszcz, miesnie i cos jeszcze
    seg_nifti = nib.load(f'E:/radio/data/common/total_breast/nii/seg/{i:02g}.nii')
    seg_data = seg_nifti.get_fdata()
    # condition = np.logical_or(seg_data == 117, seg_data == 116)
    seg_data = np.where(seg_data < 116, 0, seg_data)
    seg_data = np.where(seg_data == 116, 1, seg_data)
    seg_data = np.where(seg_data == 117, 2, seg_data)

    seg2_affine = seg_nifti.affine
    rotated_data = np.flip(seg_data, axis=0)  
    rotated_data = np.flip(rotated_data, axis=1)  
    nii = nib.Nifti1Image(rotated_data, seg2_affine)
    nib.save(nii, f'E:/radio/data/common/total_breast/nii/res/{i:02g}.nii.gz')

def run_total_ribs():
    # os.makedirs(f'E:/radio/data/common/total/nii/true', exist_ok=True)
    # process_map(prep_true, range(29,62), max_workers=4)
    # os.makedirs('E:/radio/data/common/total/nii/img', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/total_breast/nii/seg', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/total_breast/nii/res', exist_ok=True)
    process_map(total_ribs, range(29,62), max_workers=1)  
    # for i in range(29,62):
    # total_ribs(59)