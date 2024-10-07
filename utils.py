import glob
import os
from pathlib import Path

import PIL.Image
import nibabel
import numpy as np
import pydicom
import SimpleITK as sitk
import torch

def makedirs(name):
    return os.makedirs(name, exist_ok=True)


def makedirs_file(file):
    file = Path(file).resolve().parent
    file.mkdir(parents=True, exist_ok=True)


def savez_np(file, *args, **kwargs):
    makedirs_file(file)
    np.savez_compressed(file, *args, **kwargs)


def window(scan):
    return np.clip((scan - (40 - 200)) / 400, 0, 1)


def save_img_combined(file, scan, m1, m2=None):
    makedirs_file(file)
    scan = PIL.Image.fromarray(np.asarray(window(scan) * 255, dtype=np.uint8), mode='L')
    m1 = PIL.Image.fromarray(np.asarray(m1 * 255, dtype=np.uint8), mode='L')
    m2 = PIL.Image.fromarray(np.asarray(m2 * 255, dtype=np.uint8), mode='L') if m2 is not None else PIL.Image.new('L',
                                                                                                                  size=scan.size)
    PIL.Image.merge('RGB', [
        m1,
        scan,
        m2
    ]).save(file)


def get_affine(i):
    dcm = glob.glob(f'E:/radio/data/dicom/{i + 1}/CT*')[0]
    dcm = pydicom.dcmread(dcm, force=True)
    spacing_y, spacing_x = tuple(dcm.PixelSpacing)
    spacing_z = dcm.SliceThickness
    affine = np.eye(4)
    affine[0, 0] = spacing_x
    affine[1, 1] = spacing_y
    affine[2, 2] = spacing_z
    return affine


def get_affine_cropped(i):
    return np.load(f'E:/radio/data/common/cropped_XY/affine/{i:02}.npz')['arr_0']

def get_affine_cropped(total, i):
    return np.load(f'E:/radio/data/common/{total}/crop/affine/{i:02}.npz')['arr_0']


def save_subject_np(arr, path, s: int):
    os.makedirs(path, exist_ok=True)
    for i, layer in enumerate(arr):
        np.savez_compressed(f'{path}/{s:02g}_{i:04g}.npz', layer)


def save_subject_nii(arr, path, s: int):
    os.makedirs(path, exist_ok=True)
    arr = np.moveaxis(arr, (0, 1, 2), (2, 1, 0))[::-1, ::-1, ::-1] 
    affine = get_affine(s)
    nii = nibabel.Nifti1Image(arr, affine)
    nibabel.save(nii, f'{path}/{s:02g}.nii.gz')



def dcm_to_nii(i, dicom_dir='E:/radio/data/dicom/', dicoms=35):
    # for i in range(dicoms):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(f'{dicom_dir}/{i+1}')
        if i < 63:
            series_ids = series_ids[1]
        else :
            series_ids = series_ids[0]
        dicom_series = reader.GetGDCMSeriesFileNames(f'{dicom_dir}/{i+1}', series_ids)
        reader.SetFileNames(dicom_series)
        sitk_image = reader.Execute()
        path = 'E:/radio/data/common/total/nii/img'
        # if dicom_dir == 'E:/radio/data/dicom_v2/':
        #     path += '2' 
        sitk.WriteImage(sitk_image, f"{path}/{i:02g}.nii")
        # save_subject_np(sitk_image, f"E:/radio/data/common/total/np/img/", i)

def load_from_nii(path):
    # noinspection PyTypeChecker
    nii: nibabel.Nifti1Image = nibabel.load(path)
    # noinspection PyTypeChecker
    arr: np.ndarray = nii.dataobj
    arr = np.moveaxis(arr, (0, 1, 2), (2, 1, 0))[::-1, ::-1, ::-1]
    return arr


def compress_nii_to_npz(path, dtype=np.float32):
    nii = nibabel.load(path)
    arr = np.asarray(nii.dataobj, dtype=dtype)
    affine = nii.affine
    nii = nibabel.Nifti1Image(arr, affine)
    nibabel.save(nii, path)

def save_nifti_file(image_data, affine, file_path):
    """Save the NIfTI file."""
    nii = nibabel.Nifti1Image(image_data, affine=affine)
    nibabel.save(nii,  f'{file_path}.nii.gz')

def load_nii(path):
    # noinspection PyTypeChecker
    nii: nibabel.Nifti1Image = nibabel.load(path)
    # noinspection PyTypeChecker
    # arr: np.ndarray = nii.dataobj
    # arr = np.moveaxis(arr, (0, 1, 2), (2, 1, 0))[::-1, ::-1, ::-1]
    return nii.get_fdata()

def flip(mask):
    rotated_data = np.flip(mask, axis=0)  
    mask_data = np.flip(rotated_data, axis=1) 
    return mask_data 

def prepare_mask_for_metrics(mask):
    """
    Ensure the mask has the correct shape (C, 1, H, W, D).
    
    Parameters:
        mask (np.ndarray): Input mask.
    
    Returns:
        tensor: Mask with the correct shape.
    """
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)

    return torch.tensor(mask.copy())
