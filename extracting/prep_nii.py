import glob
import numpy as np
from utils import save_subject_nii, flip
from tqdm.contrib.concurrent import process_map
import nibabel as nib
from scipy.ndimage import  zoom
from utils import save_nifti_file, save_subject_np, dcm_to_nii
import os
import pathlib

def prep_new(i):
    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/CTV1/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/nii/bed', i)
    # save_subject_np(y_list, f'E:/radio/data/common//np/true/CTV1', i)

    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/CTV2/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/nii/cancer_breast', i)
    # save_subject_np(y_list, f'E:/radio/data/common/total/np/true/CTV2', i)

    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/second breast/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/nii/second_breast', i)
    # save_subject_np(y_list, f'E:/radio/data/common/total/np/true/second breast', i)

    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/LAD/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/nii/LAD', i)

    y_list = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/heart/{i:02g}_*.npz')])
    save_subject_nii(y_list, f'E:/radio/data/common/nii/heart', i)

def prep_old(i):
    y_list = [
            np.stack([np.load(p)['arr_0'] for p in part])
            for part in [
                glob.glob(f'E:/radio/data/common/raw/np/heart/{i:02g}_*.npz'),
                glob.glob(f'E:/radio/data/common/raw/np/chambers/{i:02g}_*.npz'),
                glob.glob(f'E:/radio/data/common/raw/np/LAD/{i:02g}_*.npz'),
            ]
    ]
    zpp = zip([
        'heart',
        'chambers',
        'LAD'
    ], y_list)

    for name, y in zpp:
        save_subject_nii(y, f'E:/radio/data/common/nii/{name}', i)
        # save_subject_np(y, f'E:/radio/data/common/cropped_XY/np/{name}', i)

def prep_img(i):
    img = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/raw/np/img/{i:02g}_*.npz')])
    save_subject_nii(img, 'E:/radio/data/common/nii/img', i)

def run_prep():
    process_map(prep_img, range(68), max_workers=4)
    process_map(prep_old, range(0,29), max_workers=4)
    process_map(prep_new, range(29,62), max_workers=4)
    process_map(dcm_to_nii, range(63), max_workers=4)  


    # os.makedirs(f'E:/radio/data/common/all_crop/cancer_breast', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/all_crop/second_breast', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/all_crop/bed', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/all_crop/img', exist_ok=True)
    # os.makedirs(f'E:/radio/data/common/all_crop/LAD', exist_ok=True)
    # for i in range(0, 62): 
    #     prep_img_only_body(i, 'LAD')
    # for i in range(29, 62):
    #     prep_img_only_body(i, 'cancer_breast')
    #     prep_img_only_body(i, 'second_breast')
    #     prep_img_only_body(i, 'bed')

    # process_map(prep_img_only_body, range(29,62), max_workers=4)

    # prep_img_only_body(0, 'LAD')




# def prep_img_only_body(i, label):
#     image_nii = nib.load(f'E:/radio/data/common/nii/img/{i:02g}.nii.gz')
#     mask_nii = nib.load( f'E:/radio/data/common/nii/{label}/{i:02g}.nii.gz')

#     img_data = image_nii.get_fdata()
#     mask_data = flip(mask_nii.get_fdata())

#     def extract_bounding_box(img_data, threshold=-500):
#         binary_img = img_data > threshold
#         coords = np.array(np.nonzero(binary_img))
#         x_min, y_min, z_min = coords.min(axis=1)
#         x_max, y_max, z_max = coords.max(axis=1)
#         return (x_min, x_max), (y_min, y_max), (z_min, z_max)

#     def crop_to_bbox(img_data, bbox):
#         (x_min, x_max), (y_min, y_max), (z_min, z_max) = bbox
#         cropped_img = img_data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
#         return cropped_img
    
#     bbox = extract_bounding_box(img_data, threshold=-500)
#     cropped_ct = crop_to_bbox(img_data, bbox)
#     mask_data = crop_to_bbox(mask_data, bbox)    

#     original_size = cropped_ct.shape
#     new_shape = (original_size[0],512,512)
#     zoom_factors = [new_shape[i] / original_size[i] for i in range(len(original_size))]
#     cropped_ct = zoom(cropped_ct, zoom_factors, order=3)  
#     mask_data = zoom(mask_data, zoom_factors, order=0)
#     mask_data = flip(mask_data)

#     if not pathlib.Path( f'E:/radio/data/common/all_crop/img/{i:02}.nii').is_file():
#         save_subject_np(cropped_ct,  f'E:/radio/common/all_crop/np/img', i)
#         save_nifti_file(cropped_ct, image_nii.affine, f'E:/radio/data/common/all_crop/img/{i:02g}')

#     save_subject_np(mask_data,  f'E:/radio/data/common/all_crop/np/{label}', i)
#     save_nifti_file(mask_data, mask_nii.affine, f'E:/radio/data/common/all_crop/{label}/{i:02g}')


