
from scipy.ndimage import binary_dilation, zoom

import nibabel as nib
import os

import torch
from extracting.crop_xy import cropping_bbox
from utils import save_nifti_file, savez_np, save_subject_np
import numpy as np
from extracting.crop_lad import adjust_contrast, flip
from tqdm.contrib.concurrent import process_map
from skimage import exposure
from torchvision.transforms.functional import resized_crop, gaussian_blur, InterpolationMode

def crop_breasts():
    os.makedirs(f'E:/radio/data/common/total_breast/crop/cancer_breast', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total_breast/crop/second_breast', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total_breast/crop/bed', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total_breast/crop/img_left', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total_breast/crop/img_right', exist_ok=True)

    process_map(crop, range(29,62), max_workers=4)
    # for i in range(29,62):
    # crop(40)

def crop(i):
    
    image_nii = nib.load(f'E:/radio/data/common/total/nii/img/{i:02g}.nii')
    ctv2_nii = nib.load( f'E:/radio/data/common/nii/cancer_breast/{i:02g}.nii.gz')
    ctv1_nii = nib.load( f'E:/radio/data/common/nii/bed/{i:02g}.nii.gz')
    second_breast_nii = nib.load( f'E:/radio/data/common/nii/second_breast/{i:02g}.nii.gz')
    ribs = nib.load( f'E:/radio/data/common/total_breast/nii/res/{i:02g}.nii.gz')
   
    img_data = image_nii.get_fdata()
    ctv2_data = flip(ctv2_nii.get_fdata())
    ctv1_data = flip(ctv1_nii.get_fdata())
    second_data = flip(second_breast_nii.get_fdata())
    ribs_data = flip(ribs.get_fdata())

    # img = remove_heart(img_data, i)

    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bounding_box(ribs_data, img_data)
    segmented_region = img_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    second_data = second_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    
    original_size = segmented_region.shape
    new_shape = (original_size[0],256,256)
    zoom_factors = [new_shape[i] / original_size[i] for i in range(len(original_size))]
    segmented_region = zoom(segmented_region, zoom_factors, order=3)  # Use order=3 for cubic interpolation
    second_data = zoom(second_data, zoom_factors, order=0)
    segmented_region = contrast_stretching(segmented_region)

    save_nifti_file(segmented_region, image_nii.affine, f'E:/radio/data/common/total_breast/crop/img_right/{i:02g}')
    save_nifti_file(second_data, second_breast_nii.affine, f'E:/radio/data/common/total_breast/crop/second_breast/{i:02g}')
    
    # save_subject_np(second_data,  f'E:/radio/data/common/total_breast/crop/np/second_breast', i)
    # save_subject_np(segmented_region,  f'E:/radio/data/common/total_breast/crop/np/img_right', i)
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bounding_box(ribs_data,  img_data, left=True)
    segmented_region = img_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    ctv1_data = ctv1_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    cancer_breast = ctv2_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

    original_size = segmented_region.shape
    new_shape = (original_size[0],256,256)
    zoom_factors = [new_shape[i] / original_size[i] for i in range(len(original_size))]
    segmented_region = zoom(segmented_region, zoom_factors, order=3)  # Use order=3 for cubic interpolation
    ctv1_data = zoom(ctv1_data, zoom_factors, order=0)
    cancer_breast = zoom(cancer_breast, zoom_factors, order=0)
    segmented_region = contrast_stretching(segmented_region)


    l, t, w, h = cropping_bbox(segmented_region)
    new_affine(i, h, w)
    save_nifti_file(ctv1_data, ctv1_nii.affine, f'E:/radio/data/common/total_breast/crop/bed/{i:02g}')
    save_nifti_file(segmented_region, image_nii.affine, f'E:/radio/data/common/total_breast/crop/img_left/{i:02g}')
    save_nifti_file(cancer_breast, ctv2_nii.affine, f'E:/radio/data/common/total_breast/crop/cancer_breast/{i:02g}')
    # save_subject_np(ctv1_data,  f'E:/radio/data/common/total_breast/crop/np/bed', i)
    # save_subject_np(segmented_region,  f'E:/radio/data/common/total_breast/crop/np/img_left', i)
    # save_subject_np(cancer_breast,  f'E:/radio/data/common/total_breast/crop/np/cancer_breast', i)

def contrast_stretching(image, min_percentile=2, max_percentile=98):
    """
    Apply contrast stretching to the image based on given percentiles.

    Parameters:
    image (numpy array): Input image to be stretched.
    min_percentile (float): Lower percentile for contrast stretching (default=2).
    max_percentile (float): Upper percentile for contrast stretching (default=98).

    Returns:
    numpy array: Contrast-stretched image.
    """
    p2, p98 = np.percentile(image, (min_percentile, max_percentile))
    stretched_image = np.clip((image - p2) / (p98 - p2), 0, 1)
    # stretched_image = stretched_image * 255  # Scale to 0-255 range
    return stretched_image


def get_bounding_box(segmentation_data, img_data, left=False):
    # chrząstka
    seg_data = np.where(segmentation_data == 1)
    # mostek
    seg_data1 = np.where(segmentation_data == 2)

    img = np.where(img_data > -500 )
    min_img_x, max_img_x = np.min(img[0]), np.max(img[0])
    min_img_y, max_img_y = np.min(img[1]), np.max(img[1])
    min_img_z, max_img_z = np.min(img[2]), np.max(img[2])

    min_x, max_x = np.min(seg_data[0]), np.max(seg_data[0])
    min_y, max_y = np.min(seg_data[1]), np.max(seg_data[1])
    min_z, max_z = np.min(seg_data[2]), np.max(seg_data[2])
   
    min1_x, max1_x = np.min(seg_data1[0]), np.max(seg_data1[0])
    min1_y, max1_y = np.min(seg_data1[1] ), np.max(seg_data1[1])
    min1_z, max1_z = np.min(seg_data1[2]), np.max(seg_data1[2])

    # min_y, max_y = int(min1_y - 50), int(max_y + 70)
    # min_y, max_y = int(min_img_y), int(max1_y)
    min_y, max_y = int(min_img_y), int(max1_y + 50)
    min_z, max_z = int(min_z - 20), int(max1_z + 30)
    # min_z, max_z = int(min_z - 20), int(max1_z + 50)
    if left:
        min_x, max_x = int(min_x) , int(max_img_x) 
    else: 
        min_x, max_x = int(min_img_x) , int(max_x) 
    
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def remove_inner(img, ribs_mask):
    extension_pixels = 10
    segmented_mask = (ribs_mask > 0)
    # extended_mask = binary_dilation(segmented_mask, iterations=extension_pixels)
    # extended_seg_data = np.where(extended_mask, 1, 0)

    # segmented_img_data = img * (extended_seg_data > 0)
    # segmented_img_data[segmented_img_data == 0] = -1000
    # segmented_mask = np.triu(segmented_mask)
    # masked_data = np.copy(img)
    # for i in range(masked_data.shape[2]):
    img[segmented_mask] = 0

    # z_indices = np.any(segmented_mask, axis=(1, 2))
    # topmost_slice = np.argmax(z_indices)
    
    # # Create a mask to keep only the areas below the topmost skin or bone voxel
    # keep_mask = np.zeros_like(img, dtype=bool)
    # keep_mask[:, :topmost_slice + 1, :] = True
    
    # # Apply the mask to the image data
    # modified_data = np.where(keep_mask, img, 0)
    
    # segmented_region = img[:, :max_y, :]

    return img

def remove_heart(img, i):
    heart = nib.load( f'E:/radio/data/common/nii/heart/{i:02g}.nii.gz').get_fdata()
    heart = flip(heart)
    margin_size = 10
    heart = heart > 0
    dilated_mask = binary_dilation(heart, iterations=margin_size)
    img_new = np.where(dilated_mask, -1000, img)
    return img_new

def new_affine(i, h, w):
    affine = np.load(f'E:/radio/data/common/raw/affine/{i:02g}.npz')['arr_0']
    affine[0] /= 256 / w
    affine[1] /= 256 / h
    savez_np(f'E:/radio/data/common//total_breast/crop/affine/{i:02g}.npz', affine)
