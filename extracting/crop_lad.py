import glob
from itertools import product
import nibabel as nib
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os
import torch
from torchvision.transforms.functional import resized_crop, gaussian_blur, InterpolationMode
from skimage import exposure
from scipy.ndimage import zoom

from atlas.greedy import register_deformable, register_moments, transform
from extracting.crop_xy import cropping_bbox
from utils import makedirs, save_subject_np, save_subject_nii, save_nifti_file, flip, load_from_nii
DICOM_NUM = 68 
import os
from dataset.datasets import split_subjects, DataSplit
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from utils import savez_np
import PIL

def crop_lad():
    os.makedirs(f'E:/radio/data/common/total/crop/LAD', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total/crop/chambers', exist_ok=True)
    os.makedirs(f'E:/radio/data/common/total/crop/img', exist_ok=True)

    process_map(crop, range(62), max_workers=4)
    # for i in range(62):
    #    crop(i)
    # crop(0)

def crop(i):

    image_nii = nib.load(f'E:/radio/data/common/total/nii/img/{i:02g}.nii')
    mask_nii = nib.load( f'E:/radio/data/common/nii/LAD/{i:02g}.nii.gz')
    chambers_nii = nib.load( f'E:/radio/data/common/total/nii/res/{i:02g}.nii.gz')
    # image_nii = load_from_nii(f'E:/radio/data/common/total/nii/img/{i:02g}.nii.gz')
    # extract_segmented_region(image_nii, chambers_nii, mask_nii, i)
    extract_cube(image_nii, chambers_nii, mask_nii, i)


def stretch_histogram(data, new_min=-50, new_max=50):
    # Normalize data to range [0, 1]
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data)) 
    # data_normalized = np.where(data_normalized <=200, data_normalized*0.02, data_normalized)

    # Scale the data to the desired range [new_min, new_max]
    data_stretched = (new_max - new_min) * data_normalized + new_min 
    
    # Clip values to ensure they are within the desired range
    data_stretched = np.clip(data_stretched, new_min, new_max)
    # rgb_image = np.stack((data_stretched, data_stretched, data_stretched), axis=-1)

    return data_stretched


def get_bounding_box(segmentation_data):
    cond = np.logical_or(segmentation_data == 1, segmentation_data == 2)
    seg_data = np.where(cond)

    min_x, max_x = np.min(seg_data[0]), np.max(seg_data[0])
    min_y, max_y = np.min(seg_data[1]), np.max(seg_data[1])
    min_z, max_z = np.min(seg_data[2]), np.max(seg_data[2])

   
    # min_x, max_x = int(min_x) , int(max_x) -20
    # min_y, max_y = int(min_y) +30, int(max_y) +20
    # min_z, max_z = int(min_z) - 10, int(max_z)
    margin = 10
    min_x, max_x = int(min_x) - margin, int(max_x) +margin
    min_y, max_y = int(min_y) - margin, int(max_y) +margin
    min_z, max_z = int(min_z) - margin, int(max_z) +margin

    # min_x, max_x = int(min_x) , int(max_x - 0.1 * min_x)
    # min_y, max_y = int(min_y + 50), int(max_y+ 50)#0.2 * min_y)
    # min_z, max_z = int(0.5 * min_z), int(max_z + 0.5 * min_z)
    # print((min_x, max_x), (min_y, max_y), (min_z, max_z))
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def adjust_contrast(image_data, contrast_factor):
    """
    Adjusts the contrast of an image data array.
    """
    # Rescale intensity values to range [0, 1]
    image_data_rescaled = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    # Adjust contrast
    image_data_adjusted = exposure.adjust_gamma(image_data_rescaled, contrast_factor)

    # Rescale intensity values back to original range
    image_data_adjusted = image_data_adjusted * (image_data.max() - image_data.min()) + image_data.min()

    return image_data_adjusted

def adjust_hu_window(image_data, center=-120, width=180):
    """
    Adjusts the window of Hounsfield Unit (HU) values in the image data.
    
    Parameters:
    image_data (numpy.ndarray): The input image data as a numpy array.
    center (float): The center value of the HU window.
    width (float): The width of the HU window.
    
    Returns:
    numpy.ndarray: The adjusted image data.
    """
    min_value = center - (width / 2)
    max_value = center + (width / 2)
    
    # Clip the values outside the desired window
    adjusted_data = np.clip(image_data, min_value, max_value)
    
    # Normalize the values to the range [0, 1]
    # adjusted_data = (adjusted_data - min_value) / (max_value - min_value)
    
    return adjusted_data


def extract_cube(img_file, chambers_file, LAD_file, i):
    img_data = flip(img_file.get_fdata())
    lad_data = LAD_file.get_fdata()
    chambers_data = chambers_file.get_fdata()
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bounding_box(chambers_data)
    if min_z < 0: 
        img = np.where(img_data > -500 )
        min_z = np.min(img[2])

    segmented_region = img_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    ch_mask = chambers_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    lad_mask = lad_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]
    bone_mask = segmented_region > 50
    # segmented_region = img_data
    # bone_mask = (segmented_region >= 200) & (segmented_region <= 3000)

    # Remove the identified white regions
    segmented_region[bone_mask] = 0

    original_size = segmented_region.shape
    new_shape = (original_size[0],128,128)
    print(original_size)
    zoom_factors = [new_shape[i] / original_size[i] for i in range(len(original_size))]
    segmented_region = zoom(segmented_region, zoom_factors, order=0)  # Use order=3 for cubic interpolation
    lad_mask = zoom(lad_mask, zoom_factors, order=0)
    ch_mask = zoom(ch_mask, zoom_factors, order=0)
    # 0 - nearest 
    # 1 -linear


    # blood_mask = (segmented_region >= 80) & (segmented_region <= 90)
    # segmented_region = stretch_histogram(segmented_region)
  
    # Apply a linear transformation to increase contrast
    # mean_intensity = np.mean(segmented_region)
    # enhanced_data = 4 * (segmented_region - mean_intensity) + mean_intensity

    # # Ensure the new data is within the original data range
    # segmented_region = np.clip(enhanced_data, segmented_region.min(), segmented_region.max())
    
    # blood_data = np.zeros_like(segmented_region)
    # blood_data[blood_mask] = segmented_region[blood_mask]


    segmented_region = adjust_contrast(segmented_region, 5.5)
    l, t, w, h = cropping_bbox(segmented_region)
    new_affine(i, h, w)


    segmented_region = remove_all_expect_heart(segmented_region, ch_mask, img_file)

    # save_subject_nii(img_data, f'E:/radio/data/common/total/crop/img/', i)
    # save_nifti_file(segmented_region, img_file.affine, f'E:/radio/data/common/total/crop/img/{i:02g}')
    # save_nifti_file(lad_data, img_file.affine, f'E:/radio/data/common/total/crop/LAD/{i:02g}')



    cube_nii = nib.Nifti1Image(segmented_region, img_file.affine)
    nib.save(cube_nii, f'E:/radio/data/common/total/crop/img/{i:02g}.nii.gz')
    seg_nii = nib.Nifti1Image(lad_mask, img_file.affine)
    nib.save(seg_nii, f'E:/radio/data/common/total/crop/LAD/{i:02g}.nii.gz')
    cham_nii = nib.Nifti1Image(ch_mask, img_file.affine)
    # nib.save(cham_nii, f'E:/radio/data/common/total/crop/chambers/ch{i:02g}.nii.gz')

    # save_subject_np(ch_mask, f'E:/radio/data/common/total/crop/np/chambers' ,i)
    # save_subject_np(segmented_region, f'E:/radio/data/common/total/crop/np/img' ,i)
    # save_subject_np(lad_mask, f'E:/radio/data/common/total/crop/np/LAD' ,i)

    # cube_nii = nib.Nifti1Image(segmented_region, img_file.affine)
    # nib.save(cube_nii, f'E:/radio/data/common/total/crop/img_{i:02g}.nii')
    # seg_nii = nib.Nifti1Image(lad_mask, img_file.affine)
    # nib.s ave(seg_nii, f'E:/radio/data/common/total/crop/lad_{i:02g}.nii')
        



def remove_all_expect_heart(image_data, mask_data, img_file):
    # Define the number of pixels to extend the segmentation region
    extension_pixels = 20

    # Create a binary mask indicating the segmented region
    segmented_mask = mask_data > 0

    # Dilate the segmented mask to extend the segmentation region
    extended_mask = binary_dilation(segmented_mask, iterations=extension_pixels)

    # Apply the extended mask to the original segmentation data
    extended_seg_data = np.where(extended_mask, 1, 0)

    segmented_img_data = image_data * (extended_seg_data > 0)
    segmented_img_data[segmented_img_data == 0] = -1000

    return segmented_img_data

 
def normalize():
    all = []
    for i in range(62):
      img = np.stack([np.load(p)['arr_0'] for p in glob.glob(f'E:/radio/data/common/total/crop/np/img/{i:02g}_*.npz')])
      all.append(img)
    #   save_subject_nii(img, f'E:/radio/data/common/total/crop/np/img_nor' ,i)

    all_values = np.concatenate(all)
    min = np.min(all_values)
    max = np.max(all_values)

    # normalized_arrays = [-1, 1]
    for i in range(len(all)):
        # normalized_array = 2 * ((all[i] - min) / (max - min)) - 1
        if max - min == 0:
            normalized_data = np.zeros_like(all[i])
        else:
            normalized_data =  (all[i] - min) / (max - min)
        # normalized_arrays.append(normalized_array)
        # print(normalized_data.min(), normalized_data.max())
        save_subject_np(normalized_data, f'E:/radio/data/common/total/crop/np/img' ,i)
        # save_subject_nii(normalized_array, f'E:/radio/data/common/total/crop/np/img_nor' ,i)

def new_affine(i, h, w):
    affine = np.load(f'E:/radio/data/common/raw/affine/{i:02g}.npz')['arr_0']
    affine[0] /= 512 / w
    affine[1] /= 512 / h
    savez_np(f'E:/radio/data/common//total/crop/affine_c10/{i:02g}.npz', affine)
