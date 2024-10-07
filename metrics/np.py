import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from sklearn.metrics import precision_score, recall_score
import torch
import monai
import nibabel
from monai.metrics import ConfusionMatrixMetric
from torchmetrics.functional import dice

import pydicom

# from atlas.post_greedy import _dice
from dataset.datasets import DataSplit, split_subjects
from metrics.hd import hd_score, msd_score, volume_ratio
from utils import get_affine_cropped, load_from_nii, prepare_mask_for_metrics
# def compute_hausdorff_distance(array1, array2):
#     """
#     Compute the Hausdorff distance between two binary masks.
    
#     Parameters:
#         array1 (np.ndarray): First binary mask array.
#         array2 (np.ndarray): Second binary mask array.
    
#     Returns:
#         float: The Hausdorff distance.
#     """
#     # Get the coordinates of the non-zero elements in each array
#     coords1 = np.column_stack(np.nonzero(array1))
#     coords2 = np.column_stack(np.nonzero(array2))
    
#     # Compute the directed Hausdorff distances
#     d_forward = directed_hausdorff(coords1, coords2)[0]
#     d_backward = directed_hausdorff(coords2, coords1)[0]
    
#     # The Hausdorff distance is the maximum of the two directed distances
#     hausdorff_distance = max(d_forward, d_backward)
    
#     return hausdorff_distance

# import numpy as np

# def compute_volume_ratio(mask1, mask2):
#     """
#     Compute the volume ratio between two binary masks.
    
#     Parameters:
#         mask1 (np.ndarray): First binary mask array.
#         mask2 (np.ndarray): Second binary mask array.
    
#     Returns:
#         float: The volume ratio of mask1 to mask2.
#     """
#     # Ensure masks are binary
#     mask1 = mask1 > 0
#     mask2 = mask2 > 0
    
#     # Compute volumes
#     volume1 = np.sum(mask1)
#     volume2 = np.sum(mask2)
    
#     if volume2 == 0:
#         raise ValueError("The volume of the second mask is zero, cannot compute volume ratio.")
    
#     # Compute volume ratio
#     volume_ratio = volume1 / volume2
    
#     return volume_ratio

def compute_volume_ratio(preds_labels, true_labels):
    ratios = []
    for p, l in zip(preds_labels, true_labels):
        sum_p = np.sum(p)
        sum_l = np.sum(l)
        if sum_l == 0:
            # Handle the case where true labels sum to zero
            ratio = np.nan  # or use 0, or any other value that makes sense in your context
        else:
            ratio = sum_p / sum_l
        ratios.append(ratio)
    
    vr = np.nanmean(ratios)
    return vr

def compute_metrics_mm(path, lad=True):

    if lad:
        subjects = range(56,62)
        total = 'total'
    else:
        subjects = range(59,62)
        total = 'total_breast'

    score_hd_list = []
    score_msd_list = []
    score_vr_list = []
    precisions = []
    recalls = []
    dsc = []

    for i in subjects:
 
        ground_truth_array = nibabel.load(f'{path}/nii/true/{i}.nii.gz').get_fdata()
        prediction_array = nibabel.load(f'{path}/nii/fusion/{i}.nii.gz').get_fdata()

        # affine = get_affine_cropped(i)
        affine = get_affine_cropped(total, i)
        target = ground_truth_array
        preds = prediction_array
        score_hd_list.append(hd_score(preds, target, affine, 1))
        score_msd_list.append(msd_score(preds, target, affine, 1))
        score_vr_list.append(compute_volume_ratio(preds, target))
        dsc.append(_dice(preds.copy(), target.astype(np.int32)[0])
)
        ground_truth_flat = target.flatten()
        predicted_mask_flat = preds.flatten()
        precisions.append(precision_score(ground_truth_flat, predicted_mask_flat))
        recalls.append(recall_score(ground_truth_flat, predicted_mask_flat))

    df = pd.DataFrame({
                        'subject': subjects,
                        'dsc' : dsc,
                        'hd [mm]': score_hd_list,
                        'vr': score_vr_list,
                        'msd [mm]': score_msd_list,
                        'ppv': precisions,
                        'tpr' : recalls,


    })
    print(df)
    df.to_csv(f'{path}/all_sipinski_metrics.csv', index=False)

    # return hd_distance

def compute_metrics_atlases(path, stage):
    # train_subjects, val_subjects, test_subjects = split_subjects(DataSplit)
    # val_subjects.extend(test_subjects)
    test_subjects = range(56,62)
    train_subjects = range(0,50)
    score_hd_list = []
    score_msd_list = []
    score_vr_list = []
    precisions = []
    recalls = []

    tab = []
    for test_i in test_subjects:
        # true_labels = load_from_nii(f'E:/radio/data/FIRST/unet_LAD_cube_resnet50/nii/true/{test_i:02g}.nii.gz')
        # true_labels = load_from_nii(f'E:/radio/data/common/cropped_XY/nii/LAD/{test_i:02g}.nii.gz')
        true_labels = load_from_nii(f'{path}/{test_i:02g}.nii.gz')
        preds_list = []
        # true_labels = load_from_nii(
        #         f'E:/radio/data/{data_split.name}/unet_LAD_all/nii/true/{test_i:02g}.nii.gz')
        
        for train_i in train_subjects:
            preds_labels = load_from_nii(
                f'E:/radio/data/FIRST/atlas_{stage}/nii/preds/{train_i:02g}_{test_i:02g}.nii.gz')
            preds_list.append(preds_labels)
           
            affine = get_affine_cropped('total', test_i)
            target = true_labels
            preds = preds_labels
            score_hd_list.append(hd_score(preds, target, affine, 1))
            score_msd_list.append(msd_score(preds, target, affine, 1))
            score_vr_list.append(compute_volume_ratio(preds, target))

            ground_truth_flat = target.flatten()
            predicted_mask_flat = preds.flatten().astype(int)
            precisions.append(precision_score(ground_truth_flat, predicted_mask_flat))
            recalls.append(recall_score(ground_truth_flat, predicted_mask_flat))
            dice = [_dice(preds_labels.copy(), true_labels.astype(np.int32))]
            dice = dice[0]

            tab.append({
                'subject': test_i,
                'train':train_i,
                'dsc' : dice, 
                'hd [mm]': hd_score(preds, target, affine, 1),
                'vr': compute_volume_ratio(preds, target),
                'msd [mm]': msd_score(preds, target, affine, 1),
                'ppv': precision_score(ground_truth_flat, predicted_mask_flat),
                'tpr' : recall_score(ground_truth_flat, predicted_mask_flat),
                })

        dff = pd.DataFrame(tab)  

    # print(df)
    # df.to_csv(f'E:/radio/data/{data_split.name}/atlas_{stage}/scores.csv', index=False)
    dff.to_csv(f'E:/radio/data/FIRST/atlas_{stage}/all_mm.csv', index=False)
        # t.to_csv(f'E:/radio/data/{data_split.name}/atlas_{stage}/metrics.csv', index=False)


def compute_average_surface_distance(ground_truth_path, prediction_path):
    ground_truth_image = sitk.ReadImage(ground_truth_path)
    prediction_image = sitk.ReadImage(prediction_path)

    # Pobierz tablice wokseli
    ground_truth_array = sitk.GetArrayFromImage(ground_truth_image)
    prediction_array = sitk.GetArrayFromImage(prediction_image)

    # Pobierz rozmiar wokseli (np. w milimetrach)
    voxel_spacing = ground_truth_image.GetSpacing()


    # Konwertuj tablice do tensora PyTorch
    ground_truth_tensor = torch.tensor(ground_truth_array, dtype=torch.float32).unsqueeze(0)
    prediction_tensor = torch.tensor(prediction_array, dtype=torch.float32).unsqueeze(0)

    # Oblicz odległość Hausdorffa, uwzględniając rozmiar wokseli
    hd_distance = monai.metrics.compute_average_surface_distance(
        y_pred=prediction_tensor,
        y=ground_truth_tensor,
        include_background=False,
        spacing=voxel_spacing
    )
    return hd_distance


def compute_all_metrics_mm(path, lad):
    if lad:
        subjects = range(56,62)
    else:
        subjects = range(30,33)

    score_hd_list = []
    score_msd_list = []
    score_vr_list = []
    dsc = []
    for i in subjects:
 
        ground_truth_array = nibabel.load(f'{path}/nii/true/{i}.nii.gz').get_fdata()
        prediction_array = nibabel.load(f'{path}/nii/preds/{i}.nii.gz').get_fdata()
        preds = prepare_mask_for_metrics(ground_truth_array)
        true = prepare_mask_for_metrics(prediction_array)
            # affine = get_affine_cropped_lad(i)
        files = glob.glob(f'E:/radio/data/dicom/{i + 1}/CT*')
        files = [pydicom.dcmread(file, force=True) for file in files]
        scale_x, scale_y = tuple(files[0].PixelSpacing)[::-1]

        # scale = tuple(files[0].PixelSpacing)[::-1]
        scale_z = files[0].SliceThickness
        scale = np.asarray([scale_x, scale_y, scale_z], dtype=float)
        print(scale)

        dsc.append(monai.metrics.compute_dice(preds, true).item())
        score_hd_list.append(monai.metrics.compute_hausdorff_distance(preds, true, spacing=scale).item())
        score_msd_list.append(monai.metrics.compute_average_surface_distance(preds, true, spacing=scale).item())
        score_vr_list.append(compute_volume_ratio(prediction_array, ground_truth_array))

    df = pd.DataFrame({
                        'subject': subjects,
                        'dsc' : dsc,
                        'hd [mm]': score_hd_list,
                        'vr': score_vr_list,
                        'msd [mm]': score_msd_list,

    })
    print(df)
    df.to_csv(f'{path}/all_metrics.csv', index=False)



    # ground_truth_image = sitk.ReadImage(ground_truth_path)
    # prediction_image = sitk.ReadImage(prediction_path)
    # pred_nii = nib.load(f'E:/radio/data/common/total/nii/img/{i:02g}.nii')
    # true_nii = nib.load(f'E:/radio/data/common/total/nii/img/{i:02g}.nii')

    # # Pobierz tablice wokseli
    # ground_truth_array = sitk.GetArrayFromImage(ground_truth_image)
    # prediction_array = sitk.GetArrayFromImage(prediction_image)

    # # Pobierz rozmiar wokseli (np. w milimetrach)
    # voxel_spacing = ground_truth_image.GetSpacing()
    

    # # Konwertuj tablice do tensora PyTorch
    # ground_truth_tensor = torch.tensor(ground_truth_array, dtype=torch.float32).unsqueeze(0)
    # prediction_tensor = torch.tensor(prediction_array, dtype=torch.float32).unsqueeze(0)

    # # Oblicz odległość Hausdorffa, uwzględniając rozmiar wokseli
    # hd_distance = monai.metrics.compute_average_surface_distance(
    #     y_pred=prediction_tensor,
    #     y=ground_truth_tensor,
    #     include_background=False,
    #     spacing=voxel_spacing
    # )
    # return hd_distance

# def compute_ppv_ptr(path):
   
#    ground_truth_nii = nib.load(ground_truth_path)
# predicted_mask_nii = nib.load(predicted_mask_path)

# ground_truth = ground_truth_nii.get_fdata().astype(np.uint8)
# predicted_mask = predicted_mask_nii.get_fdata().astype(np.uint8)

def _dice(preds, target):
    return dice(
        torch.tensor(preds),
        torch.tensor(target.copy()),
        ignore_index=0,
    ).numpy()
