import glob

import numpy as np
from atlas.greedy import register_transform, register_transform_crop_cancer_breast, register_transform_crop_lad, register_transform_all_lad, register_transform_all_ladxy
from atlas.post_greedy import post_greedy
from dataset.datamodules import HeartSegmentationWithMaskDataModule, \
    LADSegmentationWithMaskDataModule, ChambersSegmentationWithMaskDataModule, LADCubeSegmentationModule, LADCubeWithChambersSegmentationModule, \
    CancerBedCubeSegmentationModule, CancerBreastCubeSegmentationModule, \
    CancerBreastSegmentationModule, LADSegmentationModule
from dataset.datasets import DataSplit, SliceDataset
from extracting.affine import extract_affine
from extracting.crop_xy import crop_xy_all
from extracting.from_dcm_to_np import from_dcm_to_np_all
from nn import runs_classification
from nn.FCNWithMask import HeartFCNWithMask, LADFCNWithMask, ChambersFCNWithMask
from nn.runs_segmentation import runs_segmentation
import torch
from extracting.total_LAD import run_total
from extracting.total_ribs import run_total_ribs
from extracting.crop_lad import crop_lad, normalize
DICOM_NUM = 35
from nn.FCNLAD import LADAlone, LADwithChambers, CancerBreast, LADwithHeart
# from atlas.atlas import run
from extracting.prep_nii import run_prep
from extracting.crop_breast import crop_breasts
from nn.uu import uu
from dataset.segmentationDataModule import LADcubeSegmentationDM, BreastCancerCubeSegmentationDM
from nn.UnetModel import UnetLADCube, UnetBreastCancerCube, UnetLADAll, UnetCancerBedCube, UnetCancerBedAll
from nn.combined import combined
from dataset.segmentationDataModule import AtlasLADcubeSegmentationDM, LADSegmentationDM, AtlasLADSegmentationDM, CancerBedCubeSegmentationDM, CancerBedSegmentationDM
import warnings


def main_sipinski():
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument.*")

    extract_affine()
    from_dcm_to_np_all()
    crop_xy_all()
    runs_classification.run()

    for data_split in [
        DataSplit.FIRST,
        DataSplit.SECOND,
    ]:
        runs_segmentation(data_split, HeartFCNWithMask(), HeartSegmentationWithMaskDataModule(data_split))
        register_transform(data_split, 'heart', 'heart')
        post_greedy(data_split, 'heart', 'heart')

        runs_segmentation(data_split, ChambersFCNWithMask(), ChambersSegmentationWithMaskDataModule(data_split))
        register_transform(data_split, 'chambers', 'chambers')
        post_greedy(data_split, 'chambers', 'chambers')

        runs_segmentation(data_split, LADFCNWithMask(), LADSegmentationWithMaskDataModule(data_split))
        register_transform(data_split, 'LAD', 'LAD')
        post_greedy(data_split, 'LAD', 'LAD')

def main():
    warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated.*")
    warnings.filterwarnings("ignore", ".*Consider increasing  the value of the `num_workers` argument.*")
    warnings.filterwarnings("ignore", message=" Detected old nnU-Net plans format. Attempting to reconstruct network architecture parameters.*")
    torch.set_float32_matmul_precision('high')  
    extract_affine()
    from_dcm_to_np_all()
    crop_xy_all() 
    run_prep()

    run_total()
    crop_lad()

    run_total_ribs()
    crop_breasts()
    
    # ---------------------- LAD -----------------------------
    runs_segmentation(DataSplit.FIRST, UnetLADCube(), LADcubeSegmentationDM())
    register_transform_crop_lad(DataSplit.FIRST, 'crop_LAD', 'LAD')
    combined(UnetLADCube(), AtlasLADcubeSegmentationDM(), "crop_LAD")


    runs_segmentation(DataSplit.FIRST, UnetLADAll(), LADSegmentationDM())
    register_transform_all_lad(DataSplit.FIRST, 'all_LAD', 'LAD')
    combined(UnetLADAll(), AtlasLADSegmentationDM(), "all_LAD")

# ---------------------- cancer bed -----------------------------
    runs_segmentation(DataSplit.FIRST, UnetCancerBedCube(), CancerBedCubeSegmentationDM())
    runs_segmentation(DataSplit.FIRST, UnetCancerBedAll(), CancerBedSegmentationDM())

    post_greedy(DataSplit.FIRST, 'crop_LAD', 'LAD', path='E:/radio/data/common/total/crop', ext='nii.gz')
    post_greedy(DataSplit.FIRST, 'crop_LAD', 'LAD', path='E:/radio/data/common/total/crop', ext='nii.gz')
  

if __name__ == '__main__':
    main()
