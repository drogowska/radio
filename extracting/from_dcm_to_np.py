import glob

import numpy as np
import pydicom
from skimage.draw import polygon2mask
from tqdm.contrib.concurrent import process_map

from utils import savez_np

# DICOM_BASE = '../main/E:/radio/data/dicom'
DICOM_BASE = 'E:/radio/data/dicom'

DICOM_NUM_v1 = 29 + 5
DICOM_NUM_v2 = 33
 

def from_dcm_to_np_all():
    # process_map(from_dcm_to_np, range(DICOM_NUM), max_workers=12)
    # for i in range(35):
    #     from_dcm_to_np(i, DICOM_BASE)
    for i in range(68):
        from_dcm_to_np(i)
    # from_dcm_to_np(0)


def from_dcm_to_np(i):
    # scans
    # scans
    # scans
    files = glob.glob(f'{DICOM_BASE}/{i + 1}/CT*')
    files = [pydicom.dcmread(file, force=True) for file in files]
    files.sort(key=lambda file: file.InstanceNumber)

    assert all(ct.Rows == 512 for ct in files)
    assert all(ct.Columns == 512 for ct in files)
    assert all(ct.ImageOrientationPatient == [1, 0, 0, 0, 1, 0] for ct in files)

    assert 1 == len(np.unique([ct.RescaleSlope for ct in files], axis=0))
    assert 1 == len(np.unique([ct.RescaleIntercept for ct in files], axis=0))

    assert 1 == len(np.unique([ct.SliceThickness for ct in files], axis=0))
    assert 1 == len(np.unique([tuple(ct.PixelSpacing) for ct in files], axis=0))

    img_3d = []
    if i < 29 or i > 61 :
        # img_path = 'E:/radio/data/common/raw/np/img/'
        labels_name_sets = [
        ('heart', ['serce cale'] if i < 30 or i==62  else ['Heart']),
        ('chambers',
         ['Lprzedsionek', 'Lkomora', 'Pprzedsionek', 'Pkomora']
         if i < 30 or i==62  else ['Atrium_L', 'Ventricle_L', 'Atrium_R', 'Ventricle_R']),
        ('LAD', ['LAD']),
         ]
    if i > 28 and i < 62:
        labels_name_sets = [
        ('heart', ['Serce']), 
        ('second breast',['Druga piers']),
        ('CTV1', ['CTV1']),
        ('CTV2', ['CTV2']),
        ('LAD', ['LAD']),
    ]

    for layer_i, ct in enumerate(files):
        layer = np.asarray(ct.pixel_array) * ct.RescaleSlope + ct.RescaleIntercept 
        # if i < 29 or i > 61:
        layer = np.clip(layer, -1_000, 1_000) / 2000 + 0.5
        layer = np.asarray(layer, dtype=float)
        savez_np(f'E:/radio/data/common/raw/np/img/{i:02g}_{layer_i:04g}.npz', layer)
        img_3d.append(layer)
    img_3d = np.stack(img_3d)
    # savez_np(f'E:/radio/data/np/3d/{test}/img/{i:02g}.npz', img_3d)

    # rois
    # rois
    # rois
    scale_x, scale_y = tuple(files[0].PixelSpacing)[::-1]
    # scale = tuple(files[0].PixelSpacing)[::-1]
    scale_z = files[0].SliceThickness
    scale = np.asarray([scale_x, scale_y, scale_z], dtype=float)
    offset = np.asarray(tuple(files[0].ImagePositionPatient), dtype=float)
    rs_file = glob.glob(f'{DICOM_BASE}/{i + 1}/RS*')
    assert 1 == len(rs_file)

    file = rs_file[0]
    file = pydicom.dcmread(file, force=True)

    for name, labels in labels_name_sets:
        roi_name_to_id = {roi.ROIName: roi.ROINumber for roi in file.StructureSetROISequence
                          if roi.ROIName in labels}
        roi_id_to_contour = {roi.ReferencedROINumber: roi.ContourSequence for roi in file.ROIContourSequence
                             if roi.ReferencedROINumber in roi_name_to_id.values()}
        # print(file.StructureSetROISequence)
        def paint_contours(contours):
            mask_3d = np.zeros_like(img_3d, dtype=bool)
            for contour in contours:
                assert contour.ContourGeometricType == 'CLOSED_PLANAR'
                contour = np.asarray(contour.ContourData).reshape((-1, 3))
                contour2 = (contour - offset) / scale
                contour2 *= [1, 1, -1]
                contour2 = np.round(contour2)
                contour2 = np.asarray(contour2, dtype=np.uint32)
                mask = polygon2mask(img_3d.shape[1:], contour2[:, :2:][:, ::-1])
                mask_3d[contour2[0, -1]] = np.logical_or(mask_3d[contour2[0, -1]], mask)
            return mask_3d

        masks = combine_masks([paint_contours(
            roi_id_to_contour[roi_name_to_id[label]]
        ) for label in labels])
        # savez_np(f'E:/radio/data/np/3d/{test}/mask/{i:02g}.npz', masks)
        for layer_i in range(masks.shape[0]):
            savez_np(f'E:/radio/data/common/raw/np/{name}/{i:02g}_{layer_i:04g}.npz', masks[layer_i])


def combine_masks(masks: list[np.ndarray]):
    result = masks[0] * 1
    for i, mask in enumerate(masks[1:], start=2):
        result = np.where(mask, i, result)
    return result

