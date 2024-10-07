import numpy as np
from tqdm import trange
from utils import get_affine, savez_np
DICOM_NUM = 68


def extract_affine():
    for i in trange(DICOM_NUM):
        affine = get_affine(i)
        affine = np.asarray(affine, dtype=np.float32)
        savez_np(f'E:/radio/data/common/raw/affine/{i:02}.npz', affine)
