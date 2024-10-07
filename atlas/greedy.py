import subprocess
import sys
from itertools import product

from tqdm import tqdm

from dataset.datasets import DataSplit, split_subjects
from utils import makedirs


def greedy(args):
    with subprocess.Popen(['C:/Program Files/ITK-SNAP 4.0/bin/greedy.exe', *args],
                          # stdout=sys.stdout, stderr=sys.stderr
                          stdout=subprocess.DEVNULL, stderr=sys.stderr
                          ):
        pass


def register_moments(fixed, moving, out, d=3):
    greedy([
        '-moments', '1',
        '-d', str(d),
        '-i', fixed, moving,
        '-o', out,
    ])


def register_affine(fixed, moving, out, moments, mask_fixed=None, mask_moving=None, d=3):
    greedy([
        '-a',
        '-d', str(d),
        '-ia', moments,
        '-i', fixed, moving,
        '-o', out,
        '-m', 'WNCC', '2x2x2',
        '-n', '100x50x10',
        *(['-gm', mask_fixed] if mask_fixed is not None else []),
        *(['-mm', mask_moving] if mask_moving is not None else []),
    ])


def register_deformable(fixed, moving, out, pre=None, mask_fixed=None, mask_moving=None, d=3):
    greedy([
        '-d', str(d),
        '-i', fixed, moving,
        # '-e', '5', 
        # '-s', ''
        *(['-it', pre] if pre is not None else []),
        '-o', out,
        '-m', 'SSD',
        '-n', '100x50x0',
        *(['-gm', mask_fixed] if mask_fixed is not None else []),
        *(['-mm', mask_moving] if mask_moving is not None else []),
    ])


def transform(fixed, moving, label, transformations, out_img, out_pred, d=3):
    greedy([
        '-d', str(d),
        '-rf', fixed,
        '-rm', moving, out_img,
        '-ri', 'LABEL', '0.2vox',
        '-rm', label, out_pred,
        '-r', *transformations,
    ])


def register_transform(data_split: DataSplit, stage: str, labels: str, begin=0):
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation')
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/preds')
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/img')
    train_subjects, test_subjects = split_subjects(data_split)
    for test_i, train_i in tqdm(list(product(test_subjects, train_subjects))[begin:]):
        register_moments(
            fixed=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{test_i:02g}.nii.gz',
            moving=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{train_i:02g}.nii.gz',
            out=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt'
        )
        register_deformable(
            # fixed=f'E:/radio/data/common/cropped_XY/nii/img/{test_i:02g}.nii.gz',
            # moving=f'E:/radio/data/common/cropped_XY/nii/img/{train_i:02g}.nii.gz',
            fixed=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{test_i:02g}.nii.gz',
            moving=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{train_i:02g}.nii.gz',
            out=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/d_{train_i:02g}_{test_i:02g}.nii.gz',
            pre=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt',
        )
        transform(
            fixed=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{test_i:02g}.nii.gz',
            moving=f'E:/radio/data/{data_split.name}/fcn_{stage}/nii/img/{train_i:02g}.nii.gz',
            label=f'E:/radio/data/common/cropped_XY/nii/{labels}/{train_i:02g}.nii.gz',
            transformations=[
                f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/d_{train_i:02g}_{test_i:02g}.nii.gz',
                # f'E:/radio/data/atlas/transformation/a_{train_i:02g}_{test_i:02g}.txt',
                f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt',
            ],
            out_pred=f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/preds/{train_i:02g}_{test_i:02g}.nii.gz',
            out_img=f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/img/{train_i:02g}_{test_i:02g}.nii.gz'
        )

def affine(fixed, moving, out, mask_fixed=None, mask_moving=None, d=3):
    greedy([
        '-a',
        '-d', str(d),
        '-m', 'NCC', '2x2x2',
        '-i', fixed, moving,
        '-o', out,
        # '-ia-image-centers ',
        '-n', '100x50x10',
    ])


def test():
        path = 'C:\\Users\\Pc\\Documents\\All_in_one\\mgr\\greedy_test\\affine\\'
        affine(
            fixed=f'E:/radio/data/common/cropped_XY/nii/img/61.nii.gz',
            moving=f'E:/radio/data/common/cropped_XY/nii/img/00.nii.gz',
            out=f'{path}\\affine.txt'
        )
        # register_moments(
        #     fixed=f'E:/radio/data/common/cropped_XY/nii/img/61.nii.gz',
        #     moving=f'E:/radio/data/common/cropped_XY/nii/img/00.nii.gz',
        #     # fixed=f'{path}\\IN.nii',
        #     # moving=f'{path}\\ATLAS.nii',
        #     out=f'{path}\\affine.txt'
        # )
        register_deformable(
            # fixed=f'E:/radio/data/common/cropped_XY/nii/img/{test_i:02g}.nii.gz',
            # moving=f'E:/radio/data/common/cropped_XY/nii/img/{train_i:02g}.nii.gz',
            fixed=f'E:/radio/data/common/cropped_XY/nii/img/61.nii.gz',
            moving=f'E:/radio/data/common/cropped_XY/nii/img/00.nii.gz',
            out=f'{path}\\deformable.nii',
            pre=f'{path}\\affine.txt'
        )
        transform(
            fixed=f'E:/radio/data/common/cropped_XY/nii/img/61.nii.gz',
            moving=f'E:/radio/data/common/cropped_XY/nii/img/00.nii.gz',
            label=f'E:/radio/data/common/cropped_XY/nii/LAD/00.nii.gz',
            transformations=[
               f'{path}\\deformable.nii',
                # f'E:/radio/data/atlas/transformation/a_{train_i:02g}_{test_i:02g}.txt',
               f'{path}\\affine.txt'
            ],
            out_pred=f'{path}\\out_lad.nii',
            out_img=f'{path}\\out_img.nii',
        )



def run_register(data_split: DataSplit, stage: str, path: str, ext: str, img: str, target: str, breast=False, begin=0, ):
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation')
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/preds')
    makedirs(f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/img')
    train_subjects, val_subjects, test_subjects = split_subjects(data_split, breast=breast)
    val_subjects.extend(test_subjects)
    test_subjects = val_subjects
    for test_i, train_i in tqdm(list(product(test_subjects, train_subjects))[begin:]):
        register_moments(
            fixed=f'{path}/{img}/{test_i:02g}.{ext}',
            moving=f'{path}/{img}/{train_i:02g}.{ext}',
            out=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt'
        )
        register_deformable(
            fixed=f'{path}/{img}/{test_i:02g}.{ext}',
            moving=f'{path}/{img}/{train_i:02g}.{ext}',
            out=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/d_{train_i:02g}_{test_i:02g}.nii.gz',
            pre=f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt',
        )
        transform(
            fixed=f'{path}/{img}/{test_i:02g}.{ext}',
            moving=f'{path}/{img}/{train_i:02g}.{ext}',
            label=f'{path}/{target}/{train_i:02g}.{ext}',
            transformations=[
                f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/d_{train_i:02g}_{test_i:02g}.nii.gz',
                f'E:/radio/data/{data_split.name}/atlas_{stage}/transformation/m_{train_i:02g}_{test_i:02g}.txt',
            ],
            out_pred=f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/preds/{train_i:02g}_{test_i:02g}.nii.gz',
            out_img=f'E:/radio/data/{data_split.name}/atlas_{stage}/nii/img/{train_i:02g}_{test_i:02g}.nii.gz'
        )


def register_transform_all_lad(data_split: DataSplit, stage: str, labels: str, begin=0):
    run_register(data_split=data_split, stage=stage, path='E:/radio/data/common/nii', ext='nii.gz', img='img', target='LAD')

def register_transform_all_ladxy(data_split: DataSplit, stage: str, labels: str, begin=0):
    run_register(data_split=data_split, stage=stage, path='E:/radio/data/common/cropped_xy/nii', ext='nii.gz', img='img', target='LAD')


def register_transform_crop_lad(data_split: DataSplit, stage: str, labels: str, begin=0):
    run_register(data_split=data_split, stage=stage, path='E:/radio/data/common/total/crop', ext='nii.gz', img='img', target='LAD')

def register_transform_crop_cancer_breast(data_split: DataSplit, stage: str, labels: str, begin=0):
    run_register(data_split=data_split, stage=stage, path='E:/radio/data/common/total_breast/crop', ext='nii', img='img_left', target='cancer_breast', breast=True)

def deform(fixed, moving, out):
    greedy([
       '-d', str(3),
       '-i', fixed, moving ,
       '-o', out,
       '-m', 'NCC', '4x4x4',
       '-n', '100x50x20',
       '-s', '1.5vox', '0.5vox',
       '-e', '0.5',
    ]) 


def reslice(fixed, moving_seg, transformations, out_pred):
    greedy({
        '-d', '3',
        '-rf', fixed,
        '-ri', 'LABEL', '0.2vox',
        '-rm', moving_seg, out_pred,
        '-r', transformations,
    })

