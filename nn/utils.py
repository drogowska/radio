from typing import Union

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.datamodules import AbstractDataModule
from dataset.datasets import DataSplit
from nn.FCNWithMask import FCNWithMask
from nn.FCNLAD import UnetPuls
from nn.SliceClassifier import SliceClassifier
import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

def make_trainer(data_split: DataSplit, clazz: type):

    trainer = Trainer(accelerator='gpu', 
                      gradient_clip_val=0.5 ,
                      max_epochs=20,
                      precision='16-mixed', 
                      # accumulate_grad_batches=1,
                    #   limit_val_batches=0,
                      logger=TensorBoardLogger('E:\\radio\\logs', name=f'{data_split.name}_{clazz.__name__}'),
                      # fast_dev_run=3,
                      # check_val_every_n_epoch=1,
                      # detect_anomaly=True
                      # enable_checkpointing=True,
                       # num_sanity_val_steps=1,
                      log_every_n_steps=1,
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
                      )
    return trainer


def train_model(model: Union[FCNWithMask, UnetPuls, SliceClassifier], dm: AbstractDataModule):
    trainer = make_trainer(dm.data_split, type(model))
    trainer.fit(model=model, datamodule=dm)