import pandas as pd
import torch
from torchmetrics.functional.classification import multiclass_accuracy

from dataset.datamodules import AbstractDataModule, HeartSliceClassificationDataModule
from dataset.datasets import DataSplit
from nn.SliceClassifier import SliceClassifier
from nn.utils import make_trainer, train_model
from utils import savez_np
import pathlib

def run():
    for data_split in [
        DataSplit.FIRST,
        DataSplit.SECOND,
    ]:
        model = SliceClassifier()
        dm = HeartSliceClassificationDataModule(data_split)    
        path = F'data\models\{data_split.name}_SliceClassifier.pth'

        if not pathlib.Path(path).is_file(): 
            train_model(model, dm)
            torch.save(model.state_dict(), path)
        else: 
            print("Skipped training of SliceClassifier...")
            model.load_state_dict(torch.load(path))

        predict_model(model, dm)


def predict_model(model: SliceClassifier, dm: AbstractDataModule):
    trainer = make_trainer(dm.data_split, type(model))
    df = []
    dl = dm.dataloader_for_subjects(dm.train_subjects + dm.test_subjects)
    preds = trainer.predict(model=model, dataloaders=dl)
    preds = torch.concat(preds)

    target = torch.concat([b[-1] for b in dl])

    acc = multiclass_accuracy(preds, target, num_classes=2, average='micro').item()
    df.append(dict(
        accuracy=acc,
    ))
    print(f'ACC: {acc}')
    savez_np(f'E:/radio/data/{dm.data_split.name}/{model.stage}/mask.npz', preds.numpy())

    pd.DataFrame(df).to_csv(f'E:/radio/data/{dm.data_split.name}/{model.stage}/dice.csv')
