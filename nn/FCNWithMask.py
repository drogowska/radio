import glob

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torchvision.models.segmentation import fcn_resnet101

from dataset.datasets import DataSplit


class FCNWithMask(LightningModule):
    @classmethod
    def load(cls, data_split: DataSplit, v_num: int):
        files = glob.glob(f'logs/{data_split.name}_{cls.__name__}/version_{v_num}/checkpoints/*.ckpt')
        if len(files) != 1:
            raise ValueError
        return cls.load_from_checkpoint(files[0])

    def __init__(self, channels, num_classes, stage, back_weight=0.1):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.stage = stage

        # self.model = nn.Sequential(
        #     FCNPre(channels),
        #     fcn_resnet101(pretrained_backbone=True, num_classes=num_classes))

        self.model = fcn_resnet101(weights_backbone='DEFAULT', num_classes=num_classes)
        if self.channels > 1:
            self.model.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.model = deeplabv3_resnet101(pretrained_backbone=True, num_classes=num_classes)
        # if self.channels > 1:
        #     self.model.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.loss = CrossEntropyLoss(
            weight=torch.Tensor(np.concatenate([
                [back_weight],
                np.ones(num_classes - 1) / (num_classes - 1) * (1 - back_weight)]))
            # weight=torch.ones(num_classes, dtype=torch.float) / num_classes
        )
        # self.loss = CrossEntropyLoss(
        #     weight=torch.full(size=(num_classes,), fill_value=1 / num_classes, dtype=torch.float)
        # )

    def configure_optimizers(self):
        # return torch.optim.SGD(self.model.parameters(), lr=0.01) # ain't it :/
        return torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.99)

    def forward(self, x):
        x, slice_mask = x
        x = self.expand_x(x)
        x = x.float()

        if torch.all(slice_mask):
            y_hat = self.model(x)['out']
            return y_hat

        y_hat = torch.concat([
            torch.full((x.shape[0], 1, x.shape[-2], x.shape[-1]), fill_value=100),
            torch.full((x.shape[0], self.num_classes - 1, x.shape[-2], x.shape[-1]), fill_value=-100)
        ], dim=1).to(x)
        # y_hat = torch.concat([
        #     torch.full((x.shape[0], 1, x.shape[-2], x.shape[-1]), fill_value=torch.inf),
        #     torch.full((x.shape[0], self.num_classes - 1, x.shape[-2], x.shape[-1]), fill_value=-torch.inf)
        # ], dim=1).to(x)
        if torch.any(slice_mask):
            model_output = self.model(x[slice_mask])['out']
            y_hat = y_hat.to(model_output)
            y_hat[slice_mask] = model_output
        return y_hat

    def expand_x(self, x):
        if len(x.shape) == 3:
            x = x[:, None]
        if self.channels == 1:
            x = x.expand(-1, 3, -1, -1)
        return x

    def shared_eval_step(self, phase, batch, *args, **kwargs):
        x, slice_mask, y = batch

        # x = self.expand_x(x)
        # x = x[slice_mask]
        # y = y[slice_mask]
        # y_hat = self.model(x)

        y_hat = self((x, slice_mask))
        # pred_label = torch.argmax(y_hat, dim=1)

        y = y.long()
        loss = self.loss(y_hat, y)
        # loss = dice_loss(y_hat, y)

        # loss = wce(y_hat, y)
        self.log(f'{phase}_loss', loss, on_step=True, on_epoch=True, logger=True)
        return dict(
            loss=loss,
            # _dice=torch.stack(
            #     [torchmetrics.functional.dice(sample[0], sample[1], ignore_index=0, num_classes=self.num_classes)
            #      for sample in zip(torch.unsqueeze(y_hat, 1), torch.unsqueeze(y, 1))]
            # ),
            # _dice_support=torch.sum(pred_label, dim=(-2, -1)) + torch.sum(y, dim=(-2, -1))
        )

    def training_step(self, *args, **kwargs):
        return self.shared_eval_step('train', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.shared_eval_step('test', *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[:-1]
        y_hat = self(x)
        return torch.argmax(y_hat, dim=1)

    # def training_epoch_end(self, outputs: list[dict]) -> None:
    #     weights = torch.concat([output['_dice_support'] for output in outputs])
    #     dice = torch.concat([output['_dice'] for output in outputs])
    #     dice = dice * weights / torch.sum(weights)
    #     self.log('train_dice', dice, on_epoch=True, prog_bar=True, logger=True)
    #
    # def test_epoch_end(self, outputs: list[dict]) -> None:
    #     weights = torch.concat([output['_dice_support'] for output in outputs])
    #     dice = torch.concat([output['_dice'] for output in outputs])
    #     dice = dice * weights / torch.sum(weights)
    #     self.log('test_dice', dice, on_epoch=True, prog_bar=True, logger=True)


def dice_loss(preds, target):
    num_classes = preds.shape[1]

    preds = torch.softmax(preds, dim=1)

    target = one_hot(target, num_classes=num_classes)
    target = torch.moveaxis(target, -1, 1)

    preds = preds[:, 1:]
    target = target[:, 1:]

    dims = (1, 2, 3)
    intersection = torch.sum(preds * target, dims)
    cardinality = torch.sum(preds + target, dims)

    dice_score = 2. * intersection / (cardinality + 1e-6)
    return torch.mean(1. - dice_score)


class FCNPre(nn.Sequential):
    def __init__(self, in_channels: int) -> None:
        # layers = [
        #     nn.Conv2d(in_channels, 3, 3, padding=1, bias=False),
        #     # nn.Softmax(dim=1),
        # ] if in_channels > 1 else []

        layers = [
            nn.Conv2d(in_channels, 1, 1, bias=False)
        ] if in_channels > 1 else []

        super().__init__(*layers)

    def forward(self, x):
        x = super().forward(x)

        x = x.expand(-1, 3, -1, -1)
        return x


class HeartFCNWithMask(FCNWithMask):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_heart', back_weight=0.5)


class ChambersFCNWithMask(FCNWithMask):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 4, stage='fcn_chambers')


class ChambersWithHeartFCNWithMask(FCNWithMask):
    def __init__(self):
        super().__init__(channels=2, num_classes=1 + 4, stage='fcn_chambers_with_heart')


class LADFCNWithMask(FCNWithMask):
    def __init__(self):
        super().__init__(channels=1 + 1 + 4, num_classes=1 + 1, stage='fcn_LAD')


class LADAloneWithMask(FCNWithMask):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_LAD_alone')

# def wce(preds, target, back_weight=0.4):
#     fore = cross_entropy(preds, target, ignore_index=0)
#     back = cross_entropy(preds, torch.where(target == 0, 0, 1), ignore_index=1)
#     return back_weight * back + (1 - back_weight) * fore
