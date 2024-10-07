import glob

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models import ResNet101_Weights
from dataset.datasets import DataSplit
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch import Unet, Linknet, UnetPlusPlus
from segmentation_models_pytorch.utils.metrics import IoU
import segmentation_models_pytorch as smp
import fastai

class FCNLAD(LightningModule):
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

        self.model = fcn_resnet101(weights_backbone=ResNet101_Weights.DEFAULT, num_classes=num_classes, progress=True)
        if self.channels > 1:
            self.model.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


        self.model = UnetPlusPlus(
                    encoder_name='resnet34',
                    encoder_weights='imagenet',
                    # encoder_weights=None,
                    classes=num_classes,
                    activation=None,
                    in_channels = channels
                )

        # self.model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        # if self.channels > 1:
        # self.model.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
       
        self.loss = CrossEntropyLoss(
            # weight=torch.Tensor(np.concatenate([
            #     [back_weight],
            #     np.ones(num_classes - 1) / (num_classes - 1) * (1 - back_weight)]))
            # weight=torch.ones(num_classes, dtype=torch.float) / num_classes
        )

        # self.loss = DiceLoss(smp.losses.MULTICLASS_MODE
        #     #   mode='binary',
        #     #   log_loss=True

        # )
        # self.iou = IoU(threshold=0.5)
        # self.loss = CrossEntropyLoss(
        #     weight=torch.full(size=(num_classes,), fill_value=1 / num_classes, dtype=torch.float)
        # )

    def configure_optimizers(self):
        # return torch.optim.SGD(self.model.parameters(), lr=0.01) # ain't it :/
        # return torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.99)
        # return torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.99)
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)
        # return torch.optim.Adam(self.model.parameters(), lr=0.01)
        #  torch.optim.AdamW
        # return torch.optim.Adam(self.parameters(), lr=1e-3)


    def forward(self, x):
        x = x
        # x = self.expand_x(x)
        if len(x.shape) == 3:  # Assuming the shape is [channels, height, width]
            x = x.unsqueeze(0)
        x = x.float()
        # x = self.model(x)['out']        
        x = self.model(x)
    
        return x

    def expand_x(self, x):
        if len(x.shape) == 3:
            x = x[:, None]
        if self.channels == 1:
            x = x.expand(-1, 3, -1, -1)
        
        return x

    def shared_eval_step(self, phase, batch, *args, **kwargs):
        x, y = batch
        y_hat = self(x)
        # if y_hat.shape[1] > 1:  # for multiclass
        #     y = y.unsqueeze(1)  # Add the channel dimension
        #     y = y.repeat(1, y_hat.shape[1], 1, 1) 

        # if len(y.shape) == 2:  # Assuming the shape is [height, width]
        #     y = y.unsqueeze(0)
        # y = y.unsqueeze(1)
        y = y.long()
        loss = self.loss(y_hat, y)
        dice = dice_loss(y_hat, y)

        # loss = wce(y_hat, y)
        self.log(f'{phase}_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        # self.log(f'{phase}_dice', dice, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return dict(
            loss=loss,
            # _dice=torch.stack(
            #     [torchmetrics.functional.dice(sample[0], sample[1], ignore_index=0, num_classes=self.num_classes)
            #      for sample in zip(torch.unsqueeze(y_hat, 1), torch.unsqueeze(y, 1))]
            # ),
            # _dice_support=torch.sum(y_hat, dim=(-2, -1)) + torch.sum(y, dim=(-2, -1))
        )

    
    def validation_step(self, *args, **kwargs):
        return self.shared_eval_step('val', *args, **kwargs)    
   

    def training_step(self, *args, **kwargs):
        return self.shared_eval_step('train', *args, **kwargs)
    

    def test_step(self, *args, **kwargs):
        return self.shared_eval_step('test', *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[:-1][0]
        y_hat = self(x)
        return torch.argmax(y_hat, dim=1)
    

class UnetPuls(LightningModule):
    @classmethod
    def load(cls, data_split: DataSplit, v_num: int):
        files = glob.glob(f'E:/radio/logs/{data_split.name}_{cls.__name__}/version_{v_num}/checkpoints/*.ckpt')
        if len(files) != 1:
            raise ValueError
        return cls.load_from_checkpoint(files[0])

    def __init__(self, channels, num_classes, stage, back_weight=0.1):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.stage = stage

        self.model = smp.Unet(
            encoder_name='resnet101',
            encoder_weights='imagenet',
            # encoder_weights=None,
            in_channels=channels ,
            classes=num_classes,
            # activation=None,
            encoder_depth=7, 
            # decoder_channels=(128, 64, 32, 16, 8, 4, 2)
        )
        # params = smp.encoders.get_preprocessing_params('resnet34')
        # self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        # self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss = CrossEntropyLoss(
                # weight=torch.Tensor(np.concatenate([
                # [back_weight],
                # np.ones(num_classes - 1) / (num_classes - 1) * (1 - back_weight)]))

        )
        self.dice_loss = DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def configure_optimizers(self):
        # print(self.model)
        return torch.optim.SGD(self.model.parameters(), lr=1e-3)

        return self.optimizer

    def forward(self, x):
        # print(f"x shape: {x.shape}")
        # if len(x.shape) == 3:  # Assuming the shape is [channels, height, width]
        #     x = x.unsqueeze(0).repeat(2, 1, 1, 1)
        # elif len(x.shape) == 4 and x.shape[0] == 1:
        #     x = x.repeat(2, 1, 1, 1)
        # if len(x.shape) == 3:  # Assuming the shape is [height, width, channels]
        #     x = x.unsqueeze(0).unsqueeze(0)  # Adding batch and channel dimensions
        # elif len(x.shape) == 4 and x.shape[1] != self.channels:
        #     x = x.unsqueeze(1)  # Adding channel dimension
        # elif len(x.shape) == 5:
        #     x = x.squeeze(0)
        #     # x = x.squeeze(0).unsqueeze(0)
        # if len(x.shape) == 3:  # Assuming the shape is [channels, height, width]
        #     x = x.unsqueeze(0).repeat(2, 1, 1, 1)
        # if self.channels == 1:
        #     x = x.expand(-1, 3, -1, -1)
        # if x.dim() == 3:  # Assuming the shape is [batch_size, height, width]
        #     x = x.unsqueeze(1)  # Add channel dimension
        # elif x.dim() == 4 and x.size(1) != self.channels:
        #     x = x[:, :self.channels]         print(f"x shape: {x.shape}")
        # if len(x.shape) == 3:  # [channels, height, width]
        #     x = x.unsqueeze(0)  # Add batch dimension
    
        # if self.channels == 1:
        #     x = x.expand(-1, 3, -1, -1)

        # channels = x.shape[1]
        # mean_expanded = self.mean[:, :channels, :, :].expand(x.shape[0], channels, x.shape[2], x.shape[3])
        # mean_expanded = mean_expanded.to(x.device)

        # std_expanded = self.std[:, :channels, :, :].expand(x.shape[0], channels, x.shape[2], x.shape[3])
        # std_expanded = std_expanded.to(x.device)
        x = x.float()

        # print(f"x shape: {x.shape}")
        # print(self.mean, self.std.shape )


        # x = (x - mean_expanded) / std_expanded
        # print(f"x shape: {x.shape}")

        x = self.model(x)
        return x

    def expand_x(self, x):
        if len(x.shape) == 3:
            x = x[:, None]
        if self.channels == 1:
            x = x.expand(-1, 3, -1, -1)
        return x
    
    def normalize_data(self, data):
        """Normalize the image data to the range [0, 1]."""
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data


    def shared_eval_step(self, phase, batch, *args, **kwargs):
        x, y = batch
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
        # if len(y.shape) == 3:  # [channels, height, width]
        #     y = y[:, None]  
        # if self.channels == 1:
        #     y = y.expand(-1, 3, -1, -1)
        # x = self.normalize_data(x)
# (batch_size, num_channels, height, width)
        if len(x.shape) == 3:
            x = x[:, None]    
        assert y.max() <= 1.0 and y.min() >= 0

# [batch_size, num_classes, height, width]
        y_hat = self(x)


        # print(f"y shape: {y.shape}")

        # if y_hat.shape[1] > 1:  # for multiclass
        #     y = y.unsqueeze(1)  # Add the channel dimension
        #     y = y.repeat(1, y_hat.shape[1], 1, 1) 
        # print(f"y_hat shape: {y_hat.shape}")        
        y = y.long()
        if y.shape[0] != y_hat.shape[0]:
            raise ValueError(f'Batch size of predictions ({y_hat.shape[0]}) does not match batch size of targets ({y.shape[0]})')

        loss = self.loss(y_hat, y)
        # dice = self.dice_loss(y_hat, y)
        if torch.isnan(x).any() or torch.isnan(y).any() :
            print("NaN detected in inputs, outputs, or loss")
        if  torch.isnan(loss).any():
            print('loss none')



        self.log(f'{phase}_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        # self.log(f'{phase}_dice', dice, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        return dict(
            loss=loss
            # dice=dice
        )
    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
    #     # Clip gradients here
    #     if gradient_clip_algorithm == "norm":
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
    #     elif gradient_clip_algorithm == "value":
    #         torch.nn.utils.clip_grad_value_(self.parameters(), gradient_clip_val)

    def validation_step(self, *args, **kwargs):
        return self.shared_eval_step('val', *args, **kwargs)

    def training_step(self, *args, **kwargs):
        return self.shared_eval_step('train', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.shared_eval_step('test', *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        y_hat = self(x)
        return torch.argmax(y_hat, dim=1)


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



class LADAlone(FCNLAD):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_LAD_cube')

class LADwithChambers(UnetPuls):
    def __init__(self):
        super().__init__(channels=3, num_classes=1 + 1, stage='fcn_LAD_cube_with_chambers')

class LADwithHeart(FCNLAD):
    def __init__(self):
        super().__init__(channels=2, num_classes=1 + 1, stage='fcn_LAD_all_with_heart', back_weight=0.5)


class CancerBed(FCNLAD):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_cancer_bed_cube')

class CancerBedWithBreast(FCNLAD):
    def __init__(self):
        super().__init__(channels=2, num_classes=1 + 1, stage='fcn_cancer_bed_cube_with_breast')
# nan loss 
class CancerBreast(UnetPuls):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_cancer_breast_cube', back_weight=0.5)

class SecondBreast(FCNLAD):
    def __init__(self):
        super().__init__(channels=1, num_classes=1 + 1, stage='fcn_second_breast_cube', back_weight=0.5)
