import torch
from pytorch_lightning import LightningModule
from torch.nn import Sequential, Dropout, Conv2d, ReLU, AdaptiveAvgPool2d
from torch.nn.functional import cross_entropy
from torchmetrics.functional.classification import multiclass_accuracy
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


class SliceClassifier(LightningModule):
    def __init__(self, classes=2):
        super().__init__()

        # self.model = squeezenet1_1(pretrained=True)
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        self.model.classifier = Sequential(
            Dropout(p=0.5, inplace=False),
            Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1)),
            ReLU(inplace=True),
            AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.stage = 'slice_classifier'

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.01)

    def forward(self, x):
        x = x.to(torch.float)
        x = x[:, None].expand(-1, 3, -1, -1)
        y_hat = self.model(x)
        return y_hat

    def shared_eval_step(self, phase, batch, *args, **kwargs):
        x, y = batch
        y_hat = self(x)

        y = y.to(int)

        loss = cross_entropy(y_hat, y)
        acc = multiclass_accuracy(y_hat, y, num_classes=2, average='micro')

        self.log(f'{phase}_accuracy', acc, prog_bar=True, logger=True, on_epoch=True)
        return dict(
            loss=loss,
            accuracy=acc,
        )

    def training_step(self, *args, **kwargs):
        return self.shared_eval_step('train', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.shared_eval_step('test', *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return torch.argmax(y_hat, dim=1)
