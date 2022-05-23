import pytorch_lightning as pl

from torchvision import models

class Deeplabv3Resnet50ExplainerModel(pl.LightningModule):
    def __init__(self, num_classes=20):
        super().__init__()
        self.explainer = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        x = self.explainer(x)['out']
        return x


