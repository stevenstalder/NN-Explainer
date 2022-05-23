import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn

class RTSalTotalVariationConv(pl.LightningModule):
    def __init__(self):
        super().__init__()

        weights_right_variance = torch.tensor([[0.0, 0.0, 0.0],
                                              [0.0, 1.0, -1.0],
                                              [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        weights_down_variance = torch.tensor([[0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, -1.0, 0.0]], device=self.device).view(1, 1, 3, 3)

        self.variance_right_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_right_filter.weight.data = weights_right_variance
        self.variance_right_filter.weight.requires_grad = False

        self.variance_down_filter = nn.Conv2d(in_channels=1, out_channels=1,
                                    kernel_size=3, padding=1, padding_mode='reflect', groups=1, bias=False)
        self.variance_down_filter.weight.data = weights_down_variance
        self.variance_down_filter.weight.requires_grad = False

    def forward(self, mask):
        variance_right = self.variance_right_filter(mask.unsqueeze(1)).square()

        variance_down = self.variance_down_filter(mask.unsqueeze(1)).square()

        total_variance = (variance_right + variance_down).mean()
        return total_variance

### Adaption from the preserver loss in Dabkowski et. al. 2017 to account for multi-class target labels ###
def preserver_loss(logits, targets):
    probs = logits.sigmoid() 

    batch_size, num_classes = probs.size()
    num_object_classes_in_batch = 0
    loss = 0.0
    for i in range(batch_size):
        for j in range(num_classes):
            if targets[i][j] == 1.0:
                num_object_classes_in_batch += 1
                loss -= torch.log(probs[i][j])

    loss = loss / num_object_classes_in_batch

    return loss

### Adaption from the destroyer loss in Dabkowski et. al. 2017 to account for multi-class target labels ###
def destroyer_loss(logits, targets):
    probs = logits.sigmoid() 

    batch_size, num_classes = probs.size()
    num_object_classes_in_batch = 0
    loss = 0.0
    for i in range(batch_size):
        for j in range(num_classes):
            if targets[i][j] == 1.0:
                num_object_classes_in_batch += 1
                loss += probs[i][j]

    loss = loss / num_object_classes_in_batch

    return loss