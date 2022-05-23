import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn

class TotalVariationConv(pl.LightningModule):
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
        variance_right = self.variance_right_filter(mask.unsqueeze(1)).abs()

        variance_down = self.variance_down_filter(mask.unsqueeze(1)).abs()

        total_variance = (variance_right + variance_down).mean()
        return total_variance

class MaskAreaLoss():
    def __init__(self, image_size=224, min_area=0.0, max_area=1.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.image_size = image_size
        self.min_area = min_area
        self.max_area = max_area

        assert(self.min_area >= 0.0 and self.min_area <= 1.0)
        assert(self.max_area >= 0.0 and self.max_area <= 1.0)
        assert(self.min_area <= self.max_area)
        
    def __call__(self, masks):
        batch_size = masks.size()[0]
        losses = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            mask = masks[i].flatten()
            sorted_mask, indices = mask.sort(descending=True)
            losses[i] += (self._min_mask_area_loss(sorted_mask) + self._max_mask_area_loss(sorted_mask)).mean()

        return losses.mean()

    def _min_mask_area_loss(self, sorted_mask):
        if (self.min_area == 0.0):
            return torch.tensor(0.0)

        ones_length = (int)(self.image_size * self.image_size * self.min_area)
        ones = torch.ones(ones_length, device=self.device)
        zeros = torch.zeros((self.image_size * self.image_size) - ones_length, device=self.device)
        ones_and_zeros = torch.cat((ones, zeros), dim=0)

        # [1, 1, 0, 0, 0] - [0.9, 0.9, 0.9, 0.5, 0.1] = [0.1, 0.1, -0.9, -0.5, -0.1] -> [0.1, 0.1, 0, 0, 0]
        loss = F.relu(ones_and_zeros - sorted_mask)

        return loss
    
    def _max_mask_area_loss(self, sorted_mask):
        if (self.max_area == 1.0):
            return torch.tensor(0.0)

        ones_length = (int)(self.image_size * self.image_size * self.max_area)
        ones = torch.ones(ones_length, device=self.device)
        zeros = torch.zeros((self.image_size * self.image_size) - ones_length, device=self.device)
        ones_and_zeros = torch.cat((ones, zeros), dim=0)

        # [0.9, 0.9, 0.9, 0.5, 0.1] - [1, 1, 1, 1, 0] = [-0.1, -0.1, -0.1, -0.5, 0.1] -> [0, 0, 0, 0, 0.1]
        loss = F.relu(sorted_mask - ones_and_zeros)

        return loss

class ClassMaskAreaLoss(MaskAreaLoss):
    def __call__(self, segmentations, target_vectors):
        masks = segmentations.sigmoid()
        batch_size, num_classes, h, w = masks.size()

        losses = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            class_indices = target_vectors[i].eq(1.0)
            class_masks = masks[i][class_indices]
            for j in range(class_masks.size()[0]):
                mask = class_masks[j].flatten()
                sorted_mask, indices = mask.sort(descending=True)
                losses[i] += (self._min_mask_area_loss(sorted_mask) + self._max_mask_area_loss(sorted_mask)).mean()

            losses[i] = losses[i].mean()

        return losses.mean()

def entropy_loss(logits):
    min_prob = 1e-16
    probs = F.softmax(logits, dim=-1).clamp(min=min_prob)
    log_probs = probs.log()
    entropy = (-probs * log_probs)
    entropy_loss = -entropy.mean()

    return entropy_loss



