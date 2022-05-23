import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from pathlib import Path

from models.explainer import Deeplabv3Resnet50ExplainerModel
from utils.helper import get_filename_from_annotations, get_targets_from_annotations, extract_masks
from utils.image_utils import save_all_class_masks, save_mask, save_masked_image
from utils.loss import TotalVariationConv, ClassMaskAreaLoss
from utils.metrics import MultiLabelMetrics, SingleLabelMetrics

class InterpretableFCNN(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, class_mask_min_area=0.05, class_mask_max_area=0.3, use_mask_coherency_loss=True, use_mask_variation_loss=True,
                 mask_variation_regularizer=1.0, use_mask_area_loss=True, mask_area_constraint_regularizer=1.0, mask_total_area_regularizer=0.1, ncmask_total_area_regularizer=0.3, 
                 metrics_threshold=-1.0, save_masked_images=False, save_masks=False, save_all_class_masks=False, save_path="./results/"):

        super().__init__()

        self.dataset = dataset

        self.setup_model(num_classes)

        self.setup_losses(dataset=dataset, class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.use_mask_coherency_loss = use_mask_coherency_loss
        self.use_mask_variation_loss = use_mask_variation_loss
        self.mask_variation_regularizer = mask_variation_regularizer
        self.use_mask_area_loss = use_mask_area_loss
        self.mask_area_constraint_regularizer = mask_area_constraint_regularizer
        self.mask_total_area_regularizer = mask_total_area_regularizer
        self.ncmask_total_area_regularizer = ncmask_total_area_regularizer

        # Image display/save settings
        self.save_masked_images = save_masked_images
        self.save_masks = save_masks
        self.save_all_class_masks = save_all_class_masks
        self.save_path = Path(save_path)

        self.learning_rate = learning_rate

    def setup_model(self, num_classes):
        self.model = Deeplabv3Resnet50ExplainerModel(num_classes=num_classes)

    def setup_losses(self, dataset, class_mask_min_area, class_mask_max_area):
        self.total_variation_conv = TotalVariationConv()

        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area)

    def setup_metrics(self, num_classes, metrics_threshold):
        if self.dataset == "CUB":
            self.train_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.valid_metrics = SingleLabelMetrics(num_classes=num_classes)
            self.test_metrics = SingleLabelMetrics(num_classes=num_classes)
        else:
            self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
            self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, image, targets):
        t_seg, t_mask, t_ncmask, t_logits = self._forward(image, targets)
        masked_image = t_mask.unsqueeze(1) * image
        s_seg, s_mask, s_ncmask, s_logits = self._forward(masked_image, targets)

        return t_seg, s_seg, t_mask, s_mask, t_ncmask, s_ncmask, t_logits, s_logits

    def _forward(self, image, targets):
        segmentations = self.model(image) # [batch_size, num_classes, height, width]
        target_mask, non_target_mask = extract_masks(segmentations, targets) # [batch_size, height, width]
        target_mask_inversed = torch.ones_like(target_mask) - target_mask

        logits = segmentations.mean(dim=(2,3)) # [batch_size, num_classes]
        
        return segmentations, target_mask, non_target_mask, logits
        
    def training_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        t_seg, s_seg, t_mask, s_mask, t_ncmask, s_ncmask, t_logits, s_logits = self(image, targets)
        
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_teacher = self.classification_loss_fn(t_logits, labels)
            classification_loss_student = self.classification_loss_fn(s_logits, labels)
        else:
            classification_loss_teacher = self.classification_loss_fn(t_logits, targets)
            classification_loss_student = self.classification_loss_fn(s_logits, targets)

        logits_similarity_loss = (t_logits - s_logits).abs().mean()
        classification_loss = classification_loss_teacher + classification_loss_student + logits_similarity_loss

        loss = classification_loss

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
            mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            loss += mask_area_loss

        if self.use_mask_coherency_loss:
            mask_coherency_loss = (t_mask - s_mask).abs().mean()
            loss += mask_coherency_loss

        self.log('loss', loss)
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.valid_metrics(s_logits, labels)
        else:
            self.valid_metrics(s_logits, targets)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        t_seg, s_seg, t_mask, s_mask, t_ncmask, s_ncmask, t_logits, s_logits = self(image, targets)
        
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_teacher = self.classification_loss_fn(t_logits, labels)
            classification_loss_student = self.classification_loss_fn(s_logits, labels)
        else:
            classification_loss_teacher = self.classification_loss_fn(t_logits, targets)
            classification_loss_student = self.classification_loss_fn(s_logits, targets)

        logits_similarity_loss = (t_logits - s_logits).abs().mean()
        classification_loss = classification_loss_teacher + classification_loss_student + logits_similarity_loss

        loss = classification_loss

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
            mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            loss += mask_area_loss

        if self.use_mask_coherency_loss:
            mask_coherency_loss = (t_mask - s_mask).abs().mean()
            loss += mask_coherency_loss

        self.log('val_loss', loss)
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.valid_metrics(s_logits, labels)
        else:
            self.valid_metrics(s_logits, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        t_seg, s_seg, t_mask, s_mask, t_ncmask, s_ncmask, t_logits, s_logits = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            filename = self.save_path / "masked_image" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, s_mask, filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            save_mask(s_mask, self.save_path / "masks" / filename)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = self.save_path / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, t_seg, filename)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_teacher = self.classification_loss_fn(t_logits, labels)
            classification_loss_student = self.classification_loss_fn(s_logits, labels)
        else:
            classification_loss_teacher = self.classification_loss_fn(t_logits, targets)
            classification_loss_student = self.classification_loss_fn(s_logits, targets)

        logits_similarity_loss = (t_logits - s_logits).abs().mean()
        classification_loss = classification_loss_teacher + classification_loss_student + logits_similarity_loss

        loss = classification_loss

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(t_mask) + self.total_variation_conv(s_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * (self.class_mask_area_loss_fn(t_seg, targets) + self.class_mask_area_loss_fn(s_seg, targets))
            mask_area_loss += self.mask_total_area_regularizer * (t_mask.mean() + s_mask.mean())
            mask_area_loss += self.ncmask_total_area_regularizer * (t_ncmask.mean() + s_ncmask.mean())
            loss += mask_area_loss

        if self.use_mask_coherency_loss:
            mask_coherency_loss = (t_mask - s_mask).abs().mean()
            loss += mask_coherency_loss

        self.log('test_loss', loss)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            self.test_metrics(s_logits, labels)
        else:
            self.test_metrics(s_logits, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)