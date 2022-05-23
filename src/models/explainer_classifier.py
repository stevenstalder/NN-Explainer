import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from pathlib import Path

from models.explainer import Deeplabv3Resnet50ExplainerModel
from models.classifier import VGG16ClassifierModel, Resnet50ClassifierModel
from utils.helper import get_targets_from_annotations, get_filename_from_annotations, extract_masks
from utils.image_utils import save_mask, save_masked_image, save_all_class_masks
from utils.loss import TotalVariationConv, ClassMaskAreaLoss, entropy_loss
from utils.metrics import MultiLabelMetrics

class ExplainerClassifierModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", classifier_type="vgg16", classifier_checkpoint=None, fix_classifier=True, learning_rate=1e-5, class_mask_min_area=0.05, 
                 class_mask_max_area=0.3, entropy_regularizer=1.0, use_mask_variation_loss=True, mask_variation_regularizer=1.0, use_mask_area_loss=True, 
                 mask_area_constraint_regularizer=1.0, mask_total_area_regularizer=0.1, ncmask_total_area_regularizer=0.3, metrics_threshold=-1.0,
                 save_masked_images=False, save_masks=False, save_all_class_masks=False, save_path="./results/"):

        super().__init__()

        self.setup_explainer(num_classes=num_classes)
        self.setup_classifier(classifier_type=classifier_type, classifier_checkpoint=classifier_checkpoint, fix_classifier=fix_classifier, num_classes=num_classes)

        self.setup_losses(dataset=dataset, class_mask_min_area=class_mask_min_area, class_mask_max_area=class_mask_max_area)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.dataset = dataset
        self.classifier_type = classifier_type

        # Hyperparameters
        self.learning_rate = learning_rate
        self.entropy_regularizer = entropy_regularizer
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
        self.save_path = save_path

    def setup_explainer(self, num_classes):
        self.explainer = Deeplabv3Resnet50ExplainerModel(num_classes=num_classes)

    def setup_classifier(self, classifier_type, classifier_checkpoint, fix_classifier, num_classes):
        if classifier_type == "vgg16":
            self.classifier = VGG16ClassifierModel(num_classes=num_classes)
        elif classifier_type == "resnet50":
            self.classifier = Resnet50ClassifierModel(num_classes=num_classes)
        else:
            raise Exception("Unknown classifier type " + classifier_type)
            
        if classifier_checkpoint is not None:
            self.classifier = self.classifier.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes)
            if fix_classifier:
                self.classifier.freeze()

    def setup_losses(self, dataset, class_mask_min_area, class_mask_max_area):
        self.total_variation_conv = TotalVariationConv()

        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.class_mask_area_loss_fn = ClassMaskAreaLoss(min_area=class_mask_min_area, max_area=class_mask_max_area)

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, image, targets):
        segmentations = self.explainer(image)
        target_mask, non_target_mask = extract_masks(segmentations, targets)
        inversed_target_mask = torch.ones_like(target_mask) - target_mask

        masked_image = target_mask.unsqueeze(1) * image
        inversed_masked_image = inversed_target_mask.unsqueeze(1) * image

        logits_mask = self.classifier(masked_image)
        logits_inversed_mask = self.classifier(inversed_masked_image)

        return logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations

    def training_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        else:
            classification_loss_mask = self.classification_loss_fn(logits_mask, targets)

        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss

        self.log('train_loss', loss)
        self.train_metrics(logits_mask, targets)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)
        
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        else:
            classification_loss_mask = self.classification_loss_fn(logits_mask, targets)

        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss

        self.log('val_loss', loss)
        self.valid_metrics(logits_mask, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask, non_target_mask, segmentations = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, target_mask, filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)
            save_mask(target_mask, Path(self.save_path) / "masks" / filename)

        if self.save_all_class_masks and image.size()[0] == 1 and self.dataset == "VOC":
            filename = Path(self.save_path) / "all_class_masks" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_all_class_masks(image, segmentations, filename)
        
        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            classification_loss_mask = self.classification_loss_fn(logits_mask, labels)
        else:
            classification_loss_mask = self.classification_loss_fn(logits_mask, targets)

        classification_loss_inversed_mask = self.entropy_regularizer * entropy_loss(logits_inversed_mask)
        loss = classification_loss_mask + classification_loss_inversed_mask

        if self.use_mask_variation_loss:
            mask_variation_loss = self.mask_variation_regularizer * (self.total_variation_conv(target_mask) + self.total_variation_conv(non_target_mask))
            loss += mask_variation_loss

        if self.use_mask_area_loss:
            mask_area_loss = self.mask_area_constraint_regularizer * self.class_mask_area_loss_fn(segmentations, targets)
            mask_area_loss += self.mask_total_area_regularizer * target_mask.mean()
            mask_area_loss += self.ncmask_total_area_regularizer * non_target_mask.mean()
            loss += mask_area_loss

        self.log('test_loss', loss)
        self.test_metrics(logits_mask, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="explainer", classifier_type=self.classifier_type, dataset=self.dataset)

        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
