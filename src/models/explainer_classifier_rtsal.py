import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from pathlib import Path

from models.explainer import *
from models.classifier import *
from utils.helper import *
from utils.rtsal_helper import generate_alternative_image
from utils.image_utils import *
from utils.rtsal_loss import RTSalTotalVariationConv, preserver_loss, destroyer_loss
from utils.metrics import *

class RTSalExplainerClassifierModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", classifier_type="vgg16", classifier_checkpoint=None, 
                 fix_classifier=True, learning_rate=1e-5, metrics_threshold=-1.0, save_masked_images=False, 
                 save_masks=False, save_all_class_masks=False, save_path="./results/"):

        super().__init__()

        self.setup_explainer(num_classes=num_classes)
        self.setup_classifier(classifier_type=classifier_type, classifier_checkpoint=classifier_checkpoint, fix_classifier=fix_classifier, num_classes=num_classes)

        self.setup_losses()
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.dataset = dataset
        self.classifier_type = classifier_type

        # Hyperparameters
        self.learning_rate = learning_rate

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

    def setup_losses(self):
        self.total_variation_conv = RTSalTotalVariationConv()

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, image, targets):
        segmentations = self.explainer(image)
        target_mask, _ = extract_masks(segmentations, targets)
        inversed_target_mask = torch.ones_like(target_mask) - target_mask

        alternative_image = generate_alternative_image(image)
        masked_image = target_mask.unsqueeze(1) * image + inversed_target_mask.unsqueeze(1) * alternative_image
        inversed_masked_image = inversed_target_mask.unsqueeze(1) * image + target_mask.unsqueeze(1) * alternative_image

        logits_mask = self.classifier(masked_image)
        logits_inversed_mask = self.classifier(inversed_masked_image)

        return logits_mask, logits_inversed_mask, target_mask

    def training_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask = self(image, targets)

        tv_loss = self.total_variation_conv(target_mask)
        av_loss = target_mask.mean()
        pres_loss = preserver_loss(logits_mask, targets)
        destr_loss = destroyer_loss(logits_inversed_mask, targets)

        loss = 10*tv_loss + 1.0*av_loss + pres_loss + 5*torch.pow(destr_loss, 0.3) # lambda regularisers taken from Dabkowski et. al. 2017, second regulariser has been tuned from 0.001 to 1.0

        self.log('loss', loss)
        self.train_metrics(logits_mask, targets)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask = self(image, targets)
        
        tv_loss = self.total_variation_conv(target_mask)
        av_loss = target_mask.mean()
        pres_loss = preserver_loss(logits_mask, targets)
        destr_loss = destroyer_loss(logits_inversed_mask, targets)

        loss = 10*tv_loss + 1.0*av_loss + pres_loss + 5*torch.pow(destr_loss, 0.3) # lambda regularisers taken from Dabkowski et. al. 2017, second regularizer has been tuned from 0.001 to 1.0

        self.log('val_loss', loss)
        self.valid_metrics(logits_mask, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=self.dataset)
        logits_mask, logits_inversed_mask, target_mask = self(image, targets)

        if self.save_masked_images and image.size()[0] == 1:
            filename = Path(self.save_path) / "masked_images" / get_filename_from_annotations(annotations, dataset=self.dataset)
            save_masked_image(image, target_mask, filename)

        if self.save_masks and image.size()[0] == 1:
            filename = get_filename_from_annotations(annotations, dataset=self.dataset)

            ### Commented code was used to produce exactly those masks that correspond to segmentation groundtruths ###
            #if self.dataset == "VOC":
            #    segmentation_filename = '/scratch/snx3000/sstalder/VOCData/VOCdevkit/VOC2007/SegmentationClass/' + os.path.splitext(filename)[0] + '.png'
            #elif self.dataset == "COCO":
            #    segmentation_filename = './benchmark/coco_segmentations/' + filename

            #if not os.path.exists(segmentation_filename):
            #    return

            save_mask(target_mask, Path(self.save_path) / "masks" / filename)

        tv_loss = self.total_variation_conv(target_mask)
        av_loss = target_mask.mean()
        pres_loss = preserver_loss(logits_mask, targets)
        destr_loss = destroyer_loss(logits_inversed_mask, targets)

        loss = 10*tv_loss + 1.0*av_loss + pres_loss + 5*torch.pow(destr_loss, 0.3) # lambda regularisers taken from Dabkowski et. al. 2017, second regularizer has been tuned from 0.001 to 1.0

        self.log('test_loss', loss)
        self.test_metrics(logits_mask, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="rtsal_explainer", classifier_type=self.classifier_type, dataset=self.dataset)

        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
