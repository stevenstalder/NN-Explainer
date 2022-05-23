import torch
import pytorch_lightning as pl

from torch import nn
from torch.optim import Adam
from torchvision import models

from utils.helper import get_targets_from_annotations
from utils.metrics import MultiLabelMetrics

class VGG16ClassifierModel(pl.LightningModule):
    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, use_imagenet_pretraining=True, fix_classifier_backbone=True, metrics_threshold=0.0):
        super().__init__()

        self.setup_model(num_classes=num_classes, use_imagenet_pretraining=use_imagenet_pretraining, fix_classifier_backbone=fix_classifier_backbone)

        self.setup_losses(dataset=dataset)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.dataset = dataset
        self.learning_rate = learning_rate

    def setup_model(self, num_classes, use_imagenet_pretraining, fix_classifier_backbone):
        backbone = models.vgg16(pretrained=use_imagenet_pretraining)

        layers = list(backbone.children())[:-1]

        self.feature_extractor = nn.Sequential(*layers[0])
        self.avgpool = layers[1]

        if fix_classifier_backbone:
            self.feature_extractor.eval()

            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            self.avgpool.eval()
            for param in self.avgpool.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def setup_losses(self, dataset):
        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('train_loss', loss)
        self.train_metrics(logits, targets)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('val_loss', loss)
        self.valid_metrics(logits, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('test_loss', loss)
        self.test_metrics(logits, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="classifier", classifier_type="vgg16", dataset=self.dataset)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

class Resnet50ClassifierModel(pl.LightningModule):

    def __init__(self, num_classes=20, dataset="VOC", learning_rate=1e-5, use_imagenet_pretraining=True, fix_classifier_backbone=True, metrics_threshold=0.0):
        super().__init__()

        self.setup_model(num_classes=num_classes, use_imagenet_pretraining=use_imagenet_pretraining, fix_classifier_backbone=fix_classifier_backbone)

        self.setup_losses(dataset=dataset)
        self.setup_metrics(num_classes=num_classes, metrics_threshold=metrics_threshold)

        self.dataset = dataset
        self.learning_rate = learning_rate

    def setup_model(self, num_classes, use_imagenet_pretraining, fix_classifier_backbone):
        backbone = models.resnet50(pretrained=use_imagenet_pretraining)

        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        if fix_classifier_backbone:
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(in_features=num_filters, out_features=num_classes, bias=True)

    def setup_losses(self, dataset):
        if dataset == "CUB":
            self.classification_loss_fn = nn.CrossEntropyLoss()
        else:
            self.classification_loss_fn = nn.BCEWithLogitsLoss()

    def setup_metrics(self, num_classes, metrics_threshold):
        self.train_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.valid_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)
        self.test_metrics = MultiLabelMetrics(num_classes=num_classes, threshold=metrics_threshold)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('train_loss', loss)
        self.train_metrics(logits, targets)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_metrics', self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('val_loss', loss)
        self.valid_metrics(logits, targets)

    def validation_epoch_end(self, outs):
        self.log('val_metrics', self.valid_metrics.compute(), prog_bar=True)
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        targets = get_targets_from_annotations(y, dataset=self.dataset)

        if self.dataset == "CUB":
            labels = targets.argmax(dim=1)
            loss = self.classification_loss_fn(logits, labels)
        else:
            loss = self.classification_loss_fn(logits, targets)

        self.log('test_loss', loss)
        self.test_metrics(logits, targets)

    def test_epoch_end(self, outs):
        self.log('test_metrics', self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.save(model="classifier", classifier_type="resnet50", dataset=self.dataset)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)