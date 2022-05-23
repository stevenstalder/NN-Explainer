import os
import torch
import numpy as np

import pytorch_lightning as pl
import torchvision.transforms as T

from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

from data.dataset import COCODataset, CUB200Dataset

class VOCDataModule(pl.LightningDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False):
        super().__init__()

        self.data_path = Path(data_path)

        if os.path.exists(self.data_path) and len(os.listdir(self.data_path)) > 2:
            self.download = False
        else:
            self.download = True

        self.train_transformer = get_training_image_transformer(use_data_augmentation)
        self.test_transformer = get_testing_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        
    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = VOCDetection(self.data_path, year="2007", image_set="train", 
                                      download=self.download, transform=self.train_transformer)

            self.val   = VOCDetection(self.data_path, year="2007", image_set="val", 
                                      download=self.download, transform=self.test_transformer)

        if stage == "test" or stage is None:
            self.test  = VOCDetection(self.data_path, year="2007", image_set="test", 
                                      download=self.download, transform=self.test_transformer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, 
                          shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())

class COCODataModule(pl.LightningDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False):
        super().__init__()

        self.data_path = Path(data_path)
        self.annotations_path = self.data_path / 'annotations'

        self.train_transformer = get_training_image_transformer(use_data_augmentation)
        self.test_transformer = get_testing_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = COCODataset(root=self.data_path / 'train2014', 
                                     annotation=self.annotations_path / 'train2014_train_split.json', 
                                     transform_fn=self.train_transformer)

            self.val = COCODataset(root=self.data_path / 'train2014', 
                                   annotation=self.annotations_path / 'train2014_val_split.json', 
                                   transform_fn=self.test_transformer)

        if stage == "test" or stage is None:
            self.test = COCODataset(root=self.data_path / 'val2014', 
                                    annotation=self.annotations_path / 'instances_val2014.json', 
                                    transform_fn=self.test_transformer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, 
                          shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())

class CUB200DataModule(pl.LightningDataModule):

    def __init__(self, data_path, train_batch_size=16, val_batch_size=16, test_batch_size=16, use_data_augmentation=False):
        super().__init__()

        self.data_path = Path(data_path)
        self.annotations_path = self.data_path / 'annotations'

        self.train_transformer = get_training_image_transformer(use_data_augmentation)
        self.test_transformer = get_testing_image_transformer()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train = CUB200Dataset(root=self.data_path / 'train', 
                                       annotations=self.annotations_path / 'train.txt', 
                                       transform_fn=self.train_transformer)

            self.val = CUB200Dataset(root=self.data_path / 'val', 
                                     annotations=self.annotations_path / 'val.txt', 
                                     transform_fn=self.test_transformer)

        if stage == "test" or stage is None:
            self.test = CUB200Dataset(root=self.data_path / 'test', 
                                      annotations=self.annotations_path / 'test.txt', 
                                      transform_fn=self.test_transformer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.train_batch_size, collate_fn=collate_fn, 
                          shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size, collate_fn=collate_fn, 
                          num_workers=4, pin_memory=torch.cuda.is_available())


def get_training_image_transformer(use_data_augmentation=False):
    if use_data_augmentation:
        transformer = T.Compose([ T.RandomHorizontalFlip(),
                                  T.RandomRotation(10),
                                  T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                  #T.Resize(256),
                                  #T.CenterCrop(224),
                                  T.Resize(size=(224,224)),
                                  T.ToTensor(), 
                                  T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    else:
        transformer = T.Compose([ T.Resize(size=(224,224)),
                                  # T.Resize(256),
                                  # T.CenterCrop(224),
                                  T.ToTensor(), 
                                  T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    return transformer

def get_testing_image_transformer():
    transformer = T.Compose([ T.Resize(size=(224,224)),
                              T.ToTensor(), 
                              T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    return transformer

def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])
    target = [item[1] for item in batch]
    return data, target
