import pytorch_lightning as pl
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from pathlib import Path
from timeit import default_timer

from data.dataloader import *
from utils.helper import *
from utils.image_utils import *
from models.classifier import *

from torchray.attribution.rise import rise

############################## Change to your settings ##############################
dataset = 'VOC' # one of: ['VOC', 'COCO']
data_base_path = '../../datasets/'
classifier_type = 'vgg16' # one of: ['vgg16', 'resnet50']
classifier_checkpoint = '../checkpoints/pretrained_classifiers/vgg16_voc.ckpt'

mode = 'seg' # one of: ['seg', 'classes']
if mode == 'seg':
    VOC_segmentations_directory = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
    COCO_segmentations_directory = './coco_segmentations/'
elif mode == 'classes':
    masks_for_classes = [4, 6, 7, 11, 14]
#####################################################################################
    
# Set up data module
if dataset == "VOC":
    num_classes = 20
    data_path = Path(data_base_path) / "VOC2007"
    data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
elif dataset == "COCO":
    num_classes = 91
    data_path = Path(data_base_path) / "COCO2014"
    data_module = COCODataModule(data_path=data_path, test_batch_size=1)
else:
    raise Exception("Unknown dataset " + dataset)

save_path = Path('masks/{}_{}_{}/'.format(dataset, classifier_type, "rise"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

total_time = 0.0

class RISEModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # Set up model
        if classifier_type == "vgg16":
            self.model = VGG16ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
        elif classifier_type == "resnet50":
            self.model = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
        else:
            raise Exception("Unknown classifier type " + classifier_type)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, image):
        saliency = rise(self.model, image)

        return saliency

    def test_step(self, batch, batch_idx):
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=dataset)
        filename = get_filename_from_annotations(annotations, dataset=dataset)

        global total_time
        if mode == 'seg':
            if dataset == "VOC":
                segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
            elif dataset == "COCO":
                segmentation_filename = COCO_segmentations_directory + os.path.splitext(filename)[0] + '.png'
            else:
                raise Exception("Illegal dataset: " + dataset)

            if not os.path.exists(segmentation_filename):
                return
            assert(targets.size()[0] == 1)

            saliencies = torch.zeros(num_classes, 224, 224)
            start_time = default_timer()
            saliency = self(image)
            for class_index in range(num_classes):
                if targets[0][class_index] == 1.0:
                    class_sal = saliency[:, class_index].squeeze()
                    min_val = class_sal.min()
                    max_val = class_sal.max()
                    class_sal = class_sal - min_val
                    class_sal = torch.mul(class_sal, 1 / (max_val - min_val))
                    class_sal = class_sal.clamp(0, 1)
                    saliencies[class_index] = class_sal

            saliency_map = saliencies.amax(dim=0)
            end_time = default_timer()
            total_time += end_time - start_time

            save_mask(saliency_map, save_path / filename)

        elif mode == 'classes':
            target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
            intersection = set(target_classes) & set(masks_for_classes)
            if intersection:
                saliency = self(image)
                for target_class in intersection:
                    for mask_class in masks_for_classes:
                        class_sal = saliency[:, mask_class].squeeze()
                        min_val = class_sal.min()
                        max_val = class_sal.max()
                        class_sal = class_sal - min_val
                        class_sal = torch.mul(class_sal, 1 / (max_val - min_val))
                        class_sal = class_sal.clamp(0, 1)

                        save_mask(class_sal, save_path / "class_masks" 
                                                       / "target_class_{}".format(target_class)
                                                       / "masks_for_class_{}".format(mask_class)
                                                       / filename)

model = RISEModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

if mode == 'seg':
    print("Total time for masking process of RISE with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
