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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import *

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

save_path = Path('masks/{}_{}_{}/'.format(dataset, classifier_type, "grad_cam"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

total_time = 0.0

class GradCAMModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.use_cuda = (torch.cuda.device_count() > 0)
        # Set up model
        if classifier_type == "vgg16":
            self.model = VGG16ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset, fix_classifier_backbone=False)
            self.target_layer = self.model.feature_extractor[-1]
        elif classifier_type == "resnet50":
            self.model = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset, fix_classifier_backbone=False)
            self.target_layer = self.model.feature_extractor[-2][-1]
        else:
            raise Exception("Unknown classifier type " + classifier_type)

        self.cam = GradCAM(model=self.model, target_layer=self.target_layer, use_cuda=self.use_cuda)

    def forward(self, image, target):
        saliency = self.cam(input_tensor=image, target_category=target)

        return saliency

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        image, annotations = batch
        targets = get_targets_from_annotations(annotations, dataset=dataset)
        filename = get_filename_from_annotations(annotations, dataset=dataset)

        assert(targets.size()[0] == 1)

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

            saliencies = torch.zeros(num_classes, 224, 224)
            start_time = default_timer()
            for class_index in range(num_classes):
                if targets[0][class_index] == 1.0:
                    saliencies[class_index] = torch.tensor(self(image, class_index)[0, :])

            saliency_map = saliencies.amax(dim=0)
            end_time = default_timer()
            total_time += end_time - start_time

            saliency_map.nan_to_num(nan=0.0)

            save_mask(saliency_map, save_path / filename)

        elif mode == 'classes':
            target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
            intersection = set(target_classes) & set(masks_for_classes)
            if intersection:
                for target_class in intersection:
                    for mask_class in masks_for_classes:
                        saliency = torch.tensor(self(image, mask_class)[0, :])
                        saliency.nan_to_num(nan=0.0)

                        save_mask(saliency, save_path / "class_masks" 
                                                      / "target_class_{}".format(target_class)
                                                      / "masks_for_class_{}".format(mask_class)
                                                      / filename)

model = GradCAMModel(num_classes=num_classes)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0)
trainer.test(model=model, datamodule=data_module)

if mode == 'seg':
    print("Total time for masking process of GradCAM with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
