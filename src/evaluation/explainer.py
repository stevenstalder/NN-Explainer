import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from tqdm import tqdm
from pathlib import Path
from torchray.utils import get_device
from timeit import default_timer

from models.explainer_classifier import ExplainerClassifierModel
from data.dataloader import *
from utils.helper import *
from utils.image_utils import *

############################## Change to your settings ##############################
dataset = 'VOC' # one of: ['VOC', 'COCO']
data_base_path = '../../datasets/'
classifier_type = 'vgg16' # one of: ['vgg16', 'resnet50']
explainer_classifier_checkpoint = '../checkpoints/explainer_vgg16_voc.ckpt'

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

data_module.setup(stage = "test")

model = ExplainerClassifierModel.load_from_checkpoint(explainer_classifier_checkpoint, num_classes=num_classes, dataset=dataset, classifier_type=classifier_type)
device = get_device()
model.to(device)
model.eval()

save_path = Path('masks/{}_{}_{}/'.format(dataset, classifier_type, "explainer"))
if not os.path.isdir(save_path):
    os.makedirs(save_path)

total_time = 0.0
for batch in tqdm(data_module.test_dataloader()):
    image, annotations = batch
    image = image.to(device)

    filename = get_filename_from_annotations(annotations, dataset=dataset)
    targets = get_targets_from_annotations(annotations, dataset=dataset)

    if mode == 'seg':
        if dataset == "VOC":
            segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
        elif dataset == "COCO":
            segmentation_filename = COCO_segmentations_directory + os.path.splitext(filename)[0] + '.png'

        if not os.path.exists(segmentation_filename):
            continue

        start_time = default_timer()
        _, _, mask, _, _ = model(image, targets)
        end_time = default_timer()
        total_time += end_time - start_time

        save_mask(mask, save_path / filename)
        save_masked_image(image, mask, save_path / "images" / filename)
    
    elif mode == 'classes':
        target_classes = [index for index, value in enumerate(targets[0]) if value == 1.0]
        intersection = set(target_classes) & set(masks_for_classes)
        if intersection:
            segmentations = model.explainer(image)
            for target_class in intersection:
                for mask_class in masks_for_classes:
                    mask = segmentations[0][mask_class].sigmoid()

                    save_mask(mask, save_path / "class_masks" 
                                              / "target_class_{}".format(target_class)
                                              / "masks_for_class_{}".format(mask_class)
                                              / filename)

if mode == 'seg':
    print("Total time for masking process of the Explainer with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))

