import sys
import os

sys.path.insert(0, os.path.abspath(".."))

from pathlib import Path

from eval_utils.compute_masks import compute_and_save_masks
from data.dataloader import VOCDataModule, COCODataModule
from models.classifier import VGG16ClassifierModel, Resnet50ClassifierModel

##################### Change to your settings ################################
dataset = "VOC" # one of: ["VOC", "COCO"]
data_base_path = "../../datasets/"
classifier_type = "vgg16" # one of: ["vgg16", "resnet50"]
classifier_checkpoint = "../checkpoints/pretrained_classifiers/vgg16_voc.ckpt"
VOC_segmentations_path = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
COCO_segmentations_path = './coco_segmentations/'
##############################################################################

if dataset == "VOC":
	num_classes = 20
	data_path = Path(data_base_path) / 'VOC2007'
	data_module = VOCDataModule(data_path=data_path, test_batch_size=1)
	segmentations_path = Path(VOC_segmentations_path)
elif dataset == "COCO":
	num_classes = 91
	data_path = Path(data_base_path) / 'COCO2014'
	data_module = COCODataModule(data_path=data_path, test_batch_size=1)
	segmentations_path = Path(COCO_segmentations_path)
else:
	raise Exception("Unknown dataset: " + dataset)

if classifier_type == "vgg16":
	classifier = VGG16ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
elif classifier_type == "resnet50":
	classifier = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
else:
	raise Exception("Unknown classifier type: " + classifier_type)

for param in classifier.parameters():
    param.requires_grad_(False)

data_module.setup(stage = "test")

save_path = Path('masks/{}_{}_{}/'.format(dataset, classifier_type, "guided_backprop"))

compute_and_save_masks(model=classifier, data_module=data_module, path_segmentation=segmentations_path, path_masks=save_path, method="guided_backprop")
