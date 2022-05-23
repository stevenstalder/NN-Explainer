"""
main file to call the explanations methods and run experiments, given a pre-trained
model and a data loader.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""
import torch
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from pathlib import Path
from torchray.utils import get_device
from tqdm import tqdm
from timeit import default_timer

from args import init_args
from igos_utils import *
from methods_helper import *
from methods import iGOS_pp


from data.dataloader import VOCDataModule, COCODataModule
from models.classifier import VGG16ClassifierModel, Resnet50ClassifierModel
from utils.helper import *
from utils.image_utils import save_mask

##################### Change to your settings ################################
dataset = "VOC" # one of: ["VOC", "COCO"]
data_base_path = "../../../datasets/"
classifier_type = "vgg16" # one of: ["vgg16", "resnet50"]
classifier_checkpoint = "../../checkpoints/pretrained_classifiers/vgg16_voc.ckpt"
VOC_segmentations_path = '../../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
COCO_segmentations_path = '../coco_segmentations/'
##############################################################################

def gen_explanations(model, dataloader, args):

    device = get_device()

    model.eval()

    method = iGOS_pp

    save_path = Path('../masks/{}_{}_{}/'.format(dataset, classifier_type, "igos_pp"))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    eprint(f'Size is {args.size}x{args.size}')

    total_time = 0.0
    for data in tqdm(dataloader):

        image, annotations = data

        filename = get_filename_from_annotations(annotations, dataset=dataset)
        if dataset == "VOC":
            num_classes = 20
            segmentation_filename = Path(VOC_segmentations_path) / (os.path.splitext(filename)[0] + '.png')
            data_path = Path(data_base_path) / 'VOC2007' / 'VOCdevkit' / 'VOC2007' / 'JPEGImages' 
        elif dataset == "COCO":
            num_classes = 91
            segmentation_filename = Path(COCO_segmentations_path) / (os.path.splitext(filename)[0] + '.png')
            data_path = Path(data_base_path) / 'COCO2014' / 'val2014' 
        if not os.path.exists(segmentation_filename):
            continue
        
        img_path = data_path / filename
        blurred_image = get_blurred_image(img_path)

        image = image.to(device)
        blurred_image = blurred_image.to(device).unsqueeze(0)

        targets = get_targets_from_annotations(annotations, dataset=dataset)[0]

        masks = torch.zeros(num_classes, 224, 224)

        start_time = default_timer()
        for i in range(num_classes):
            if targets[i] == 1.0:
                # generate masks
                label = torch.tensor([i], dtype=torch.int64, device=device)
                mask = method(
                    model,
                    images=image.detach(),
                    baselines=blurred_image.detach(),
                    labels=label,
                    size=args.size,
                    iterations=args.ig_iter,
                    ig_iter=args.iterations,
                    L1=args.L1,
                    L2=args.L2,
                    alpha=args.alpha,
                )

                mask = mask.view(-1, 1, args.size, args.size)
                up = torch.nn.UpsamplingBilinear2d(size=(224, 224)).to(device)

                u_mask = up(mask)
                u_mask = u_mask.permute((0, 2, 3, 1))

                # Normalize the mask
                u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))

                masks[i] = torch.ones(size=(224, 224), device=device) - u_mask.squeeze()

        aggregated_mask = masks.amax(dim=0)
        end_time = default_timer()
        total_time += end_time - start_time

        save_mask(aggregated_mask, save_path / filename)

    print("Total time for masking process of iGOS++ with dataset {} and classifier {}: {} seconds".format(dataset, classifier_type, total_time))
    model.train()


if __name__ == "__main__":

    args = init_args()
    eprint(f"args:\n {args}")

    torch.manual_seed(args.manual_seed)

    init(args.input_size)
    init_sns()
    
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


    eprint("Loading the model...")

    if classifier_type == "vgg16":
        classifier = VGG16ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
    elif classifier_type == "resnet50":
        classifier = Resnet50ClassifierModel.load_from_checkpoint(classifier_checkpoint, num_classes=num_classes, dataset=dataset)
    else:
        raise Exception("Unknown classifier type: " + classifier_type)

    for param in classifier.parameters():
        param.requires_grad_(False)

    device = get_device()
    classifier.to(device)
    classifier.eval()

    eprint(f"Model({classifier_type}) successfully loaded!\n")

    
    gen_explanations(classifier, data_module.test_dataloader(), args)