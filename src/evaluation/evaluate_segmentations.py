from PIL import Image
import torchvision.transforms as T
import numpy as np
import os

################################# Change to your settings #################################
dataset = 'VOC'
mask_dir = './masks/VOC_vgg16_explainer/'
VOC_segmentations_directory = '../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/'
COCO_segmentations_directory = './coco_segmentations/'
###########################################################################################


transformer = T.Compose([T.Resize(size=(224,224))])

n_files = len(os.listdir(mask_dir))

total_mask_coverage = 0.0
total_non_seg_mask_coverage = 0.0
total_overlap = 0.0
for filename in os.listdir(mask_dir):
    if os.path.splitext(filename)[1] == '.npz':
        n_files -= 1
        continue

    if dataset == "VOC":
        segmentation_filename = VOC_segmentations_directory + os.path.splitext(filename)[0] + '.png'
    elif dataset == "COCO":
        segmentation_filename = COCO_segmentations_directory + filename
    else:
        raise Exception("Illegal dataset: " + dataset)

    if not os.path.isfile(segmentation_filename):
        n_files -= 1
        continue

    mask = np.array(Image.open(mask_dir + filename).convert('L')) / 255.0

    seg_mask = Image.open(segmentation_filename).convert('L')
    seg_mask = transformer(seg_mask)
    seg_mask = np.array(seg_mask) / 255.0
    if dataset == "VOC":
        seg_mask[seg_mask > 0] = 1

    seg_area = np.average(seg_mask)
    
    if seg_area > 0.0:
        seg_diff = np.clip(seg_mask - mask, a_min=0.0, a_max=None)
        mask_coverage = 1 - (np.average(seg_diff) * (1 / seg_area))
        total_mask_coverage += mask_coverage

    if seg_area < 1.0:
        non_seg_diff = np.clip(mask - seg_mask, a_min=0.0, a_max=None)
        non_seg_mask_coverage = (np.average(non_seg_diff) * (1 / (1-seg_area)))
        total_non_seg_mask_coverage += non_seg_mask_coverage

    overlap = 1 - np.average(np.absolute(seg_mask - mask))
    total_overlap += overlap

mean_mask_coverage = total_mask_coverage / n_files
mean_non_seg_mask_coverage = total_non_seg_mask_coverage / n_files
mean_overlap = total_overlap / n_files

print("Number of files: ", n_files)
print("Mean mask coverage: ", mean_mask_coverage)
print("Mean mask coverage for non-segmented parts: ", mean_non_seg_mask_coverage)
print("Mean overlap between segmentation and mask: ", mean_overlap)
