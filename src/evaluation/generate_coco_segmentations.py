from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys

############################## Change to your settings ##############################
test_annotations_path = '../../datasets/COCO2014/annotations/instances_val2014.json'
num_segmentations = 1000
seg_dir = './coco_segmentations/'
#####################################################################################

if not os.path.isdir(seg_dir):
    os.mkdir(seg_dir)

annotation = Path(test_annotations_path)
coco = COCO(annotation)
keys = coco.imgs.keys()
ids = list(random.sample(keys, len(keys)))
for index in range(num_segmentations):
    img_id = ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    coco_annotation = coco.loadAnns(ann_ids)
    img = coco.loadImgs(img_id)[0]
    path = img['file_name'] 

    num_objects = len(coco_annotation)
    mask = np.zeros((img["height"], img["width"]))
    for i in range(num_objects):
        mask = np.maximum(coco.annToMask(coco_annotation[i]), mask)

    path_file = seg_dir + os.path.splitext(path)[0]
    plt.imsave(path_file + ".png", mask, cmap='gray',format="png")
    np.savez_compressed(path_file + ".npz", mask)



