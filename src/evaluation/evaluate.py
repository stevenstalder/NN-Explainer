import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
from pathlib import Path

from eval_utils.compute_scores import compute_numbers

############################################## Change to your settings ##########################################################
masks_path = Path("./masks/")
data_base_path = Path("../../datasets/")
VOC_segmentations_path = Path("../../datasets/VOC2007/VOCdevkit/VOC2007/SegmentationClass/")
COCO_segmentations_path = Path("./coco_segmentations/")

datasets = ["VOC", "COCO"]
classifiers = ["vgg16", "resnet50"]
vgg16_voc_checkpoint = "../checkpoints/pretrained_classifiers/vgg16_voc.ckpt"
vgg16_coco_checkpoint = "../checkpoints/pretrained_classifiers/vgg16_coco.ckpt"
resnet50_voc_checkpoint = "../checkpoints/pretrained_classifiers/resnet50_voc.ckpt"
resnet50_coco_checkpoint = "../checkpoints/pretrained_classifiers/resnet50_coco.ckpt"

methods = ["explainer", "extremal_perturbations", "grad_cam", "rise",
          "rt_saliency", "igos_pp", "0.5", "0", "1", "perfect"]
#################################################################################################################################

try:
    results = np.load("results.npz", allow_pickle=True)["results"].item()
except:
    results = {}

for dataset in datasets:
    if not(dataset in results):
        results[dataset] = {}
    for classifier in classifiers:
        if not(classifier in results[dataset]):
            results[dataset][classifier] = {}
        for method in methods:
            if not(method in results[dataset][classifier]):
                results[dataset][classifier][method] = {}            
            try:
                if dataset == "VOC":
                    data_path = data_base_path / "VOC2007"
                    segmentations_path = VOC_segmentations_path
                    if classifier == "vgg16":
                        model_path = vgg16_voc_checkpoint
                    elif classifier == "resnet50":
                        model_path = resnet50_voc_checkpoint
                elif dataset == "COCO":
                    data_path = data_base_path / "COCO2014"
                    segmentations_path = COCO_segmentations_path
                    if classifier == "vgg16":
                        model_path = vgg16_coco_checkpoint
                    elif classifier == "resnet50":
                        model_path = resnet50_coco_checkpoint

                d_f1_25,d_f1_50,d_f1_75,c_f1,a_f1s, acc, aucs, d_IOUs, c_IOU, sal, over, background_c, mask_c, sr = compute_numbers(data_path=data_path,
                                                                                                                              masks_path=masks_path, 
                                                                                                                              segmentations_path=segmentations_path, 
                                                                                                                              dataset_name=dataset, 
                                                                                                                              model_name=classifier, 
                                                                                                                              model_path=model_path, 
                                                                                                                              method=method)


                d = {}
                d["d_f1_25"] = d_f1_25
                d["d_f1_50"] = d_f1_50
                d["d_f1_75"] = d_f1_75
                d["d_f1"] = ((np.array(d_f1_25) + np.array(d_f1_50) + np.array(d_f1_75)) /3).tolist()
                d["c_f1"] = c_f1
                d["a_f1s"] = a_f1s
                d["acc"] = acc
                d["aucs"] = aucs
                d["d_IOUs"] = d_IOUs
                d["c_IOU"] = c_IOU
                d["sal"] = sal
                d["over"] = over
                d["background_c"] = background_c
                d["mask_c"] = mask_c
                d["sr"] = sr
                results[dataset][classifier][method] = d
                print("Scores computed for: {} - {} - {}".format(dataset, classifier, method))
            except:
                print("Cannot compute scores for: {} - {} - {}!".format(dataset, classifier, method))

np.savez("results.npz", results=results)