import os
import pickle
import pathlib
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, root, annotation, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objects = len(coco_annotation)
        cat_ids = []
        for i in range(num_objects):
            cat_ids.append(coco_annotation[i]['category_id'])

        targets = coco.getCatIds(catIds=cat_ids)

        my_annotation = {}
        my_annotation["targets"] = targets
        my_annotation["image_id"] = img_id
        my_annotation["filename"] = path

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

class CUB200Dataset(Dataset):
    def __init__(self, root, annotations, transform_fn=None):
        self.root = root
        self.transform = transform_fn
        with open(annotations, 'rb') as fp:
            self.annotations = pickle.load(fp)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        filename = pathlib.PurePath(annotation['filename']).name

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        class_label = annotation['class']['label']

        my_annotation = {}
        my_annotation["target"] = class_label
        my_annotation["filename"] = filename

        if self.transform is not None:
            img = self.transform(img)

        return img, my_annotation

    def __len__(self):
        return len(self.annotations)
