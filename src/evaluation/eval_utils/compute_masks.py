import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchray.utils import get_device
from tqdm import tqdm 
from timeit import default_timer

from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, Perturbation

from utils.helper import get_target_dictionary

def merge_mask(masks):
    if len(masks) > 1:
        return torch.max(masks, dim = 0, keepdims=True)[0]
    return masks

def rescale_mask(mask):
    min_val = mask.min()
    max_val = mask.max()

    mask = (mask - min_val) / (max_val - min_val)
    mask = mask.clamp(0, 1)
    return mask

def compute_mask(x, model, category_id, method="extremal_perturbations"):
    masks = []
    for c in category_id:
        if method=="guided_backprop":
            m = guided_backprop(model, x,c)

        elif method=="extremal_perturbations":
            m = vedaldi2019(model, x, c)
        else:
            raise ValueError("Unknown method")
        masks.append(m)
    mask = merge_mask(torch.cat(masks, dim=0))
    mask = rescale_mask(mask)
    return mask

def vedaldi2019(model, x, c):
    areas = [0.05, 0.1, 0.2, 0.3]
    num_levels = 8
    mask, energy = extremal_perturbation(
        model, x, c,
        areas=areas,
        debug=False,
        jitter=True,
        num_levels=num_levels,
        step=7,
        sigma=7 * 3,
        max_iter=800,
        smooth=0.09,
        reward_func=contrastive_reward,
        perturbation='blur'
    )

    perturbation = Perturbation(x, num_levels=num_levels, type='blur')
    x_perturbed = perturbation.apply(mask=mask)

    # saliency = mask.mean(dim=0, keepdim=True)

    logit_perturbed = model.forward(x_perturbed)[:, c]
    logit = model.forward(x)[:, c]
    vec = logit_perturbed < logit
    i = 0
    while vec[i] and i<(len(vec)-1):
        i += 1
    saliency = mask[i:i+1]
    return saliency

def save_mask(mask, p):
    path_file = str(p)[:-4]
    img = mask.detach().cpu().numpy().squeeze()
    plt.imsave(path_file + ".png", img, cmap='gray',format="png")
    np.savez_compressed(path_file+ ".npz", img)

def compute_and_save_masks(model, data_module, path_segmentation, path_masks, method="guided_backprop"):
    # Run on GPU if available.
    device = get_device()
    model.to(device)
    
    path_masks.mkdir(parents=True, exist_ok=True)
    count = 0
    total_time = 0.0 
    for s in tqdm(data_module.test_dataloader()):
        img, meta = s
        x = img
        x = x.to(device)
        assert(len(x)==1)
        try:
            category_id = meta[0]["targets"]
            filename = meta[0]["filename"]
            filename = Path(os.path.splitext(filename)[0] + '.png')
        except:
            filename = Path(meta[0]['annotation']["filename"][:-4]+".png")
            target_dict = get_target_dictionary(include_background_class=False)
            objects = meta[0]['annotation']['object']
            category_id = [target_dict[e["name"]] for e in objects]
        segmentation_filename =  path_segmentation / filename
        if not os.path.exists(segmentation_filename):
            continue
            
        count += 1

        start_time = default_timer()
        mask = compute_mask(x, model, category_id, method=method)
        end_time = default_timer()
        total_time += end_time - start_time

        save_mask(mask, path_masks / filename)

    path_split = str(path_masks).split('/')[1].split('_')
    print("Total time for masking process of {} with dataset {} and classifier {}: {} seconds".format(method, path_split[0], path_split[1], total_time))
    return count