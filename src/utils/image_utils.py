import numpy as np
import io
import os
import torchvision.transforms as T

import matplotlib.pyplot as plt
import matplotlib as mpl

from PIL import Image

def show_max_activation(image, segmentations, class_id):
    nat_image = get_unnormalized_image(image)

    mask = segmentations[0][class_id].numpy()
    max_pixel_coords = np.unravel_index(mask.argmax(), mask.shape)

    circle = plt.Circle(max_pixel_coords[::-1], 10, fill=False, color='red')
    fig, ax = plt.subplots(1)
    ax.imshow(np.stack(nat_image.squeeze(), axis=2))
    ax.add_patch(circle)

    plt.show()

def save_mask(mask, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    path_file = os.path.splitext(filename)[0]

    img = mask.detach().cpu().numpy().squeeze()
    plt.imsave(path_file + ".png", img, cmap='gray',format="png")
    np.savez_compressed(path_file + ".npz", img)

def save_masked_image(image, mask, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    path_file = os.path.splitext(filename)[0]

    nat_image = get_unnormalized_image(image)
    masked_nat_im = get_masked_image(nat_image, mask)

    plt.imsave(path_file + ".png", np.stack(masked_nat_im.detach().cpu().squeeze(), axis=2), format="png")

def show_image_and_masked_image(image, mask):
    nat_image = get_unnormalized_image(image)
    masked_nat_im = get_masked_image(nat_image, mask)

    fig = get_fullscreen_figure_canvas("Image and masked image")
    fig.add_subplot(1, 2, 1)
    show_image(nat_image)

    fig.add_subplot(1, 2, 2)
    show_image(masked_nat_im)

    plt.show()

def save_all_class_masks(image, segmentations, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    filename = os.path.splitext(filename)[0]

    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("All class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        masked_nat_im = get_masked_image(nat_image, all_class_masks[i])
        add_subplot_with_class_mask(fig, i)
        show_image(masked_nat_im)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)
    im.save(filename, format='png')

    img_buf.close()

def show_target_class_masks(image, segmentations, targets):
    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("Target class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        if targets[0][i] == 1.0:
            masked_nat_im = get_masked_image(nat_image, all_class_masks[i])
            add_subplot_with_class_mask(fig, i)
            show_image(masked_nat_im)

def show_most_likely_class_masks(image, segmentations, logits, threshold=0.0):
    nat_image = get_unnormalized_image(image)
    all_class_masks = segmentations.transpose(0, 1).sigmoid()

    fig = get_fullscreen_figure_canvas("Predicted class masks")
    for i in range(all_class_masks.size()[0]): #loop over all classes
        if logits[0][i] >= threshold:
            masked_nat_im = get_masked_image(nat_image, all_class_masks[i])
            add_subplot_with_class_mask(fig, i)
            show_image(masked_nat_im)

def get_unnormalized_image(image):
    inverse_transform = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])

    nat_image = inverse_transform(image)

    return nat_image

def get_masked_image(image, mask):
    masked_image = mask.unsqueeze(1) * image

    return masked_image

def get_fullscreen_figure_canvas(title):
    mpl.rcParams["figure.figsize"] = (40,40)
    fig = plt.figure()
    fig.suptitle(title)

    return fig

def add_subplot_with_class_mask(fig, class_id):
    target_labels = get_target_labels(include_background_class=False)

    axis = fig.add_subplot(4, 5, class_id+1)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    axis.title.set_text(target_labels[class_id])

def show_image(image):
    plt.imshow(np.stack(image.squeeze(), axis=2))

def get_target_labels(include_background_class):
    if include_background_class:
        targets = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    else:
        targets = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    return targets
