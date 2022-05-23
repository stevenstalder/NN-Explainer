"""
Utility functions to use for logging results and explanations.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import torch
import os
import cv2
import time
import sys
import requests

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np

from PIL import Image


# mean and standard deviation for the imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def init_sns():
    """
        Sets up desired configs for plotting with Seaborn.

    :return:
    """
    sns.set()
    sns.despine(offset=10, trim=True)
    sns.set(font='serif')
    sns.set_style("darkgrid", {"font.family": "serif", "font.serif": ["Times"]})
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":14,"axes.labelsize":14})


def init_logger(args):
    """
        Initializes output directory to save the results and log the arguments.

    :param args:
    :return:
    """
    # make output directoty
    out_dir = os.path.join('Output', f"{args.method}_{time.strftime('%m_%d_%Y-%H:%M:%S')}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    eprint(f'Output Directory: {out_dir}\n')

    # save args into text file
    with open(os.path.join(out_dir, 'args.txt'), 'w') as file:
        file.write(str(args.__dict__))

    return out_dir


def eprint(*args, **kwargs):
    """
        Prints to the std.err

    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def get_imagenet_classes(labels_url='https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'):
    """
        downloads the label file for imagenet

    :param labels_url:
    :return:
    """
    labels = requests.get(labels_url)
    return {int(key): value[1] for key, value in labels.json().items()}


class ImageSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, blur=False):
        """
            Loads data given a path (root_dir) and preprocess them (transforms, blur)

        :param root_dir:
        :param transform:
        :param blur:
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blur = blur
        self.transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean,std)
                 ]
        )

        eprint(f"\nLoading filenames from '{root_dir}' directory...")
        (_, _, self.filenames) = next(os.walk(root_dir))
        self.filenames = sorted(self.filenames)
        eprint(f"{len(self.filenames)} file(s) loaded.\n")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')

        if self.blur:
            resized_image = image.resize((224, 224))
            blurred = cv2.GaussianBlur(np.asarray(resized_image), (51, 51), sigmaX=50)
            blurred = Image.fromarray(blurred.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

            if self.blur:
                blurred = self.transform(blurred)

        returns = [image]

        if self.blur:
            returns.append(blurred)
        return (*returns,)

    def __len__(self):
        return len(self.filenames)


# Added function
def get_blurred_image(img_name):
    transform_fn = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean,std)
                 ]
    )

    image = Image.open(img_name).convert('RGB')
    resized_image = image.resize((224, 224))
    blurred = cv2.GaussianBlur(np.asarray(resized_image), (51, 51), sigmaX=50)
    blurred = Image.fromarray(blurred.astype(np.uint8))
    blurred = transform_fn(blurred)

    return blurred
    

def save_heatmaps(masks, images, size, index, outdir, out=224):
    """
        Save masks and corresponding overlay

    :param masks:
    :param images:
    :param size:
    :param index:
    :param outdir:
    :param out:
    :return:
    """
    masks = masks.view(-1, 1, size, size)
    up = torch.nn.UpsamplingBilinear2d(size=(out, out)).cuda()

    u_mask = up(masks)
    u_mask = u_mask.permute((0,2, 3, 1))

    # Normalize the mask
    u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))
    u_mask = u_mask.cpu().detach().numpy()

    # deprocess images
    images = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    images = images.numpy()

    for i, (image, u_mask) in enumerate(zip(images, u_mask)):

        # get the color map and normalize to 0-1
        heatmap = cv2.applyColorMap(np.uint8(255 * u_mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap / 255)
        # overlay the mask over the image
        overlay = (u_mask ** 0.8) * image + (1 - u_mask ** 0.8) * heatmap

        plt.imsave(os.path.join(outdir, f'{index+i+1}_heatmap.jpg'), heatmap)
        plt.imsave(os.path.join(outdir, f'{index+i+1}_overlay.jpg'), overlay)


def save_masks(masks, index, categories, mask_name, outdir):
    """
        Saves the generated masks as numpy.ndarrays.

    :param masks:
    :param index:
    :param categories:
    :param mask_name:
    :param outdir:
    :return:
    """
    masks = masks.cpu().detach().numpy()
    for i, (mask, category) in enumerate(zip(masks, categories), start=index):
        np.save(os.path.join(outdir, f'{mask_name}_{i+1}_mask_{category}.npy'), mask)


def save_curves(del_curve, ins_curve, index_curve, index, outdir):
    """
        Save the deletion/insertion curves for the generated masks.

    :param del_curve:
    :param ins_curve:
    :param index_curve:
    :param index:
    :param outdir:
    :return:
    """
    for i in range(len(del_curve)):
        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(index_curve, del_curve[i], color='r', label='deletion')
        ax.fill_between(index_curve, del_curve[i], facecolor='maroon', alpha=0.4)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=14)
        ax.set_yticks(np.arange(0, 1.01, 1))
        ax.legend(['Deletion'], fontsize='x-large')
        ax.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(del_curve[i])),  fontsize=14, horizontalalignment='center', verticalalignment='center')

        ax1.plot(index_curve, ins_curve[i], color='b', label='Insertion')
        ax1.fill_between(index_curve, ins_curve[i], facecolor='darkblue', alpha=0.4)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=14)
        ax1.set_yticks(np.arange(0, 1.01, 1))
        ax1.legend(['Insertion'], fontsize='x-large')
        ax1.text(0.5, 0.5, 'AUC: {:.4f}'.format(auc(ins_curve[i])), fontsize=14, horizontalalignment='center', verticalalignment='center')

        # save the plot
        plt.savefig(os.path.join(outdir, f'{index+i+1}_curves.jpg'), bbox_inches='tight', pad_inches = 0)
        plt.close()


def save_images(images, index, outdir, classes, labels):
    """
        saves original images into output directory

    :param images:
    :param index:
    :param outdir:
    :param classes:
    :param labels:
    :return:
    """
    images_ = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    for i, image in enumerate(images_):
        plt.imsave(os.path.join(outdir, f'{index+i+1}_image_{classes[labels[i].item()]}.jpg'), image.numpy())


def load_image(path):
    """
        loades an image given a path

    :param path:
    :return:
    """
    mask = Image.open(path).convert('RGB')
    mask = np.array(mask, dtype=np.float32)
    return mask / 255


def auc(array):
    """
        calculates area under the curve (AUC)

    :param array:
    :return:
    """
    return (sum(array) - array[0]/2 - array[-1]/2)/len(array)


