# What You See is What You Classify: Black Box Attributions

## Table of Contents
* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Getting Started](#getting-started)
  * [Install Packages](#install-packages)
  * [Download the Data](#download-the-data)
* [Training and Testing the Explainer](#training-and-testing-the-explainer)
  * [Training the Explainer](#training-the-explainer)
  * [Reproducing our Results](#reproducing-our-results)
  * [Testing and Saving the Masks](#testing-and-saving-the-masks)
  * [Tensorboard Logging](#tensorboard-logging)
* [Evaluation](#evaluation)
  * [Producing Segmentation Masks](#producing-segmentation-masks)
  * [Computing the Segmentation Metrics](#computing-the-segmentation-metrics)
  * [Computing the Class-specific Metrics](#computing-the-class-specific-metrics)

## About the Project

Identifying the parts of an input image that contribute to the classification output of a deep network helps explaining its behaviour. This is challenging due to the black-box nature of such networks. Most existing approaches find such attributions either using activations and gradients, or by repeatedly perturbing the input. We instead address this challenge by training another deep network, the Explainer, to locate attributions for the pre-trained black-box classifier, the Explanandum, for a given input image. Our approach produces masks that are more boundary-precise and sharply delineated as compared to the saliency maps generated by existing methods. Unlinke most existing approaches, ours is capable of directly generating a separate mask for each class label in a multi-class setting. Additionally, our approach is very efficient because it takes a single forward pass through the Explainer to generate the masks. We show that our attributions are superior both visually and quantitatively  by evaluating them on the [Pascal VOC-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and [COCO-2014](https://cocodataset.org/#home) datasets.

![alt text](examples/Intro%20Teaser.png)

## Folder Structure
```
├── README.md
├── environment.yml                                   - environment to run the code
├── datasets
│   ├── COCO2014                                      - will hold COCO-2014 data after downloading
│   │   └── download_coco2014.sh                      - download script for COCO-2014 data (will create new directories)
│   └── VOC2007                                       - will hold VOC-2007 data (automatically downloaded)
├── examples                                          - example images for this README
├── notebooks                                         - Jupyter notebooks for evaluations
├── src                                             
│   ├── config_files                                  
│   │   ├── explainer_training                        - contains config files to train the explainer on pretrained classifiers
│   │   └── misc                                      - contains config files to train a classifier & test a pretrained explainer
│   ├── data
│   │   ├── dataloader.py                             - defines VOC, COCO and CUB data modules
│   │   └── dataset.py                                - defines custom COCO and CUB datasets
│   ├── evaluation
│   │   ├── evaluate.py                               - main evaluation script after all needed masks are stored
│   │   ├── evaluate_segmentations.py                 - old script to evaluate masks against segmentation groundtruths
│   │   ├── explainer.py                              - generates Explainer masks for evaluation
│   │   ├── extremal perturbations.py                 - generates Extremal Perturbations masks for evaluation
│   │   ├── generate_coco_segmentations.py            - saves a number of randomly selected coco segmentation groundtruths to files
│   │   ├── grad_cam.py                               - generates Grad-CAM masks for evaluation
│   │   ├── guided_backprop.py                        - generates Guided Backpropagation masks for evaluation
│   │   ├── print_mean_scores.py                      - script to print out mean scores for values in results.npz file
│   │   ├── rise.py                                   - generates RISE masks for evaluation
│   │   ├── rt_saliency.py                            - generates Real-Time Saliency masks for evaluation
│   │   ├── eval_utils
│   │   │   ├── assessment_metrics.py                 - defines metrics for evaluation
│   │   │   ├── compute_masks.py                      - helper function to produce EP and GP masks
│   │   │   └── compute_scores.py                     - functions to evaluate metrics for all methods and settings                             
│   ├── models
│   │   ├── classifier.py                             - defines VGG-16 and ResNet-50 classifiers
│   │   ├── explainer.py                              - defines the explainer architecture based on the Deeplab-v3 model
│   │   ├── explainer_classifier.py                   - defines the full model consisting of the explainer with an associated classifier
│   │   ├── explainer_classifier.py                   - defines the adapted explainer/classifier architecture for the Real-Time Saliency loss functions 
│   │   └── interpretable_fcnn.py                     - defines the self-explainer model
│   ├── utils
│   │   ├── argparser.py                              - defines command line argument parser
│   │   ├── helper.py                                 - various helper functions for model training
│   │   ├── image_utils.py                            - functions for displaying and saving generated images and masks
│   │   ├── loss.py                                   - defines loss functions used for training
│   │   ├── metrics.py                                - defines metrics to be used during training and testing
│   │   ├── rtsal_helper.py                           - helper functions for the Real-Time Saliency Architecture
│   │   └── rtsal_loss.py                             - defines loss functions for the Real-Time Saliency Architecture
│   └── main.py                                       - main script to train and test Explainer
```

## Getting Started
### Install Packages

All required packages can be found in the environment.yml file. They are most easily installed with [conda/miniconda](https://docs.conda.io/en/latest/miniconda.html), where a new environment can be easily created like this: 
```bash
conda env create -f environment.yml 
```
After that, activate the new environment (needs to be done everytime the shell is reloaded):
```bash
conda activate pytorch_gpu
```
Should you get any errors during the environment setup, make sure to set
```bash
conda config --set channel_priority false
```
and then try again.

### Download the Data

The Pascal VOC-2007 training, validation and training sets should be automatically downloaded to [`datasets/VOC2007/`](datasets/VOC2007) after you run the `main.py` script with the `--dataset='VOC'` option (default). If that does not work, the training and validation data is available [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and the testing data is available [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). 

For the COCO-2014 dataset, we have provided a script to download the data with our custom training and validation splits. Go to the [`datasets/COCO2014/`](datasets/COCO2014/) directory and execute the `download_coco2014.sh` script there before running the `main.py` script with the `--dataset='COCO'` option.  

For either dataset, the `dataset` folder is set as the default base directory in the code. If you move (or want to download) the datasets to a different place, make sure to point the code to the new directory as follows (additional arguments omitted, for more information on how to run the code, see the [Training and Testing the Explainer](#training-and-testing-the-explainer) section:
```bash
python main.py --data_base_path="<path to your directory containing the VOC2007 or COCO2014 subdirectories>"
```

The `--data_base_path` can either be an absolute path or a relative path from the `src` directory (which contains the `main.py` script).


## Training and Testing the Explainer
### Training the Explainer

Once the dependencies are installed and the datasets downloaded, one can run the main script with any number of arguments that are specified in [`src/utils/argparser.py`](src/utils/argparser.py). To get an overview over all arguments, you can run `python main.py -h`. As long as the data is downloaded and stored in the `datasets` directory, the code should execute without specifying any arguments and start to train our Explainer model. However, this will likely not lead to any satisfying results as we expect the Explainer to be trained with a pretrained classifier. To do that you would first need to run: 
```bash
python main.py --model_to_train="classifier"
```
This will store a `.ckpt` file in the `src/checkpoints/` folder. Such `.ckpt` files are used by PyTorch Lightning to store the weights of the classifier. After the classifier has been fully trained (i.e. reaching a stopping condition), this `.ckpt` file could then be loaded to train the Explainer:
```bash
python main.py --model_to_train="explainer" --classifier_checkpoint="<path to your .ckpt file from the previous step>"
```
Note that following this procedure would result in an Explainer model that would likely already start to work, it is still not optimized in terms of the hyperparameters used. Moreover, it would only use the default VOC-2007 dataset and the default VGG-16 classifier architecture. To reproduce our results with our choice of hyperparameters and to run the code with different settings, see the section [below](#reproducing-our-results).

### Reproducing our Results

Within the subdirectories of the [`src/config_files/`](src/config_files/) directory, you will find various config files to run the script with our settings and hyperparameters. For example, with the config files in the [`src/config_files/classifier_training/`](src/config_files/classifier_training/) subdirectory, the user could train a classifier, e.g. as follows:
```bash
python main.py -c "config_files/classifier_training/classifier_vgg16_voc_training.cfg"
```
As you can see, with their name the config files describe which classifier and dataset they will be using. 

On the other hand, the config files in the [`src/config_files/explainer_training/`](src/config_files/explainer_training) directory can be used to train an Explainer model. However, note that for the Explainer to work best, you need to edit the corresponding config file and assign a `.ckpt` file to the `classifier_checkpoint` argument, as was described in the section [above](#training-the-explainer). After that, you could start the training process for the corresponding Explainer (assuming a VGG-16 classifier and the VOC-2007 dataset):
```bash
python main.py -c "config_files/explainer_training/explainer_vgg16_voc_training.cfg"
```
Should you want to use your own pretrained classifier, replace the corresponding lines in the `setup_classifier()` function in the [`src/models/explainer_classifier.py`](src/models/explainer_classifier.py) file and assign your classifier to the `self.classifier` field.

### Testing and saving the Masks

If you run the code as described in the sections above, it will also do a test epoch after training finishes and store the test metrics to a file named `<model>_<classifier_type>_<dataset>_test_metrics.txt`. However, it will not store any masks or masked images, unless you set the following argument parser options: `--test_batch_size=1`, `--save_path='<some path, where the resulting image files should be stored>'`(default = `'./results/'`), and either `--save_masks=True` or `--save_masked_images=True`, or both. 

You can also train the models without these options and then load them again to save the masks and/or masked images. For that, we have provided additional config files in [`src/config_files/testing_and_mask_saving`](src/config_files/testing_and_mask_saving), which you can execute like the ones above. E.g.:
```bash
python main.py -c "config_files/testing_and_mask_saving/explainer_vgg16_voc_test_and_save_masks.cfg"
``` 
Just make sure to first edit them and point the `explainer_classifier_checkpoint` argument to the `.ckpt` file of the trained Explainer, which you want to produce results for. Note that the argument is called like that because the checkpoints will contain the weights of both the Explainer and the corresponding classifier it was built on.

### Tensorboard Logging

If you wish to use tensorboard, just set `--use_tensorboard_logger=True` for your run. After that you can use tensorboard as follows:

```bash
tensorboard --logdir=./tb_logs/fit/<directory corresponding to your run (named according to date and time of the run)>
```
Note that if you use a logger, the model weights (i.e. the .ckpt file) will also be stored in this directory (specifically in: `./tb_logs/fit/<date_and_time_of_run>/NN Explainer/checkpoints/`). 

## Evaluation
![alt text](examples/Segmentation%20Comparisons.png)

### Producing Segmentation Masks

In the folder [`src/evaluation`](src/evaluation/), we have added several python scripts to produce the masks required for evaluation against the groundtruth segmentation masks. There is one script for every of the following methods:
* Explainer (ours)
* [Grad-CAM](https://arxiv.org/abs/1610.02391)
* [RISE](https://arxiv.org/abs/1806.07421)
* [Extremal Perturbations](https://arxiv.org/abs/1910.08485)
* [Guided Backpropagation](https://arxiv.org/abs/1412.6806)
* [iGOS++](https://arxiv.org/abs/2012.15783)
* [Real Time Image Saliency](https://arxiv.org/abs/1705.07857)

In the following subsections, we explain how we have adapted the code for the different methods and how to execute the scripts to retrieve the segmentation masks. Note that all other methods produce only one mask per object class instead of an aggregate mask. We have therefore merged the individual class masks for each object in a specific image to such aggregate masks in the same way it is done for our Explainer. To give an example, if an image contains a person and a dog, we will compute a mask for the "person" class as well as a mask for the "dog" class and then merge them together by taking the maximum value for each pixel over both individual masks to get our aggregate mask. 

Also note that due to the long execution time of some of the methods, we have not produced and evaluated masks for all ~41K COCO-2014 validation images (which we use a test set). Therefore, we also provide a short python script to retrieve the segmentation groundtruths for a random subset of the dataset. Make sure to execute the [`src/evaluation/generate_coco_segmentations.py`](src/evaluation/generate_coco_segmentations.py) script before continuing with the next steps if you wish to also evaluate the methods on COCO-2014. You can change lines 11-13 of the script to adapt it to your settings:
```python
############################## Change to your settings ##############################
test_annotations_path = '../../datasets/COCO2014/annotations/instances_val2014.json'
num_segmentations = 2000
seg_dir = './coco_segmentations/'
#####################################################################################
```

#### Explainer

While in the [Training and Testing the Explainer](#training-and-testing-the-explainer) section above, we have already explained how to produce all the masks for our Explainer model, we provide an additional script to save exactly those masks that are used for the evaluation. Make sure to edit the lines at the top of the [`src/evaluation/explainer.py`](src/evaluation/explainer.py) script according to your settings before running it. Note that setting `mode = 'seg'` will produce the masks needed for the segmentation evaluation (see [Computing the Segmentation Metrics](#computing-the-segmentation-metrics)), while setting `mode = 'classes'` will produce the masks for the class specific evaluation (see [Computing the Class-specific Metrics](#computing-the-class-specific-metrics)). 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_explainer/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

#### Grad-CAM

To retrieve the Grad-CAM masks, make sure to edit the lines at the top of the [`src/evaluation/grad_cam.py`](src/evaluation/grad_cam.py) script before running it. Note that setting `mode = 'seg'` will produce the masks needed for the segmentation evaluation (see [Computing the Segmentation Metrics](#computing-the-segmentation-metrics)), while setting `mode = 'classes'` will produce the masks for the class specific evaluation (see [Computing the Class-specific Metrics](#computing-the-class-specific-metrics)). 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_grad_cam/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

Note that we have taken the underlying code for Grad-CAM from [this repository](https://github.com/jacobgil/pytorch-grad-cam).

#### RISE

To retrieve the RISE masks, make sure to edit the lines at the top of the [`src/evaluation/rise.py`](src/evaluation/rise.py) script before running it. Note that setting `mode = 'seg'` will produce the masks needed for the segmentation evaluation (see [Computing the Segmentation Metrics](#computing-the-segmentation-metrics)), while setting `mode = 'classes'` will produce the masks for the class specific evaluation (see [Computing the Class-specific Metrics](#computing-the-class-specific-metrics)). 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_rise/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps. 

Note that we have adapted the underlying code for RISE from [the Torchray package](https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/rise.py).

#### Extremal Perturbations

To retrieve the Extremal Perturbations masks, make sure to edit the lines at the top of the [`src/evaluation/extremal_perturbations.py`](src/evaluation/extremal_perturbations.py) script before running it. 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_extremal_perturbations/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

For the Extremal Perturbations method, we have taken the offical code from [the Torchray package](https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/extremal_perturbation.py).

#### Guided Backpropagation

To retrieve the Guided Backpropagation masks, make sure to edit the lines at the top of the [`src/evaluation/guided_backprop.py`](src/evaluation/guided_backprop.py) script before running it. 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_guided_backprop/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

For the Guided Backpropagation method, we have taken the code from [the Torchray package](https://github.com/facebookresearch/TorchRay/blob/master/torchray/attribution/guided_backprop.py).

#### iGOS++

To retrieve the iGOS++ masks, go to the [`src/evaluation/IGOS_pp/`](src/evaluation/IGOS_pp) directory and make sure to edit the lines at the top of the [`src/evaluation/IGOS_pp/main.py`](src/evaluation/IGOS_pp/main.py) script before running it. 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_igos_pp/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

The code for the iGOS++ baseline has been taken from the [official repository](https://github.com/khorrams/IGOS_pp) and slightly adapted to allow for a fair comparison. Specifically, instead of producing masks for the predicted class by the classifier, we produce a mask for each object in the image **according to the groundtruth** and merge the masks in the same way as it was done for all the other methods. Note that we have left all hyperparameters at their default settings, as provided by the [`src/evaluation/IGOS_pp/args.py`](src/evaluation/IGOS_pp/args.py) file.

#### Real Time Image Saliency

Evaluating the Real Time Image Saliency method was more difficult for us since we could not directly train their architecture on our classifiers in this multi-class setting. To still get a reasonable comparison with their method, we have adapted our Explainer architecture to take their loss formulation [(see their code)](https://github.com/PiotrDabkowski/pytorch-saliency/blob/master/sal/saliency_model.py). Initially, we have taken the same hyperparameters for this loss formulation that have been given in [their paper](https://arxiv.org/abs/1705.07857). However, as those first experiments led to bad results, we have tuned the λ2 parameter from 10^-3 to 1.0. This led to a significant improvement in the results but we could not perform further hyperparameter optimization.

You can train the adapted Explainer with new loss formulation like the normal Explainer (see [Training and Testing the Explainer](#training-and-testing-the-explainer)). Just make sure to set the `--model_to_train=rtsal_explainer` when running the [`main.py`](src/main.py) script. After having the fully trained model saved in a `.ckpt` file, make sure to edit the lines at the top of the [`src/evaluation/rt_saliency_test.py`](src/evaluation/rt_saliency.py) script before running it. Note that setting `mode = 'seg'` will produce the masks needed for the segmentation evaluation (see [Computing the Segmentation Metrics](#computing-the-segmentation-metrics)), while setting `mode = 'classes'` will produce the masks for the class specific evaluation (see [Computing the Class-specific Metrics](#computing-the-class-specific-metrics)). 

After running the script, the masks will be stored in the `masks/<dataset>_<classifier_type>_rt_saliency/` folder within the [`src/evaluation/`](src/evaluation) directory (`<dataset>` and `<classifier_type>` correspond to the settings you edit at the top of the script). Please do not rename this folder if you wish to evaluate the metrics for these masks in the later steps.

### Computing the Segmentation Metrics

Once we have computed the masks for all datasets, classifiers, and methods that we want to evaluate, we can edit the following lines in the [`src/evaluation/evaluate.py`](src/evaluation/evaluate.py) script before running it:
```python
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

methods = ["extremal_perturbations",  "grad_cam", "rise", "explainer", 
          "rt_saliency", "guided_backprop", "0.5", "0", "1", "perfect"]
#################################################################################################################################
```
For all given options, this script will compute several metrics (see [`src/utils/assessment_metrics.py`](src/utils/assessment_metrics.py)) and save them to a file called `results.npz` in the [`src/evaluation/`](src/evaluation) directory. Note that the methods "0", "0.5", and "1", just correspond to masks with all values being equal to 0, 0.5, and 1, respectively. The method "perfect" produces masks equal to the segmentation groundtruths. These "methods" just serve as baselines for comparison with the actual methods.

Finally, if you would like to print out the mean scores for the values in the `results.npz` file, you can run the [`src/evaluation/print_mean_scores.py`](src/evaluation/print_mean_scores.py) script.

### Computing the Class-specific Metrics

Once we have computed the masks for all classifiers and methods that we want to evaluate for the class specific metrics, we can just execute the lines in the [`notebooks/evaluate_class_scores.ipynb`](notebooks/evaluate_class_scores.ipynb) notebook to produce the confusion matrices according to the specified classes. Note that this evaluation is only done for a subset of classes in the VOC-2007 dataset. You can edit the [`src/evaluation/explainer.py`](src/evaluation/explainer.py), [`src/evaluation/grad_cam.py`](src/evaluation/grad_cam.py), [`src/evaluation/rise.py`](src/evaluation/rise.py), [`src/evaluation/rt_saliency.py`](src/evaluation/rt_saliency.py) scripts to set the classes for which you want to compute this evaluation for. When running this notebook, make sure to specify the same set of classes for it to work.