# How to use the COCO dataset

Execute the script `download_coco2014.sh` to download the COCO2014 dataset. This will take quite some time to execute since the dataset is relatively large. Note that we have used the val2014 dataset as a testing dataset and therefore needed to do a custom train/val split on the train2014 dataset to do validation after each training epoch. The corresponding annotations for the split will be downloaded to the `annotations` directory and loaded automatically in the code. 
