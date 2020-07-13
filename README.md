# Automatic Heart Features' Estimation from Transthoracic M-mode Echocardiography

## What is this?
This repository provides an end to end framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of useful heart features as well as writing useful features to a csv file. The framework first uses a classification network to classify the echocardiogram into regions of good and bad classification quality. Next it uses these results to write features for only good-classified regions and show these good and bad regions in the graphs. The files in the directory can be explained as:

* **quality_classification**: This includes all files needed to train a classification network to classify regions in an echocardiogram as belonging to good or bad acquisition quality regions.
* **heart_segmentation**: This includes all files needed to train a segmentation network to classify each pixel in an image as belonging or not to the inner heart. The segmentation is then used to extract features, such as the Left Ventricle Inner Diameter (LVID) in diastole and systole, the heart rate etc.
* **end2end_framework.py**: This script can be used to extract features and create graphs of a single echocardiogram. For more information on how to run this script see [Running for one echocardiogram](#Running-for-one-echocardiogram).
* **run4all.py**: This script can be used to extract features and create graphs of multiple echocardiograms saved in a single directory. For more information on how to run this script see [Running for an entire experiment](#Running-for-an-entire-experiment).
* **timeseries.py**: Helper class and functions for running the two scripts above.

## Data

The end2end framework extracts useful features and graphs from echocardiorgaphy data. The data needs to be in dicom format. All data used was of type Ultrasound Multi Frame Image and contained a total of 49 frames of overlapping regions. 

## Models
Both models for quality_classification and heart_segmentation can be trained from scratch following the steps explained in the two sub-directories. However, to instantly use the end2end_framework you can download the trained models [here](https://zenodo.org/record/3941857#.XwxgUC2w3s0). After download place them in the checkpoints dir of each directory, i.e.:

```
--quality_classification
  --checkpoints
    --quality-clas-net.pt
--heart_segmentation
  --checkpoints
    --heart-seg-net.pt
```

## Running for one echocardiogram

**Example run**
```
python end2end_framework.py -i home/datasets/cardioMice/30516265.dcm -m 40 -o 30516265
```

## Running for an entire experiment

**Example run**
```
python run4all.py -i home/datasets/cardioMice/ -m 40 -w all
```

## Results
Example of outputs of running the end2end_framework
