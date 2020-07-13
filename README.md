# Automatic Heart Features' Estimation from Transthoracic M-mode Echocardiography

## What is this?
This repository provides an end to end framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of useful heart features as well as writing useful features to a csv file. The framework first uses a classification network to classify the echocardiogram into regions of good and bad classification quality. Next it uses these results to write features for only good-classified regions and show these good and bad regions in the graphs. The files in the directory can be explained as:

* **quality_classification**: This includes all files needed to train a classification network to classify regions in an echocardiogram as belonging to good or bad acquisition quality regions.
* **heart_segmentation**: This includes all files needed to train a segmentation network to classify each pixel in an image as belonging or not to the inner heart. The segmentation is then used to extract features, such as the Left Ventricle Inner Diameter (LVID) in diastole and systole, the heart rate etc.
* **end2end_framework.py**: This script can be used to extract features and create graphs of a single echocardiogram. For more information on how to run this script see [Running for one echocardiogram](#Running for one echocardiogram)


## Data

## Models

## Running for one echocardiogram

## Running for an entire experiment

