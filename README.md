# Automatic Heart Features' Estimation from Transthoracic M-mode Echocardiography

## What is this?
This repository provides an end to end framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of useful heart features as well as writing useful features to a csv file. The framework first uses a classification network to classify the echocardiogram into regions of good and bad classification quality. Next it uses these results to write features for only good-classified regions and show these good and bad regions in the graphs. The files in the directory can be explained as:

* **quality_classification**: This includes all files needed to train and test a classification network to classify regions in an echocardiogram as belonging to good or bad acquisition quality regions.
* **heart_segmentation**: This includes all files needed to train and test a segmentation network to classify each pixel in an image as belonging or not to the inner heart. The segmentation is then used to extract features, such as the Left Ventricle Inner Diameter (LVID) in diastole and systole, the heart rate etc.
* **end2end_framework.py**: This script can be used to extract features and create graphs of a single echocardiogram. For more information on how to run this script see [Running for one echocardiogram](#Running-for-one-echocardiogram).
* **run4all.py**: This script can be used to extract features and create graphs of multiple echocardiograms saved in a single directory. For more information on how to run this script see [Running for an entire experiment](#Running-for-an-entire-experiment).
* **timeseries.py**: Helper class and functions for running the two scripts above.

## Installation

To install the necessary packages for this framework run:

```
pip install -r requirements.txt
```

If you are using conda first install pip by: ```conda install pip```

## Data

The end2end framework extracts useful features and graphs from echocardiorgaphy data. The data needs to be in dicom format. All data used was of type Ultrasound Multi Frame Image and contained a total of 49 frames of overlapping regions. 

## Models
Both models for quality_classification and heart_segmentation can be trained from scratch following the steps explained in the two sub-directories. However, to instantly use the end2end_framework you can download the trained models [here](https://zenodo.org/record/3941857#.XwxgUC2w3s0). After download place them in the checkpoints dir of each directory, i.e.:

```
--quality_classification
  --checkpoints
    --quality-clas-net.pth
--heart_segmentation
  --checkpoints
    --heart-seg-net.pt
```

## Running for one echocardiogram

For running the framework for a single echocardiogram the ```end2end_framework.py``` script can be used. The following arguments should/can be given:

**Required arguments**

* -i: The path to the dicom file you wish to extract features from
* -m: The body mass in **grams** of the current mouse you wish to extract features from

**Optional arguments**

* -o: The name of the directory to save graphs and images to. Default is the current working directory
* -g: This argument's default value is _True_, meaning output graphs and images will be created and saved. if you wish to turn this functionality off set this argument to _False_.
* -w: With this argument we define whether we wish to save only statistics of features (such as median, max etc.) or all features extracted for good regions. The default value is _'all'_ so all features are extracted but can be set to _'stats'_ if we only wish to extracted statistics.
* -f: With this argument we specify the name of the csv file to write features to. The default value is _'output_all.csv'_ as the default value of -w is _'all'_ but it is suggested to set to something like _'output_stats.csv'_ if you alsi change the -w argument. If the file already exists then the new features will be appended as new rows to the file, but if the file doesn't already exist then it is automatically created.

**Example run**
```
python end2end_framework.py -i home/datasets/cardioMice/30516265.dcm -m 40 -o 30516265
```

## Running for an entire experiment

If you wish to run the automatic feature estimation framework for multiple mice then you can run the ```run4all.py``` script. This will recursively call the ```end2end_framework.py```. The arguments of this script are the similar to those of the ```end2end_framework.py``` with the difference that the -i argument should take the path to the directory containing the dicom files from which we wish to extract features.

**Example run**
```
python run4all.py -i home/datasets/cardioMice/ -m 40 -w all
```

## Results
Example of outputs of running the ```end2end_framework.py```

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_img.png)

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_heartrate.png)

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_systole.png)

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_diastole.png)

