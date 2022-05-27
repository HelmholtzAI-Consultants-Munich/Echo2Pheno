# Echo2Pheno Module I

## What is this?
This repository provides the code for Module I of Echo2Pheno. It is a framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of various heart features as well as exports useful features to a csv file. Module I first uses a network to classify the echocardiogram into regions of high and low acquisition quality. Then it uses a segmentation network to segment the left ventricle inner diameter (LVID) traces of the heart. For both tasks pre-trained networks are used. From the segmentation of the LVID features are then extracted, such as the LVID in diastole and systole, the heart rate etc. The quality classification results are used to write only features from high quality regions and show these high and low regions in the resulting figures. The files in the directory can be explained as:

* **quality_classification**: This includes all files needed to train and test a classification network to classify regions in an echocardiogram as belonging to high or low acquisition quality regions.
* **heart_segmentation**: This includes all files needed to train and test a segmentation network to classify each pixel in an image as belonging or not to the inner heart. The segmentation is then used to extract features, such as the Left Ventricle Inner Diameter (LVID) in diastole and systole, the heart rate etc.
* **run4single.py**: This script can be used to extract features and create figures of a single mouse echocardiogram. For more information on how to run this script see [Running for one echocardiogram](#Running-for-one-echocardiogram).
* **run4study.py**: This script can be used to extract features and create figures of multiple echocardiograms saved in a single directory, i.e. an entire mouse study. This can be used to export the necessary data for Module II of Echo2Pheno. For more information on how to run this script see [Running for an entire experiment](#Running-for-an-entire-experiment).
* **timeseries.py**: Helper class and functions for running the two scripts above.

## Data

Module I extracts useful heart related features and creates figures from echocardiorgaphy data. The data needs to be in dicom format. All data used was of type Ultrasound Multi Frame Image and contained a total of 49 frames of overlapping regions. The raw data can be made available upon request.

## Models
Both models for quality_classification and heart_segmentation can be trained from scratch following the steps explained in the two sub-directories. However, to instantly use the Module I you can download the trained models [here](https://zenodo.org/record/3941857#.XwxgUC2w3s0). After download place them in the checkpoints dir of each directory, i.e.:

```
--quality_classification
  --checkpoints
    --quality-clas-net.pth
--heart_segmentation
  --checkpoints
    --heart-seg-net.pt
```

## Running for one echocardiogram

For running the framework for a single echocardiogram the ```run4single.py``` script can be used. The following arguments should/can be given:

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
python run4single.py -i /datasets/studyA/30516265.dcm -m 40 -o 30516265
```

## Running for an entire mouse study

If you wish to run the automatic feature estimation framework for multiple mice then you can run the ```run4study.py``` script. This will recursively call the ```run4single.py```. The arguments of this script are similar to those of ```run4single.py``` with the difference that the -i argument should take the path to the directory containing the dicom files from which we wish to extract features.

**Example run**
```
python run4all.py -i datasets/studyA -m 40 -w all
```

## Results

An example of the outputs produced by Module I can be seen in the figure below. Subfigure A shows the entire echocardiogram concatenated into one long array. Above it, a color diagram shows the sigmoid output of the classification network in regions of the image (which is then classified into high and low quality with a threshold of 0.5). This output may also give us an idea of the uncertainty of the prediction. The heatmap on the left of the image maps the colors to the probability. 

![image](https://github.com/HelmholtzAI-Consultants-Munich/Echo2Pheno/blob/master/Module%20I/ModuleI_results_example.png)

In the two plots below we see the LVID signal in diastole(subplot B) and systole(subplot C). The quality of acquisition (in this case rounded to high or low) in regions of the acquisition is shown in both plots. Outliers during low acquisition quality are more likely to represent some diffuculty during acquisition (e.g. moving transduced, moving mouse) rather than an abnormality of the mouse's heart beat and can therefore be disregarded.
