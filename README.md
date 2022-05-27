# Echo2Pheno

## What is this?
This repository provides an end to end framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of various heart features as well as writing useful features to a csv file. The framework first uses a network to classify the echocardiogram into regions of good and bad classification quality. Then it uses a segmentation network to segment the left ventricle inner diameter (LVID) of the heart. For both tasks pre-trained networks are used. With the segmentation of the LVID we then extract the features such as the LVID in diastole and systole, the heart rate etc. The quality classification results are used to write only features from good-classified regions and show these good and bad regions in the graphs. The files in the directory can be explained as:

![image](https://github.com/HelmholtzAI-Consultants-Munich/Echo2Pheno/blob/master/Echo2Pheno_graphical.png)


## Installation

To install the necessary packages for this framework run:

```
pip install -r requirements.txt
```

If you are using conda first install pip by: ```conda install pip```. The above has been tested with a Python 3.7 environment.

## Data

The end2end framework extracts useful features and graphs from echocardiorgaphy data. The data needs to be in dicom format. All data used was of type Ultrasound Multi Frame Image and contained a total of 49 frames of overlapping regions. 


## Results
Example of figures and images created and saved when running ```end2end_framework.py```.

Below you can see the echocardiogram concatenated into one long array. Above it, a color diagram showing the sigmoid output of the classification network in regions of the image. Here the sigmoid output is the output of the network before it is rounded to 0 or 1 (good or bad acquisition) which may also gives us an idea of the uncertainty of the prediction. The heatmap on the right of the image maps the colors to a sigmoid value. 






