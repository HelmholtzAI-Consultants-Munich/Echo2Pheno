# Echo2Pheno Module I

## What is this?
This repository provides the code for Module I of Echo2Pheno. It is a framework for extracting features from M-mode echocardiography data. The framework can run on one or multiple echocardiograms and creates graphs of various heart features as well as writing useful features to a csv file. The framework first uses a network to classify the echocardiogram into regions of good and bad classification quality. Then it uses a segmentation network to segment the left ventricle inner diameter (LVID) of the heart. For both tasks pre-trained networks are used. With the segmentation of the LVID we then extract the features such as the LVID in diastole and systole, the heart rate etc. The quality classification results are used to write only features from good-classified regions and show these good and bad regions in the graphs. The files in the directory can be explained as:

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

If you are using conda first install pip by: ```conda install pip```. The above has been tested with a Python 3.7 environment.

## Data

The framework extracts useful features and graphs from echocardiorgaphy data. The data needs to be in dicom format. All data used was of type Ultrasound Multi Frame Image and contained a total of 49 frames of overlapping regions. The raw data can be made available upon request.

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
Example of figures and images created and saved when running ```end2end_framework.py```.

Below you can see the echocardiogram concatenated into one long array. Above it, a color diagram showing the sigmoid output of the classification network in regions of the image. Here the sigmoid output is the output of the network before it is rounded to 0 or 1 (good or bad acquisition) which may also gives us an idea of the uncertainty of the prediction. The heatmap on the right of the image maps the colors to a sigmoid value. 

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_img.png)

The image below shows two plots. The left plot gives the heart rate of the mouse in beats per minute over time during the entire acquisition. The heart rate has been calculated for each heart beat by measuring the distance between two peaks in the beat, i.e. diastoles. Below we can again see the color diagram representing the sigmoid output of the classification network in regions of the image. On the right plot the left ventricle volume in diastole (LVID;d) is given over the heart rate. The color of the points show again the corresponding sigmoid output of the classification network. In this way points corresponding to bad acquisitions (purple, blue points) can be disregarded as they do not captured the true mouse state. Therefore, outliers during bad acquisition quality are more likely to represent some diffuculty during acquisition (e.g. moving transduced, moving mouse) rather than an abnormality of the mouse's heart beat.

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_heartrate.png)

Next we see again two plots; on the left the left ventricle volume in systole over time is represented, while on the right the left ventricle inner diameter in systole over time is shown. The quality of acquisition (in this case good or bad) in regions of the acquisition is shown in both plots.

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_systole.png)

Next, we show the same features but for diastole not systole.

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/output_diastole.png)

