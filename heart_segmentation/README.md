# Heart-Segmentation
## What is this?
This repo provides all necessary code for training and testing a network on echocardiography data to segment the the inner heart.

## Data

The data used to train a segmentation network includes only regions annotated with good quality of acquisition as explained in the _quality_classification_ directory. Annotations of the Lower and Upper Trace of the Inner Diameter of the heart were provided. An example of the annotated points can be seen here:

<img src="https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png" width="48">

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png)

A preprocessing script was then used to create segmentation masks from these annotations in order to have a ground truth for training. Next windows were generated from long images in order to get squared images of varying sizes, similar to as was done in _quality_classification_. An example of an image used for training along with its corresponding mask can be seen here:

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2.png)

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2_m.png)

## Training

## Model

## Testing

