# Heart-Segmentation
## What is this?
This repo provides all necessary code for training and testing a network on echocardiography data to segment the the inner heart.

## Data

The data used to train a segmentation network includes only regions annotated with good quality of acquisition as explained in the _quality_classification_ directory. Annotations of the Lower and Upper Trace of the Inner Diameter of the heart were provided. An example of the annotated points can be seen here:

<img src="https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png" width="400">
<!---
![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png)--->

A preprocessing script was then used to create segmentation masks from these annotations in order to have a ground truth for training. Next, windows were generated from long images in order to get squared images of varying sizes, similar to as was done in _quality_classification_. An example of an image used for training along with its corresponding mask can be seen here:
<!---
![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2.png)--->
<!---
![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2_m.png)--->

Image         |  Mask
:-------------------------:|:-------------------------:
<img src="https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2.png" width="300">  |  <img src="https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/TS_Short_9_good_1_nwin2_m.png" width="300">

The data should be organized as folllows:

```
--data_path
  --img
  --mask
```

## Training

Training of a segmentation network happens with the ```run.py``` script. The script uses the class _Trainer_ defined in ```trainer.py``` where the actual training takes place. The following arguments should/can be given:

**Required arguments**

* -d: Path to train dataset. Note that you should give the root path data_path as defined above.

**Optional arguments**

* -e: Define here for how many epochs you wish to train the network. Default is 1.
* -b: Define your batch size here. Default is 1.
* -p: Define the size of images the network takes as input. Default is 256, so when images are loaded they are resized to this value.
* -l: Define your learning rate here. Default is 0.001.
* -f: If you wish to start training with a pre-trained QuickNAT define here the path to the model you wish to load. The loaded model must have been saved in the same way as it is done in the main training function of the trainer.py (for more information see also -o argument).
* -v: With this argument you can define the training-validation split. The default value is 20, meaning that 20% of the data will be set aside for validation
* -o: Define the ouptut path, i.e. where your model(s) will be saved. The models are saved in .pt format, either on the last epoch of training or if the evaluation metrics (dice score and MSE) are the best so far. The model weights are stored in a state dictionary along with other parameters, such as epoch, train loss etc. The default path in which the model is stored is the './checkpoints' directory which will be automatically created if it does not exist.

**Example run**

```
python run.py -d ./datasets/heartSegData/train -e 20 -b 4 -l 0.0001
```

## Model

The model used in this project is QuickNAT. The code is provided [here](https://github.com/ai-med/quickNAT_pytorch) and is based on [QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy] (https://www.sciencedirect.com/science/article/abs/pii/S1053811918321232) and [Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control] (https://www.sciencedirect.com/science/article/abs/pii/S1053811919302319). The only change made to the model was to reduce the number of outputs classes to one, for a binary segmentation problem.

## Testing

To test the performance of the trained model please use the ```test.py``` script. This can be used either to test a single image or for an entire test set. If a single image is given the segmentation mask is then used to calculate the Left Ventricle (LV) Inner Diameters of the heart and the LV Volume for every column in the image. These are then plotted over time and shown. Note that for the time axis a pixel resolution of 0.0008334 sec is assumed. If an entire dataset is given the average DICE score, MSE and loss are calculated for the entire test set and the first ten images, predictions and ground truth masks are saved.

The following arguments should/can be given:

**Required arguments**

* -d: As explained above this argument should either be set to the path of an image in png format or the path to a test set directory.

**Optional arguments**

* -m: The path to the model to be loaded
* -s: The size to which the images will be reshaped to before inputted to the network
* -o: The directory into which to save the first 10 images and results if a test dataset has been given as input. If a direcotry doesn't exist it is automatically created

**Example run**

```
python test.py -d ./datasets/heartSegData/test/test_img.png
```

## Results

The network was trained for 20 epochs with a batch size of four and a learning rate of 0.0001 on images of size 256x256. The trained network can be downloaded [here] (https://zenodo.org/record/3941857#.XwxgUC2w3s0). Results on train, validation and test set can be seen here:

__ | Loss | DICE | MSE 
-------| ------------- | ------------- | ------------- 
Train | 0.054 | 0.982| 0.020
Validation | 0.049 | 0.982  | 0.018
Test | 0.063 | 0.981 | 0.022 

