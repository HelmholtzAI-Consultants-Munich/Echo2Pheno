# Heart-Segmentation
## What is this?
This repo provides all necessary code for training and testing a network on echocardiography data to segment the the inner heart.

## Data

The data used to train a segmentation network includes only regions annotated with good quality of acquisition as explained in the _quality_classification_ directory. Annotations of the Lower and Upper Trace of the Inner Diameter of the heart were provided. An example of the annotated points can be seen here:

<img src="https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png" width="400">
<!---
![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/annotation_example.png)--->

A preprocessing script was then used to create segmentation masks from these annotations in order to have a ground truth for training. Next windows were generated from long images in order to get squared images of varying sizes, similar to as was done in _quality_classification_. An example of an image used for training along with its corresponding mask can be seen here:
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
* -f: If you wish to start training with a pre-trained network define here the model you wish to load
* -v: With this argument you can define the training-validation split. The default value is 20, meaning that 20% of the data will be set aside for validation
* -o: Define the ouptut path, i.e. where you model(s) will be saved. The default is the './checkpoints' directory which will be automatically created if it does not exist

**Example run**

```
python run.py -d ./datasets/heartSegData/train -e 20 -b 4 -l 0.0001
```

## Model

The model used for training was QuickNAT. The code is provided [here](https://github.com/ai-med/quickNAT_pytorch).

## Testing

