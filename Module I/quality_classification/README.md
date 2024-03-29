# Quality Classification
## What is this?
This repo provides all necessary code for training and testing a network on echocardiography data to classify the quality of acquisition into good and bad regions.

## Data
The data for training the model need to be squared images of any size. They are reiszed as a pre-processing step in ```train.py```. In order to create such images for training from raw dicom files use the script ```create_dataset.py```. This will create squared images including about three heart beats in each image. When running ```create_dataset.py``` windows of the very long echocardiography image are created and saved in both png and npy format. To create the dataset a csv file also needs to be provided where the start and end times (in seconds) of good and bad acquisition quality regions are given. An example can how such a file should look can be seen here:

Recording | Good-Start | Good-End | Bad-Start | Bad-End
-------| ------------- | ------------- | ------------- | ------------- |
TS_short_1 | 3.4 | 3.9 | 2 | 2.6
Ts-short_2 | 2.1 | 2.9 | 0 | 1.3
TS_short_3 | 1.2 | 1.8 | 0 | 0.9
TS_short_4 | 0.9 3.0 | 1.5 4.3 | 1.6 | 3.1
TS_short_5 | 1.6 | 3.7 | 0.4 | 1.6

As you can see from the example below in recording _TS_short_4_ multiple good or bad quality regions can be defined. Then the times need to be seperated by a single space. An example of an image created for training can be seen here:

![image](https://github.com/HelmholtzAI-Consultants-Munich/Automatic-Heart-Features-Estimation-from-Transthoracic-M-mode-Echocardiography/blob/master/images/train-acquisition-example.png)

The data should be organized as folllows:
```
--data_path
  --pngs
    --good 
    --bad
  --npys
    --good
    --bad
```

This is done automatically in the script where only the _data_path_ needs to be set. Note that only one or both of the directories _pngs_ and _npys_ need be defined. However, later, during training you need to define with which type of data you are training and the corresponding directory needs to exist.

## Training 
To train a quality acquisition classification network run ```train.py```. The following arguments should/can be given:

**Required argument**

* -d: Path to train dataset. Note that you should give the root path _data_path_ as defined above.

**Optional arguments**

* -t: The data type as explained above should be either _png_ or _npy_. Default is _png_. 
* -a: ```train.py``` gives you the option to either train a single network or multiple netorks, i.e. bootstrapping. The default option is to run a single network but if you wish to train multiple set this argument to _True_.
* -r: If you have set -a to _True_ (see above) you can here define how many networks you wish to train. Default is 10.
* -e: Define here for how many epochs you wish to train the network. Default is 1.
* -b: Define your batch size here. Default is 1.
* -p: Define the size of images the network takes as input. Default is 256, so when images are loaded they are resized to this value.
* -l: Define your learning rate here. Default is 0.001.
* -f: If you wish to start training with a pre-trained network define here the model you wish to load.
* -v: With this argument you can define the training-validation split. The default value is 10, meaning that 10% of the data will be set aside for validation
* -o: Define the ouptut path, i.e. where you model(s) will be saved. The models are saved in .pth format on the last epoch of training. The default is the './checkpoints' directory which will be automatically created if it does not exist.

**Example run**
```
python train.py -d ./datasets/cardioMice/train -a True -e 20 -b 4 -l 0.0001 -v 20
```

## Model
The model architecture is described in ```model.py```. It includes five convolutional blocks (Convolution->ReLU->BatchNorm->MaxPooling) followed by a fully connected layer and sigmoid function. The input of the network is an image of size 256x256. If you wish to change the size of the input then you will need to adjust the number of input features in the fully connected layer at the end of the network. The output of several layers are returned from the forward function of the model class but in order to get the classification result 0 or 1 (corresponding to bad and good acquisition quality) the output of the final layer (sigmoid) needs to be rounded (as is done in ```train.py```).

## Testing
To test the performance of the trained model or models you can run the ```test.py``` script. The following arguments should/can be given:

**Required argument**

* -d: Path of images with which to test model. Note that you should give the root path _data_path_ as defined above

**Optional arguments**

* -m: The path to the model to be loaded. See next argument for further explanation of how to set this.
* -b: Set to _True_ if you wish to test multiple models which were trained (see -a argument above in ```train.py```). In this case you need to specify the directory path in which the models are saved (e.g. checkpoints) and the script looks for files with the same names defined during saving in ```train.py```. Default is _False_ so a single model will be tested and the path including the .pth file needs to be specified with -m.
* -o: If this is set to _True_ then a csv file is created _test-results.csv_ where the name, ground truth and predicted label of each test image are written. This can later be used for evaluating the performance of the model.

_The rest of the arguments also exist in train.py and have identical explanations so are not given here._

**Example run**
```
python test.py -d ./datasets/cardioMice/test
```

## Results
During training 10 networks were trained for 15 epochs with a batch size of four and a learning rate of 0.0001 for images of size 256x256. The trained network can be downloaded [here] (https://zenodo.org/record/3941857#.XwxgUC2w3s0). For testing all 10 networks were used; their sigmoid outputs were summed and the final average sigmoid was rounded to either zero or one with a threshold of 0.5. The performance of this averaged prediction on the test set is given here:

Accuracy | FPR | FNR | F1 Score 
-------| ------------- | ------------- | ------------- 
96.428 | 0.118 | 0.007 | 0.976

