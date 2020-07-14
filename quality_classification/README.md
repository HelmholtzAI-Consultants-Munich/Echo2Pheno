# Quality Classification
## What is this?
This repo provides all required code for training a network on echocardiography data to classify the quality of acquisition into good and bad regions.

## Data
The data for training the model need to be images of size 350x350. In order to create such images from a raw dicom file use the script ```create_dataset.py``` When running ```create_dataset.py``` windows of the image data are created and saved in both png and npy format. An example of a training image can be seen here:

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

Note that only one or both of the directories pngs and npys need be defined but later during training you need to define with which of data you are training and the corresponding directory needs to exist.

## Training 
To train a quality acquisition classification network run train.py. The following arguments should/can be given:

**Required argument**

* -d: Path to train dataset. Note that you should give the root path "data_path" as defined above.

**Optional arguments**

* -t: The data type as explained above should be either 'png' or 'npy'. Default is 'png' 
* -a: train.py gives you the option to either train a single network or multiple netorks, i.e. bootstrapping. The default option is to run a single network but if you wish to train multiple set this argument to True
* -r: If you have set -a to True (see above) you can here define how many networks you wish to train. Default is 10
* -e: Define here for how many epochs you wish to train the network. Default is 1.
* -b: Define your batch size here. Default is 1.
* -p: Define the size of images the network takes as input. Default is 256, so when images are loaded they are resized to this value.
* -l: Define your learning rate here. Default is 0.001.
* -f: If you wish to start training with a pre-trained network define here the model you wish to load
* -v: With this argument you can define the training-validation split. The default value is 10, meaning that 10% of the data will be set aside for validation
* -o: Define the ouptut path, i.e. where you model(s) will be saved. The default is the './checkpoints' directory which will be automatically created if it does not exist

**Example run**
```
python train.py -d ./datasets/cardioMice/train -a True -e 20 -b 4 -l 0.0001 -v 20
```

## Model
The model architecture is described in ```model.py```. It includes five convolutional blocks (Convolution->ReLU->BatchNorm->MaxPooling) followed by a fully connected layer and sigmoid function. The input of the network is an image of size 256x256. If you wish to change the size of the input then you will need to adjust the number of input features in the fully connected layer at the end of the network. The output of several layers are returned from the forward function of the model class but in order to get the classification result 0 or 1 (corresponding to bad and good acquisition quality) the output of the final layer (sigmoid) needs to be rounded (as is done in train.py).

## Testing
To test the performance of the trained model or models you can run the test.py script.

**Required argument**

* -d: Path of images with which to test model. Note that you should give the root path "data_path" as defined above

**Optional arguments**

* -m: The path to the model to be loaded. See next argument for further explanation of how to set this.
* -b: Set to True if you wish to save multiple models which were trained (see -a argument above in train.py). In this case you need to specify the path in which the models are saved (e.g. checkpoints) and the script looks for files with the same names defined in train.py. Default is False so a single model will be tested and the path included the .pth file needs to be specified with -m.
* -o: If this is set to True then a csv file is created _test-results.csv_ where the name, ground truth and predicted label of each test image are written. This can later be used for evaluating the performance of the model.

_The rest of the arguments also exist in train.py and have identical explanations so are not given here._

**Example run**
```
python test.py -d ./datasets/cardioMice/test
```

