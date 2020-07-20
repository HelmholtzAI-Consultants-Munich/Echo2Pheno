import os
import numpy as np
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch import from_numpy
from torchvision.transforms import Compose
import torch
import cv2
import sys

class ToNumpy(object):
    """Converts a PIL image into a numpy array"""
    def __call__(self, sample):
        return np.array(sample)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # numpy image: H x W
        # torch image: C X H X W
        im_in = from_numpy(sample/255) 
        im_in = im_in.unsqueeze(0)
        return im_in.type('torch.DoubleTensor')
    
class ResizeNpy(object):
    """Resize numpy arrays"""
    def __init__(self, size=256):
        self.size = size
    def __call__(self, sample):
        return cv2.resize(sample, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC)

class EchoDataset(Dataset):
    '''
    This class is used to load the data of the echo dataset
    Parameters
    ----------
        root_dir: string
            The root directory of the dataset. The dataset should be organized as follows:
            --root_dir
                --pngs
                    --good
                    --bad
                OR/AND
                --npys
                    --good
                    --bad
        transforn: torchvision.transforms.Compose
            The transformations to be applied to each sample after it is loaded
        data_type: string, one of two values: 'npy', 'png'
            The data can be loaded either as from a png image or from an numpy array but this needs to be specified here
    Attributes
    ----------
        root_dir: string
            The root directory of the dataset. 
        img_list: list of tuple
            A list including tuples (filename, label) of all samples in the dataset
        transforn: torchvision.transforms.Compose
            The transformations to be applied to each sample after it is loaded
        data_type: string, one of two values: 'npy', 'png'
            The data can be loaded either as from a png image or from an numpy array but this needs to be specified here
    '''
    def __init__(self, root_dir, transform=None, data_type='npy'):
        self.data_type = data_type
        self.root_dir = root_dir
        if self.data_type == 'png':
            good_dir = os.path.join(self.root_dir, 'pngs', 'good') + '/*.png'
            bad_dir = os.path.join(self.root_dir, 'pngs', 'bad') + '/*.png' 
        elif self.data_type == 'npy':
            good_dir = os.path.join(self.root_dir, 'npys', 'good') + '/*.npy' # '/*.png'
            bad_dir = os.path.join(self.root_dir, 'npys', 'bad') + '/*.npy'   # '/*.png'          
        good_list = glob.glob(good_dir)
        bad_list = glob.glob(bad_dir)
        good_list = [(item, 1) for item in good_list]
        bad_list = [(item, 0) for item in bad_list]
        self.img_list = good_list + bad_list
        self.transform = transform

    def __len__(self):
        '''len(Dataset) should return the size of the dataset (number of images in both folders 'good' and 'bad')'''
        size = len(self.img_list)
        return size 
        
    def __getitem__(self, idx):
        '''
        This class aims to support indexing so that dataset[i] gives you the ith image/sample
        Parameters
        ----------
            idx: int
                The ith sample to be loaded
        Returns
        -------
            ret: dict
                Sample of our dataset will be a dict {'image': image, 'label': label, 'filename': filename of sample}
        '''
        img_tuple = self.img_list[idx]
        if self.data_type == 'png':
            sample = Image.open(img_tuple[0])
            #sample = np.array(img)
        else:
            sample = np.load(img_tuple[0]) 
        label = img_tuple[1]
        sample = self.transform(sample)
        ret = {'image': sample, 'label': label, 'filename': img_tuple[0].split('/')[-1]} 
        return ret 