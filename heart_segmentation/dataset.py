from os import listdir
from os.path import join
from torch.utils.data import Dataset
import cv2
from PIL import Image

# this class is used to resize a numpy array to the size specified when initiated - default value is 256
class ResizeNpy(object):
    """Resize numpy arrays"""
    def __init__(self, size=256):
        self.size = size
    def __call__(self, sample):
        return cv2.resize(sample, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC)

class BasicDataset(Dataset):
    '''
    A simple dataloader
    Parameters
    ----------
    imgs_dir: string
        Path to directory of images
    masks_dir: string
        Path to direfctory of masks
    transforn: torchvision.transforms.Compose
        The transformations to be applied to each sample after it is loaded
    Attributes
    ----------
     imgs_dir: string
        Path to directory of images
    masks_dir: string
        Path to direfctory of masks
    ids: list of strings
        A list of all filenames in the data set 
    transforn: torchvision.transforms.Compose
        The transformations to be applied to each sample after it is loaded
    '''
    def __init__(self, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # keep a list of filenames - should be the same for mask and image
        self.ids = [file for file in listdir(self.masks_dir) if file.endswith('.png')] 

    def __len__(self):
        '''len(Dataset) should return the size of the dataset'''
        return len(self.ids)
   
    def __getitem__(self, i):
        '''
        This function aims to support indexing so that dataset[i] gives you the ith image/sample
        Parameters
        ----------
        i: int
            The ith sample to be loaded
        Returns
        -------
        ret: dict
            Sample of our dataset will be a dict {'image': image, 'mask': mask}
        '''
        idx = self.ids[i]
        mask_file = join(self.masks_dir, idx)
        img_file = join(self.imgs_dir, idx)
        img = Image.open(img_file)
        mask = Image.open(mask_file)
        assert img.size == mask.size, \
            'Image and mask %s should be the same size, but are %s and %s' % (idx, img.shape, mask.shape)
        img = self.transform(img)
        mask = self.transform(mask)
        return {'image': img, 'mask': mask}
