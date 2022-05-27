import os
import sys
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
from heart_segmentation.utils import dice_score_binary
import heart_segmentation.constants as constants
from heart_segmentation.quicknat import QuickNat
from heart_segmentation.dataset import BasicDataset, ResizeNpy

def return_features(mask, orig_size):
    """
    This function calculates the Left Ventricle Inner Diameter (LVID) and Left Ventricle Volume (LV Vol) from a segmentation mask. This is done for 
    each time instance, i.e. every column in the image. By counting the number of pixels in the mask corresponding to the inner heart and translating
    this to mm we get the LVID measurement for each time instance. Then using the Teichholz formula we also compute corresponding volumes.
    Parameters
    ----------
        mask: numpy array
            The segmentation mask
        orig_size: int
            They size of the image before it was resized to fit the network
    Returns
    -------
        heart_vols: list of floats
            List of LV Vol for each column in the segmentation mask
        lvids: list of floats
            List of LVID for each column in the segmentation mask
    """
    pixel_phys = constants.pixel_phys # the resolution of each pixel in the y axis in mm
    heart_vols = []
    lvids = []
    # for each column in image
    for i in range(mask.shape[1]):
        column = mask[:,i]
        # count the number of occurences of 0s and 1s
        unique, counts = np.unique(column, return_counts=True)
        uncounts = dict(zip(unique, counts))
        try:
            # the number of 1s (in pixels) will be the LVID
            heart_len = uncounts[1.0]
            # get ration of original/resized image (resized to input into network)
            ratio = orig_size[1]/mask.shape[1]
            # go from pixels to mm
            real_heart_len = heart_len * ratio * pixel_phys
            # Teichholz formula = 7/2.4+LVID * LVID^3
            heart_vol = (7 * real_heart_len**3)/(2.4+real_heart_len)
            lvids.append(real_heart_len)
            heart_vols.append(heart_vol)
        # if for this image there is no heart beat
        except KeyError:
            lvids.append(0)
            heart_vols.append(0)
    return heart_vols, lvids

def _prepare_img(img, size):
    """
    This function prepares a single image to be inputted to the network as would be done by the dataloader for a dataset
    Parameters
    ----------
        img: numpy array
            The image of which we wish to compute the segmentation mask
        size: int
            They size to which we need to resize images, i.e. the input size the network accepts
    Returns
    -------
        img: torch.Tensor
            The image tranformed so that it can now be inputted to the network            
    """
    # resize image and convert to tensor
    transforms = Compose([ResizeNpy(size), ToTensor()])
    img = transforms(img)
    # add an extra dimension - normally batch size
    img = torch.unsqueeze(img, 0)
    return img

def predict_single(net, device, img_path, size):
    """
    This function makes a prediction on a single image
    Parameters
    ----------
        net: model.QuickNat
            The network used to generate segmentation mask
        device: torch.device
            The currently running divice, e.g. cpu, gpu0  
        img_path: string or image
            This can either be a string of the image path in which case the image is loaded and then convert to a numpy array
            OR
            It can be a numpy array containing the image
        size: int
            They size to which we need to resize images, i.e. the input size the network accepts
    Returns
    -------
        mask_pred: numpy array
            The segmentation mask
        orig_size: 
            The size of the original image before it is resized to be sent to the network
    """
    sig = torch.nn.Sigmoid()
    # if the image path is given load the image from disk
    if type(img_path) == str:
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
    else:
        img = img_path
    # apply transforms and convert to torch
    orig_size = img.shape
    img = _prepare_img(img, size)
    img = img.to(device=device, dtype=torch.float32)
    # make prediction
    mask_pred = net(img)  
    mask_pred = sig(mask_pred)
    mask_pred = torch.round(mask_pred)
    #save_image(mask_pred[0], 'mask.png')
    mask_pred = torch.squeeze(mask_pred)
    mask_pred = mask_pred.detach()
    mask_pred = mask_pred.cpu().numpy()
    return mask_pred, orig_size

def plot_graph(lv, type):
    """
    This function plots either the LVIDs or LV Vols over time
    Parameters
    ----------
        lv: list of floats
            Either a list of LVIDs or of LV Vols ; this is defined by type
        type: string, can be either 'LVID' or 'VOL'
            Defines what is included in lv list          
    """
    # convert pixels in x axis to seconds
    pixel_sec = constants.pixel_sec
    # calculate total time represented in image
    tot_time = len(lv)*pixel_sec
    time = np.linspace(0, tot_time, num=len(lv))
    lv = np.array(lv)
    # plot
    plt.figure()
    plt.xlabel('Time(sec)')
    if type=='LVID':
        plt.plot(time,lv*10**(3))
        plt.ylabel('LVID(mm)')
        plt.title('LV Inner Dimensions Estimation')
    else:
        plt.plot(time,lv*10**(9))
        plt.ylabel('LV Vol(mm^3)')
        plt.title('LV Volume Estimation')
    plt.show()

def predict_all(net, device, test_loader, save_dir):
    """
    This function predicts segmentation masks for a test set, calculates and prints average mse and Dice score results on the test set
    Parameters
    ----------
        net: model.QuickNat
            The network used to make prediction
        device: torch.device
            The currently running divice, e.g. cpu, gpu0  
        test_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the test set 
        save_dir: string
            Path where the first 10 images, ground truth masks and predicted segmentation masks should be saved
    """
    # initialize evaluation metrics etc
    bce_loss = torch.nn.BCELoss()
    sig = torch.nn.Sigmoid()
    mse = torch.nn.MSELoss()
    dice_eval = 0
    mse_eval = 0
    loss = 0
    # set model into evaluation mode for predictions
    net.eval() 
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # load image and  mask and cast to device
            img = batch['image']
            mask = batch['mask']            
            img = img.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device)
            # make prediction
            mask_pred = net(img)  
            mask_pred = sig(mask_pred)
            # calculate BCE loss
            loss += bce_loss(mask_pred, mask)  
            # round sigmoid output to 0 (bnackground) or 1 (heart)
            mask_pred = torch.round(mask_pred)  
            # calculate evalutation metrics    
            mse_eval += mse(mask, mask_pred)
            dice_avg = dice_score_binary(mask_pred, mask)
            dice_eval += dice_avg
            # save the first 10 images
            if idx<10:
                img_name = 'image_' + str(idx) + '.png'
                mask_name = 'mask_' + str(idx) + '.png'
                pred_name = 'pred_' + str(idx) + '.png'
                save_image(img[0], os.path.join(save_dir, img_name))
                save_image(mask[0], os.path.join(save_dir, mask_name))
                save_image(mask_pred[0], os.path.join(save_dir, pred_name))
            # print notification every 10% of testing
            if (idx+1) % int(0.1*len(test_loader)) == 0:
                print('Done for ', (idx+1)*10//int(0.1*len(test_loader)), '% of testing')
    # print results
    print("Measured loss: ", loss.item()/len(test_loader))
    print("Average DICE: ", dice_eval.item()/len(test_loader))
    print("Average MSE: ", mse_eval.item()/len(test_loader))


def get_args():
    '''
    Required arguments
    ------------------
        -d: The path to the test dataset, or to a single image
    Optional arguments
    ------------------
        -s: The size to which images need to be resized during load. Default is 256
        -m: The path to the model you wish to test
        -o: The path to save the first ten images when testing the entire test set
    '''
    parser = argparse.ArgumentParser(description='Predict segmentation masks for test images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/heart-seg-net.pt', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--datapath', '-d', required=True,
                        help='Specify path of input images or single image full path name')
    parser.add_argument('--size', '-s', metavar='ImgS', type=int, default=256,
                        help='Specufy image width and height')
    parser.add_argument('--output', '-o', default='results', 
                        help='Specify output dir to save image results')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # get project directory and data directories
    project_path = os.path.dirname(os.path.realpath(__file__))
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device, ", device)
    # load model
    params = {'num_channels':1,
                'num_class':1,
                'num_filters':64,
                'kernel_h':5,
                'kernel_w':5,
                'kernel_c':1,
                'stride_conv':1,
                'pool':2,
                'stride_pool':2,
                'se_block': "NONE",
                'drop_out':0.2}
    net = QuickNat(params) 
    print("Loading model: ", args.model)
    net = net.to(device=device)
    state_dict = torch.load(os.path.join(project_path,args.model), map_location=device)
    net.load_state_dict(state_dict['net_state_dict'])
    print("Model loaded !")

    # if a single file has been given make heart estimation and plot graphs of lvid and lv vol
    if args.datapath.endswith('.png'):
        mask, orig_size = predict_single(net, device, args.datapath, args.size)
        volumes, lvids = return_features(mask, orig_size)
        plot_graph(volumes, 'VOL')
        plot_graph(lvids, 'LVID')
    # else generate segmentation masks and stats for a test set
    else:
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        dir_mask = os.path.join(args.datapath, 'mask')
        dir_img = os.path.join(args.datapath, 'img')
        # define transformations and initialize data loader
        transforms = Compose([Resize(args.size), ToTensor()])
        dataset = BasicDataset(dir_img, dir_mask, transforms)
        n_test = len(dataset)
        print("Going to test model on ", n_test, " images")
        # make sure shuffle is false so predictions match!
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # make predictions
        predict_all(net, device, test_loader, args.output)
