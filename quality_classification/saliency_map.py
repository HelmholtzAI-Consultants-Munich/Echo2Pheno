from model import CardioNet
from dataset import EchoDataset, ToTensor, ResizeNpy, ToNumpy
import argparse

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Normalize, Compose, Resize
from torch.utils.data import DataLoader


def get_saliency_map(network, dataset):
    """
    This function creates saliency maps for the 10 first images of the test set show the image and saliency map and saves the figures.
    By plotting the saliency maps we see where in the image the model focuses for making the classification decision.
    Parameters
    ----------
        network: model.CardioNet
            The trained network
        dataset: torch.utils.data.dataloader.DataLoader
            The test data
    """
    network.eval()
    network = network.double()
    for i,sample in enumerate(dataset):
        # only perform for the first 10 images in the test set
        if i == 10:
            break
        image = sample['image']
        label = sample['label']
        filename = sample['filename']
        image.requires_grad_()
        _, _, _, _, score, pred = network(image.double())
        # backpropagate score
        score.backward()
        saliency = image.grad.data.abs() #saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        # load original image
        title = 'File ' + filename[0] + ' label: ' + str(label.item()) + ' prediction: ' + str(round(pred.item()))
        # convert torch image to numpy
        image = torch.squeeze(image)
        image = image.detach().cpu().numpy()
        # plot image and saliency map
        fig=plt.figure()
        columns = 2
        rows = 1
        plt.title(title)
        plt.axis('off')
        fig.add_subplot(rows, columns, 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(saliency[0][0], cmap='hot')
        plt.axis('off')
        # show and save figure
        plt.savefig(str(i)+'.png')
        plt.show()
    
def get_args():
    '''
    Required arguments
    ------------------
        -d: The path to the test dataset
    Optional arguments
    ------------------
        -m: The path to the model you wish to test. Default is ./checkpoints/quality-clas-net.pth
    '''
    parser = argparse.ArgumentParser(description='Get saliency maps for test sets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/quality-clas-net.pth', metavar='FILE',
                        help="Specify the path of the trained model")
    parser.add_argument('--datapath', '-d', required=True,
                        help='Specify the path of the test set')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_path = args.model
    input_path = args.datapath #'/Users/christina.bukas/Documents/AI_projects/datasets/cardioMice/TimeSeries/newWindows'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CardioNet(n_channels=1, n_classes=1)
    #print("Loading model ", model_path)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded !")
    # load data
    transforms = Compose([Resize((256, 256)), ToNumpy(), ToTensor(), Normalize([0.5], [0.5])])   
    dataset = EchoDataset(input_path, transforms, 'png')                         # create data set class
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # Generate saliency maps
    get_saliency_map(network=net, dataset=test_loader)
    