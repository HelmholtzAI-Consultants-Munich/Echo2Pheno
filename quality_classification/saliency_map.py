from model import CardioNet
from dataset import EchoDataset, ToTensor, ResizeNpy, ToNumpy

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Normalize, Compose, Resize
from torch.utils.data import DataLoader


def get_saliency_map(network, dataset, input_path):
    network.eval()
    network = network.double()
    for i,sample in enumerate(dataset):
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
        if label.item() == 1:
            orig_img = os.path.join(input_path, 'good', filename[0])
        else:
            orig_img = os.path.join(input_path, 'bad', filename[0])
        orig_img = Image.open(orig_img)
        orig_img = np.array(orig_img)
        # plot image and saliency map
        fig=plt.figure()
        columns = 2
        rows = 1
        plt.title(title)
        plt.axis('off')
        fig.add_subplot(rows, columns, 1)
        plt.imshow(orig_img, cmap='gray')
        plt.axis('off')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(saliency[0][0], cmap='hot')
        plt.axis('off')
        plt.savefig(str(i)+'.png')
        plt.show()
    

if __name__ == "__main__":
    model_path = 'final_models/bagging_final_nets/bagging_net0.pth'
    input_path = '/Users/christina.bukas/Documents/AI_projects/datasets/cardioMice/TimeSeries/newWindows'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CardioNet(n_channels=1, n_classes=1)
    print("Loading model ", model_path)
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded !")
    # load data
    transforms = Compose([Resize((256, 256)), ToNumpy(), ToTensor(), Normalize([0.5], [0.5])])   
    dataset = EchoDataset(input_path, transforms, 'png')                         # create data set class
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # Generate saliency maps
    get_saliency_map(network=net, dataset=test_loader)
    