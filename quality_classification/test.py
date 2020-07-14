import argparse
import logging
import csv
import os
import sys
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, Resize

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
from quality_classification.dataset import EchoDataset, ToTensor, ResizeNpy, ToNumpy
from quality_classification.model import CardioNet


def save_csv(filename_list, labels_pred, labels_true):
    """
    This function creates a csv file 'test_results.csv' and adds to each row three cells: [filename, label_prediction, label_groung_truth]
    This can later be used to eavluate result or extract statistics
    Parameters
    ----------
        filename_list: list of strings
            A list of the filenames of all samples in the test set
        labels_pred: list of ints
           A list of all the predictions of the newtork
        labels_true: list of ints
            A list of all the ground truth labels of the test set
    """
    csv_new = 'test_results.csv'
    with open(csv_new, 'w', newline='') as new_csv_file:
        writer = csv.writer(new_csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['filename', 'predicted label', 'true label'])
        for name, pred, true in zip(filename_list, labels_pred, labels_true):
            writer.writerow([name, int(pred), true])

def calc_f1(tp, fp, p):
    """
    This function calculates the f1 score
    Parameters
    ----------
        tp: int
            Number of true positive samples
        fp: int
            Number of false positive samples
        p: int
            Number of positive samples
    Returns
    -------
        f1: float
            The f1 score
        precision: float
            The precision
        recall: float
            The recall
    """
    recall = tp/p
    precision = tp/(tp+fp)
    f1 = 2*(precision*recall)/(precision+recall)
    return f1, precision, recall

def predict_single(net, device, img_path, size):
    """
    This function makes a prediction on a single image
    Parameters
    ----------
        net: model.CardioNet
            The network used to make prediction
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
        label_pred: 
            The prediction of the network - this is the output of the last layer, i.e. sigmoid layer, so it will be a value in range [0,1]
        orig_size: 
            The size of the original image before it is resized to be sent to the network
    """
    net = net.double()
    if img_path is str:
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
    else:
        img = img_path
    orig_size = img.size
    transforms = Compose([ResizeNpy(size), ToTensor(), Normalize([0.5], [0.5])]) 
    img = transforms(img)
    img.to(device=device, dtype=torch.float32)
    img = torch.unsqueeze(img, dim=0) 
    with torch.no_grad():
        _, _, _, _, _, label_pred = net(img.double())
    # remember what is returned is the sigmoid output - it then needs to be rounded to 0 or 1 for final prediction
    label_pred = label_pred.item()
    return label_pred, orig_size

def predict_dataset(net, device, test_loader, net_run):
    """
    This function makes predictions on an entire test set
    Parameters
    ----------
        net: model.CardioNet
            The network used to make prediction
        device: torch.device
            The currently running divice, e.g. cpu, gpu0  
        test_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the test set 
        net_run: int 
            When bootstrapping is used (see main), it defines which network is currently being run.
            If bootstrapping is not used it is set to 100
    Returns
    -------
        file_names, labels_pred, labels_true, sig_outputs
        file_names: list of strings
            A list of the filenames of all samples in the test set
        labels_pred: list of ints, 0 or 1
           A list of all the predictions of the newtork - rounded sigmoid output
        labels_true: list of ints
            A list of all the ground truth labels of the test set
        sig_outputs: list of floats
            A list of all the output of the network - output of final sigmoid layer
    """
    # initialize evaluation metrics, loss function etc
    labels_pred = []
    labels_true = []
    file_names = []
    sig_outputs = []
    BCE_loss = torch.nn.BCELoss()
    loss = 0
    accuracy = 0
    p = 0
    n = 0
    fp = 0
    fn = 0
    tp = 0
    # set model into evaluation mode for predictions
    net.eval() 
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            net = net.double()
            image = batch['image']
            image.to(device=device, dtype=torch.float32)
            label = batch['label']
            # count total number of negatives and positives
            if label==0:
                n += 1
            else:
                p += 1
            # append ground truth and file name to lisr  
            labels_true.append(label.item())
            file_names.append(batch['filename'][0])                         
            # make prediction
            _, _, _, _, _, label_pred = net(image.double())                 
            # calculate BCE loss
            loss += BCE_loss(label_pred.double(), label.double().detach()) 
            # round sigmoid output to 0 or 1 and append prediction to list 
            sig_outputs.append(label_pred.double())
            label_pred = torch.round(label_pred.double())                   
            labels_pred.append(label_pred.item())
            # calcualte accuracy
            accuracy += (label_pred == label).sum().item()                  
            # cound number of fp, fn and tp for calculatinf fnr, fpr, f1 score
            if label_pred != label:
                if label_pred == 1:
                    fp += 1
                else:
                    fn += 1
            else:
                if label==1:
                    tp += 1
            # print message every 10%
            if (idx+1) % int(0.1*len(test_loader)) == 0:
                print('Done for ', (idx+1)*10//int(0.1*len(test_loader)), '% of testing')
    
    # print results    
    print("Measured loss: ", loss.item()/n_test)
    print("Measured accuracy: ", 100*accuracy/n_test)
    print("False positive rate (FPR): ", fp/n)
    print("False neagtive rate (FNR): ", fn/p)
    print("New f1score: ", f1_score(labels_pred, labels_true))
    if net_run == 100:
        print("Finished testing model")
    else:
        print("Finished model: ", net_run)
    return file_names, labels_pred, labels_true, sig_outputs


def get_args():
    parser = argparse.ArgumentParser(description='Predict label for test images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', '-d', metavar='INPUT', required=True,
                        help='path of input images')
    parser.add_argument('--model', '-m', default='checkpoints/bootstrap_net0.pth', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--bootstrap', '-b', default=False,
                        help="Define whether or not prediction should be made from single model or average")
    parser.add_argument('-r', '--runs', default=10, type=int, 
                       help='If you have chosen to run multiple networks define how many')
    parser.add_argument('-p', '--im-size', metavar='ImS', type=int, default=256,
                        help='Image width and height', dest='imsize')
    parser.add_argument('--datatype', '-t', default='png',
                       help='Provide npy or png to choose data input type')
    parser.add_argument('--output', '-o', default=False, 
                        help='Specify if a csv list of predictions should be created')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # load data
    if args.datatype == 'png':
        transforms = Compose([Resize((args.imsize, args.imsize)), ToNumpy(), ToTensor(), Normalize([0.5], [0.5])])   
    elif args.datatype == 'npy':
        transforms = Compose([ResizeNpy(args.imsize), ToTensor(), Normalize([0.5], [0.5])])  
    dataset = EchoDataset(args.datapath, transforms, args.datatype)                         # create data set class
    n_test = len(dataset)
    print("Going to test model on ", n_test, " images")
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # get current device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device, ", device)
    # if bootstrap load 10 models and get average of predictions
    if args.bootstrap:
        fp = 0
        tp = 0
        fn = 0
        for i in range(10):
            model_name = os.path.join(args.model, 'bootstrap_net%s.pth'% (i))
            # load model
            net = CardioNet(n_channels=1, n_classes=1)
            net.to(device=device)
            net.load_state_dict(torch.load(model_name, map_location=device))
            print("Model loaded !")
            # make predictions with current model on entire data set
            file_names, labels_pred, labels_true, sig_outputs = predict_dataset(net, device, test_loader, i)
            # sum sigmoid outputs from all the models so far
            if i==0:
                sum_list = sig_outputs
            else:
                sum_list = [a + b for a, b in zip(sig_outputs, sum_list)]
        # get average sigmoid output and round result so predictions are either 0 or 1
        sum_list = [i/10 for i in sum_list]
        sum_list = [torch.round(i) for i in sum_list]
        # count total number of tp, fp, fn, tn, p and n
        for label_pred, label_true in zip(labels_true, sum_list):
            if label_pred != label_true:
                if label_pred == 1:
                    fp += 1
                else:
                    fn += 1
            else:
                if label_true==1:
                    tp += 1
        p = labels_true.count(1)
        n = len(labels_true) - p
        tn = n_test - fp - fn - tp 
        # measure accuracy
        accuracy = 100*(tp+tn)/n_test
        # print results of bootstrapping
        print('Average Accuracy outputs is: ', accuracy)
        print("Average False Positive Rate (FPR): ", fp/n)
        print("Average False Negative Rate (FNR): ", fn/p)
        print("Average F1 score: ", f1_score(sum_list, labels_true)) 
    # if we are not bootsrapping but using only one network for predictions
    else:
        # load model
        net = CardioNet(n_channels=1, n_classes=1)
        print("Loading model ", args.model)
        net.to(device=device)
        net.load_state_dict(torch.load(args.model, map_location=device))
        print("Model loaded !")
        file_names, labels_pred, labels_true, _ = predict_dataset(net, device, test_loader, 100)
    # save image name, predictions and true labels in a csv file to evaluate later! 
    if args.output:
        save_csv(file_names, labels_pred, labels_true)
    
       