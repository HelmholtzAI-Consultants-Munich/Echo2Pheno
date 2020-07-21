import sys
import argparse
import logging
import os

import numpy as np
import torch
from datetime import datetime
from torch import float32
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torchvision.transforms import Normalize, Compose, Resize, RandomAffine
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from dataset import EchoDataset, ToTensor, ResizeNpy, ToNumpy
from model import CardioNet

project_path = os.path.dirname(os.path.realpath(__file__)) 

def train_net(net,
              device,
              train_loader,
              val_loader,
              n_train,
              n_val,
              output_path,
              net_run,
              epochs=5,
              batch_size=1,
              lr=0.001,
              weights_per_class=(0.5,0.5),
              save_cp=False):
    """
    This function trains the network
    Parameters
    ----------
        net: model.CardioNet
            The network to train 
        device: torch.device
            The currently running divice, e.g. cpu, gpu0    
        train_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the train set 
        val_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the validation set
        n_train: int
            Number of training samples
        n_val: int
            Number of validation samples
        output_path: string
            Where to save model checkpoints
        net_run: int 
            When bootstrapping is used (see main), it defines which network is currently being run.
            If bootstrapping is not used it is set to 100
        epochs: int, default: 5
            Number of epochs to train the network for 
        batch_size: int, default: 1
            Defines the batch size of the train and validation set
        lr: float, default: 0.001
            The learning rate of the optimization step
        weights_per_class: tuple of floats, default: (0.5, 0.5)
            Sets weighting of different classes on loss function for imbalanced classes
        save_cp: bool, default: False
            Defines whether or not to save the model at certain intervals
    Returns
    -------
        final_loss: dict
            The final loss of the trained model on the train and validation set in final_loss['train'] and final_loss['val']
        final_accuracy: dict
            The final accuracy of the trained model on the train and validation set in final_loss['train'] and final_loss['val']
        final_f1: dict
            The final f1 score of the trained model on the train and validation set in final_loss['train'] and final_loss['val']
    """
    logging.info("""Starting training:
        Epochs:          %s
        Batch size:      %s
        Learning rate:   %f
        Training size:   %s
        Validation size: %s
        Checkpoints:     %s
        Device:          %s
    """ % (epochs, batch_size, lr, n_train, n_val, save_cp, device.type))
    
    # create tensorboard instance in a directory named with the current date and time
    datetime_dir = datetime.now()
    datetime_dir = datetime_dir.strftime("%d%m%Y%H%M%S")
    tensorboard_dir = datetime_dir + '_' + str(net_run)
    if not os.path.exists(os.path.join(project_path, 'runs')):
        os.mkdir(os.path.join(project_path, 'runs'))
    os.mkdir(os.path.join(project_path, 'runs', tensorboard_dir))
    writer = SummaryWriter(logdir=os.path.join(project_path, 'runs', tensorboard_dir), comment="LR_%f_BS_%s" %(lr, batch_size))
    # define optimizer and loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    BCE_loss = torch.nn.BCELoss(reduction='none')  # not using nn.BCEwithLogitsLoss() since have sigmoid at last layer
    final_loss = {}
    final_accuracy = {}
    final_f1 ={}
    
    global_step = 0
    try:
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_accuracy = 0
            train_f1 = 0

            for idx, batch in enumerate(train_loader):
                image = batch['image']
                label = batch['label']
                assert image.shape[1] == net.n_channels, \
                        'Network has been defined with %s input channels, but loaded images have %s channels. Please check that the images are loaded correctly' % (net.n_channels, image.shape[1])
                # cast to device
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device)
                # create class weights for the batch
                class_weights = set_class_weights(label, weights_per_class)
                # get prediction
                _, _, _, _, _, label_pred = net(image)
                # compute loss
                loss = BCE_loss(label_pred, label.float().detach()) * class_weights
                loss = loss.mean()
                # compute accuracy and f1 score as evaluation metrics
                label_pred = torch.round(label_pred)
                train_accuracy += (label_pred == label).sum().item()
                train_f1 += f1_score(label_pred.detach(), label.detach())
                # add current loss to epoch loss
                train_loss += loss.item()
                # add losses to tensorboard
                writer.add_scalar('Loss', loss.item(), global_step)
                # backprobagate and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
                #logging.info('Batch %s/%s in epoch %s/%s - Loss: %s' % ((idx+1), (n_train//batch_size + 1), epoch+1, epochs, loss.item()))

            logging.info('Training loss: {}, accuracy: {}, f1 score: {}'. format(train_loss/n_train, 100*train_accuracy/n_train, train_f1/(idx+1)))
            # perform validation test after every epoch
            with torch.no_grad():
                val_loss, val_accuracy, val_f1  = eval_net(net, val_loader, device, n_val, weights_per_class)            
            logging.info('Validation loss: {}, accuracy: {}, f1 score: {}'.format(val_loss, val_accuracy, val_f1))
            # add loss and evaluation metrics to tensorboard
            writer.add_scalars('Epoch_loss', {'train': train_loss/n_train, 
                                            'val': val_loss}, epoch)
            writer.add_scalars('Epoch_accuracy', {'train': 100*train_accuracy/n_train,
                                                'val': val_accuracy}, epoch)
            writer.add_scalars('Epoch_f1', {'train': train_f1/(idx+1), 
                                            'val': val_f1}, epoch)
            logging.info('Done with epoch %s / %s' % (epoch+1, epochs))

        # save model after all epochs in checkpoints dir
        if save_cp: #and (epoch+1)%100==0 and epoch!=0:
            torch.save(net.state_dict(), os.path.join(
                    output_path, 'bootstrap_net%s.pth'% (net_run)))
            logging.info('Model %s saved !'%(net_run+1))
        writer.close()
        # add losses and evaluation metrics to dicts
        final_loss['train'] = train_loss/n_train
        final_loss['val'] = val_loss
        final_accuracy['train'] = 100*train_accuracy/n_train
        final_accuracy['val'] = val_accuracy
        final_f1['train'] = train_f1/(idx+1)
        final_f1['val'] = val_f1
        print('outputs: ', type(net), type(final_loss), type(final_accuracy), type(final_f1))
        return final_loss, final_accuracy, final_f1

    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(output_path, 'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def set_class_weights(label, weights_per_class):
    """
    This function creates a where each label in a batch is weighted according to its class
    Parameters
    ----------
        label: torch.Tensor
            The current batch of labels

        weights_per_class: tuple of floats, default: (0.5, 0.5)
            Sets weighting of different classes on loss function for imbalanced classes
    Returns
    -------
        class_weights: torch.Tensor
            Same size as labels, this includes a weight for each sample in the batch depending on if the sample is 0 or 1 and the weights_per_class
    """
    class_weights = np.zeros(label.size()[0])
    for i in range(label.size()[0]):
        if label[i] == 1:
            class_weights[i] = weights_per_class[0] #0.35
        else:
            class_weights[i] = weights_per_class[1] #0.65
    class_weights = torch.from_numpy(class_weights)
    return class_weights

def eval_net(net, loader, device, n_val, weights_per_class):
    """
    This function is called at the end of an epoch of training to evaluate the network's performance on the data set
    Parameters
    ----------
        net: model.CardioNet
            The network to train 
        loader: torch.utils.data.dataloader.DataLoader
            The data loader of the validation set
        device: torch.device
            The currently running divice, e.g. cpu, gpu0  
        n_val: int
            Number of samples in validation set
        weights_per_class: tuple of floats, default: (0.5, 0.5)
            Sets weighting of different classes on loss function for imbalanced classes
    Returns
    -------
        val_loss: float
            The average loss on the validation set
        val_accuracy: float
            The average accuracy on the validation set
        val_f1: float
        The average f1 score on the validation set
    """
    net.eval()
    val_loss = 0
    BCE_loss = nn.BCELoss()
    val_accuracy = 0
    val_f1 = 0

    for batch in loader:
        imgs = batch['image']
        labels = batch['label']
        imgs = imgs.to(device=device, dtype=float32)
        labels = labels.to(device=device)
        class_weights = set_class_weights(labels, weights_per_class)
        _, _, _, _, _, label_preds = net(imgs)
        loss  = BCE_loss(label_preds, labels.float().detach()) * class_weights
        val_loss += loss.mean().item()
        label_preds = torch.round(label_preds)
        val_accuracy += (label_preds == labels).sum().item()
        val_f1 += f1_score(labels.detach().cpu(), label_preds.detach().cpu())
    val_loss = val_loss/len(loader)
    val_accuracy = val_accuracy*100/n_val
    val_f1 = val_f1/len(loader)
    return val_loss, val_accuracy, val_f1

def get_args():
    '''
    Required arguments
    ------------------
        -d: The path to the training dataset
    Optional arguments
    ------------------
        -t: The datatype of the training set. Can be either 'npy' or 'png'. Default is png
        -a: Either True of False defining whether bagging should be performed. For more information see readme. Default is False.
        -r: Number of runs if bagging is defined. Default is 10
        -e: The number of epochs to train for. Default is 1
        -b: The batch size during training and validation. Default is 1
        -p: The size to which images need to be resized during load. Default is 256
        -l: The learning rate during training. Default is 0.001
        -f: The path to the pretrained model to load
        -v: The training-validation split. Default is 20, meaning 20% validation, 80% training
        -o: The path to the checkpoint dir. Default is ./checkpoints
    '''
    parser = argparse.ArgumentParser(description='Train classifier to detect good and bad acquisitions',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datapath', required=True, type=str, 
                       help='Provide path to training dataset')
    parser.add_argument('-t', '--datatype', default='png', type=str, 
                       help='Provide npy or png to choose data input type')
    parser.add_argument('-a', '--bootstrap', default=False, type=bool, 
                       help='Decided if one model should be trained or multiple')
    parser.add_argument('-r', '--runs', default=10, type=int, 
                       help='If you have chosen to run multiple networks define how many')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-p', '--im-size', metavar='ImS', type=int, default=256,
                        help='Image width and height', dest='imsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-o', '--output-path', dest='outputpath', default='checkpoints', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    # optional - create an output file to save results
    f = open("test.out", 'w')
    sys.stdout = f
    # basic set up
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    checkpoint_path = args.outputpath
    data_path = args.datapath
    data_type = args.datatype
    bootstrap = args.bootstrap
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device %s' % (device))
    # load data and define transformations
    affine = RandomAffine(degrees=0, translate=(0.2,0.05), scale=(0.8,1.2))
    if data_type == 'png':
        transforms = Compose([Resize((args.imsize, args.imsize)), affine,ToNumpy(), ToTensor(), Normalize([0.5], [0.5])])  
    elif data_type == 'npy':
        transforms = Compose([ResizeNpy(args.imsize), ToTensor(), Normalize([0.5], [0.5])])    
    else:
        print('You can only give npy or png as a valid data input type')
        sys.exit()       
    dataset = EchoDataset(data_path, transforms, data_type)  # create data set class
    n_val = int(len(dataset) * args.val / 100)                   # get number of train and val samples
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])      # split the dataset randomly to train and val
    # load train and val data with data loader
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 

    if bootstrap:
        # train 10 networks with the same configurations to get average of results
        # initialize the network
        net = CardioNet(n_channels=1, n_classes=1)
        logging.info('''Network:
                        %s input channels
                        %s output channels (classes)''' % (net.n_channels, net.n_classes))
        # load pretrained net if args.load is given
        if args.load:
            net.load_state_dict(torch.load(args.load, map_location=device))
            logging.info('Model loaded from %s' % (args.load))
        net.to(device=device, dtype=torch.float32)

        avg_loss_train=0
        avg_accuracy_train=0
        avg_f1_score_train=0
        avg_loss_val=0
        avg_accuracy_val=0
        avg_f1_score_val=0
        for i in range(args.runs):
            loss, accuracy, f1 = train_net(net=net,
                                            device=device,
                                            train_loader=train_loader,
                                            val_loader=val_loader,
                                            n_train=n_train,
                                            n_val=n_val,
                                            output_path=checkpoint_path,
                                            net_run=i,
                                            epochs=args.epochs,
                                            batch_size=args.batchsize,
                                            lr=args.lr,
                                            save_cp=True)
            avg_loss_val += loss['val']
            avg_accuracy_val += accuracy['val']
            avg_f1_score_val += f1['val']
            avg_loss_train += loss['train']
            avg_accuracy_train += accuracy['train']
            avg_f1_score_train += f1['train']
            logging.info('Done with model %s / 10' % (i+1))

        logging.info('Average train loss: {}, accuracy: {} and f1 score: {}'.format(avg_loss_train/10, avg_accuracy_train/10, avg_f1_score_train/10))
        logging.info('Average validation loss: {}, accuracy: {} and f1 score: {}'.format(avg_loss_val/10, avg_accuracy_val/10, avg_f1_score_val/10))

    else:
        # initialize the network
        net = CardioNet(n_channels=1, n_classes=1)
        logging.info('''Network:
                        %s input channels
                        %s output channels (classes)''' % (net.n_channels, net.n_classes))
        # load pretrained net if args.load is given
        if args.load:
            net.load_state_dict(torch.load(args.load, map_location=device))
            logging.info('Model loaded from %s' % (args.load))
        # train the network
        _, _, _ = train_net(net=net,
                            device=device,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            n_train=n_train,
                            n_val=n_val,                                
                            output_path=checkpoint_path,
                            net_run=0,
                            epochs=args.epochs,
                            batch_size=args.batchsize,
                            lr=args.lr,
                            save_cp=True)
    f.close()
