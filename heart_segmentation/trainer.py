import logging
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import RandomAffine, Compose, Resize, ToTensor, RandomHorizontalFlip
from tensorboardX import SummaryWriter
from datetime import datetime

from utils import dice_confusion_matrix
from dataset import BasicDataset

class Trainer(object):
    '''
    This class performs all the standard procedures for training a neural network
    Parameters
    ----------
        net: model.QuickNat
            The network to train 
        device: torch.device
            The currently running divice, e.g. cpu, gpu0    
        input_path: string
            The path to the training dataset
        val_percent: float
            The percentage of data to keep for validation, e.g. 0.2 then 20% of the data is kept for validation and 80% for training
        augmentations: bool
            True or False depending on if you want to use data augmentation. If True random affine augmentation is applied.
        img_size: int
            The value to which we need to resize images before inputting them to the network
        output_path: string
            Where to save model checkpoints
        batch_size: int, default: 1
            Defines the batch size of the train and validation set
        epochs: int, default: 1
            Number of epochs to train the network for 
        lr: float, default: 0.001
            The learning rate of the optimization step
        weights_per_class: tuple of floats, default: (0.5, 0.5)
            Sets weighting of different classes on loss function for imbalanced classes
        save_cp: bool, default: False
            Defines whether or not to save the model at certain intervals
    Attributes
    ----------
        net: model.QuickNat
            The network to train 
        device: torch.device
            The currently running divice, e.g. cpu, gpu0    
        input_path: string
            The path to the training dataset
        val_percent: float
            The percentage of data to keep for validation, e.g. 0.2 then 20% of the data is kept for validation and 80% for training
        augmentations: bool
            True or False depending on if you want to use data augmentation
        img_size: int
            The value to which we need to resize images before inputting them to the network
        output_path: string
            Where to save model checkpoints
        batch_size: int, default: 1
            Defines the batch size of the train and validation set
        epochs: int, default: 1
            Number of epochs to train the network for 
        lr: float, default: 0.001
            The learning rate of the optimization step
        global_step: int
            Counts the total number of iterations through the network
        save_cp: bool, default: False
            Defines whether or not to save the model at certain intervals
        writer: tensorboardX.SummaryWriter
            The tensorboard object used to write losses and image results to tensorboard
        optimizer: torch.optim.Adam
            The optimizer used during training 
        bce_loss:  torch.nn.BCELoss
            Binary cross entropy loss, the loss function used during training
        sigmoid: torch.nn.Sigmoid
            The sigmoid function used on the network output 
        mse: torch.nn.MSELoss
            The mse used for network evaluation on train and val set
        train_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the train set 
        val_loader: torch.utils.data.dataloader.DataLoader
            The data loader of the validation set

    '''
    def __init__(self, net, device, input_path, val_percent, augmentations, img_size, output_path, batch_size=1, epochs=1, lr=0.001, save_cp=False):

        self.net = net
        self.device = device
        self.input_path = input_path
        self.val_percent = val_percent
        self.augmentations = augmentations
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.global_step = 0

        project_path = os.path.dirname(os.path.realpath(__file__))
        # create tensorboard instance with the current date and time as name
        datetime_dir = datetime.now()
        datetime_dir = datetime_dir.strftime("%d%m%Y%H%M%S")
        if not os.path.exists(os.path.join(project_path, 'runs')):
            os.mkdir(os.path.join(project_path, 'runs'))
        os.mkdir(os.path.join(project_path, 'runs', datetime_dir))
        self.writer = SummaryWriter(logdir=os.path.join(project_path, 'runs', datetime_dir), comment="LR_%f_BS_%s" %(lr, batch_size))
        #writer.add_text('train info','this is a good training!', 0)
        # define optimizer, losses and vgg for perceptual loss
        self.optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.bce_loss = nn.BCELoss()   
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.save_cp = save_cp
        self.output_path = output_path

    def _create_loaders(self):
        """
        This function is used to create a training and validation data loader
        """
        # if augmentations is true apply random affine augmentation else just resize and convert to tensor
        if self.augmentations:
            rand_aff = RandomAffine(degrees=5, translate=(0.2,0.2), scale=(0.1,0.3))
            transforms = Compose([Resize(self.img_size ), RandomHorizontalFlip(), rand_aff, ToTensor()])
        else:
            transforms = Compose([Resize(self.img_size ), ToTensor()])
        dataset = BasicDataset(os.path.join(self.input_path, 'img'), os.path.join(self.input_path, 'mask'), transforms)       # create data set class
        logging.info('Created a dataset with %s examples' % (len(dataset)))
        # get number of train and validation samples
        n_val = int(len(dataset) * self.val_percent)                                                    
        n_train = len(dataset) - n_val
        # split the dataset randomly to train and val
        train, val = random_split(dataset, [n_train, n_val])      
        # load train and val data with data loader
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
        

    def train(self):
        """
        This function is where the actual setup and training of the network takes place. It calls other functions of the class for 
        evaluation, training for one epoch etc. and evaluates the performance according to the Mean Square Error (MSE) and Dice score.
        The best MSE and dice score on the validation set are kept track of and a model is saved if these metrics are higher in a 
        certain epoch than for any of the previous epochs of training.
        """
        # create data loaders for training and validation sets
        self._create_loaders()

        logging.info("""Starting training:
            Epochs:          %s
            Batch size:      %s
            Learning rate:   %f
            Training size:   %s
            Validation size: %s
            Checkpoints:     %s
            Device:          %s
        """ % (self.epochs, self.batch_size, self.lr, len(self.train_loader), len(self.val_loader), self.save_cp, self.device.type))
        
        best_mse = 0
        best_dice = 0

        for epoch in range(self.epochs):
            
            self.net.train()
            # train network for an entire epoch
            train_loss, train_mse, train_dice = self._train_epoch(epoch)
            # write results
            self.writer.add_scalar('Epoch_loss/train_loss', train_loss, epoch)
            logging.info('Training MSE: {}, DICE: {}'.format(train_mse, train_dice))
            logging.info('Training loss: {}'. format(train_loss))
            
            # perform validation test after every epoch
            with torch.no_grad():
                val_loss, val_mse, val_dice = self.eval(epoch)
            # write results
            self.writer.add_scalar('Epoch_loss/validation_loss', val_loss, epoch)
            logging.info('Validation MSE: {}, DICE: {}'.format(val_mse, val_dice))    
            logging.info('Validation loss: {}'.format(val_loss))
            # keep track of the best dice score and mse on validation set - they must both be better for the model to be saved 
            if val_dice > best_dice and val_mse < best_mse:
                best_dice = val_dice
                best_mse = val_mse

            # save model in checkpoints if validation dice score and mse are the best so far or else save model at least once at the end
            if self.save_cp and ((val_dice == best_dice and val_mse == best_mse) or epoch==self.epochs-1):
                if val_dice == best_dice and val_mse == best_mse:
                    logging.info('Best model so far. Going to save.')
                    filepath = os.path.join(self.output_path, 'CP_best_e%s.pt' % (epoch+1))
                elif epoch==self.epochs-1:
                    logging.info('Last epoch. Going to save model.')
                    filepath = os.path.join(self.output_path, 'CP_last_e%s.pt' % (epoch+1))
                    
                state = {
                    'epoch': epoch,
                    'net_state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': train_loss,
                    'dice': train_dice, 
                    'mse': train_mse, 
                    'best_dice': best_dice, # of val
                    'best_mse': best_mse # of val
                }
                torch.save(state, filepath)
                logging.info('Checkpoint %s saved !'%(epoch+1))

            logging.info('Done with epoch %s / %s' % (epoch+1, self.epochs))

        self.writer.close()


    def _train_epoch(self,epoch):
        """
        This function trains the network for one epoch
        Parameters
        ----------
            epoch: int
                The current training epoch
        Returns
        -------
            train_loss: float
                The average loss on the training set for this epoch
            mse_eval: torch.Tensor
                The average mean square error on the training set for this epoch
            dice_eval: torch.Tensor
                The average dice score on the training set for this epoch
        """

        train_loss = 0
        dice_eval = 0
        mse_eval = 0
            
        for batch in self.train_loader:

            #torch.cuda.empty_cache() # uncomment if OOM error occurs, could help
            # load image and cast to device
            img = batch['image']
            mask = batch['mask']
            img = img.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.float32)

            # get prediction --> classify each pixel to one of two values: 0 (background) or 1 (heart)
            self.optimizer.zero_grad()
            mask_pred = self.net(img)
            mask_pred = self.sigmoid(mask_pred)
            # compute losses
            loss = self.bce_loss(mask_pred, mask)
            # add losses to tensorboard
            self.writer.add_scalar('Loss/bce', loss.item(), self.global_step)
            train_loss += loss.item()
            # backprobagate and update weights
            loss.backward()
            self.optimizer.step()

            # calculate evaluation metrics
            mask_pred = torch.round(mask_pred)
            mse_eval += self.mse(mask, mask_pred)
            dice_avg, _ = dice_confusion_matrix(mask_pred, mask, 1)
            dice_eval += dice_avg

            self.global_step += 1
        
        # save images of last batch every 10 epochs
        if epoch % 2 == 0:
            self.writer.add_images('train/image', img, epoch)
            self.writer.add_images('train/mask', mask, epoch)
            self.writer.add_images('train/pred_mask', mask_pred, epoch)
        
        mse_eval = mse_eval/len(self.train_loader)
        dice_eval = dice_eval/len(self.train_loader)
        train_loss = train_loss/len(self.train_loader)
        
        return train_loss, mse_eval, dice_eval

    def eval(self, epoch):
        """"""
        """
        This function evaluates the network performance on the validation set
        Parameters
        ----------
            epoch: int
                The current training epoch. Validation is performed after each training epoch
        Returns
        -------
            val_loss: float
                The average loss on the validation set for this epoch
            avg_mse: float
                The average mean square error on the validation set for this epoch
            avg_dice: torch.Tensor
                The average dice score on the validation set for this epoch
        """
        self.net.eval()
        val_loss = 0
        avg_mse = 0
        avg_dice = 0

        for batch in self.val_loader:
            # load img and mask from disk and cast to device
            img = batch['image']
            mask = batch['mask']
            img = img.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.float32)
            # make prediction of segmentation mask
            mask_pred = self.net(img)
            mask_pred = self.sigmoid(mask_pred)
            # compute loss
            loss = self.bce_loss(mask_pred, mask)
            val_loss += loss.item()
            # compute mse and dice score
            mask_pred = torch.round(mask_pred)
            avg_mse += self.mse(mask, mask_pred).item()
            dice_score, _ = dice_confusion_matrix(mask_pred, mask, 1)
            avg_dice += dice_score
        # every 2 epochs save image and masks to tensorboard
        if epoch % 2 == 0:
            self.writer.add_images('val/image', img, epoch)
            self.writer.add_images('val/mask', mask, epoch)
            self.writer.add_images('val/pred_mask', mask_pred, epoch)
        # return the average loss, mse and dice score
        val_loss = val_loss / len(self.val_loader)
        avg_mse = avg_mse/len(self.val_loader)
        avg_dice = avg_dice/len(self.val_loader)    
        
        return val_loss, avg_mse, avg_dice



