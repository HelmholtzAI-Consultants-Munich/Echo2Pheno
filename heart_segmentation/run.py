import os
import sys
import logging
import argparse
import torch

from trainer import Trainer
from quicknat import QuickNat

def get_args():
    '''
    Required arguments
    ------------------
        -d: The path to the training dataset
    Optional arguments
    ------------------
        -e: The number of epochs to train for. Default is 1
        -b: The batch size during training and validation. Default is 1
        -p: The size to which images need to be resized during load. Default is 256
        -l: The learning rate during training. Default is 0.001
        -f: The path to the pretrained model to load
        -v: The training-validation split. Default is 20, meaning 20% validation, 80% training
        -o: The path to the checkpoint dir. Default is ./checkpoints
    '''
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--datapath', required=True, type=str, 
                       help='Provide path to training dataset')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-p', '--im-size', metavar='ImS', type=int, default=256,
                        help='Image width and height', dest='imsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, 
                        help='Load model from a .pth file') # default='checkpoints/CP_epoch151.pth',
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-o', '--output-path', dest='outputpath', type=str, default='checkpoints')

    return parser.parse_args()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    project_path = os.path.dirname(os.path.realpath(__file__)) 
    checkpoint_path = args.outputpath
    data_path = args.datapath
    project_path = os.path.dirname(os.path.realpath(__file__))
    # check if checkpoint path exists and if not create it
    if not os.path.exists(os.path.join(project_path, checkpoint_path)):
        os.mkdir(checkpoint_path)
    # set current device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device %s' % (device))
    # Change here to adapt to your network configuration
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
    # create model instance
    net = QuickNat(params) 
    logging.info('''Network:
         %s input channels
                 %s output channels (classes)''' % (params['num_channels'], params['num_class']))
    # load existing model if defined
    if args.load:
        load_path = os.path.join(project_path, args.load)
        state_dict = torch.load(load_path, map_location=device)
        net.load_state_dict(state_dict['net_state_dict'])
        logging.info('Model loaded from %s' % (args.load))
    # cast to device
    net.to(device=device, dtype=torch.float32)
    # create trainer instance
    trainer = Trainer(net=net,
                    device=device,
                    input_path=data_path,
                    val_percent=args.val / 100,
                    augmentations=False,
                    img_size=args.imsize,
                    output_path=checkpoint_path,
                    batch_size=args.batchsize,
                    epochs=args.epochs,
                    lr=args.lr,
                    save_cp=True)
    # train the network
    try:
        logging.info('Going to train network')
        trainer.run_train()
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)