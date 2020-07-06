"""AE main training script
This script allows the user to train an AE model using Protoplanetary disk
images loaded with the 'dataset.py' class, AE model located in 'ae_model.py' class,
and the trainig loop coded in 'ae_training.py'. 
The script also uses Weight & Biases framework to log metrics, model hyperparameters, configuration parameters, and training figures.
This file contains the following
functions:
    * run_code - runs the main code
For help, run:
    python ae_main.py --help
"""

import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
from src.dataset import ProtoPlanetaryDisks, MNIST
from src.ae_model import *
from src.ae_training import Trainer
from src.utils import count_parameters
import wandb

torch.autograd.set_detect_anomaly(True)

# set random seed
rnd_seed = 13
np.random.seed(rnd_seed)
torch.manual_seed(rnd_seed)
torch.cuda.manual_seed_all(rnd_seed)
#os.environ['PYTHONHASHSEED'] = str(rnd_seed)

# program flags
parser = argparse.ArgumentParser(description='AutoEncoder')
parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                    default=False,
                    help='Load data and initialize models [False]')
parser.add_argument('--machine', dest='machine', type=str, default='local',
                    help='were to is running (local, colab, [exalearn])')

parser.add_argument('--data', dest='data', type=str, default='PPD',
                    help='data used for training (MNIST, [PPD])')

parser.add_argument('--lr', dest='lr', type=float, default=1e-4,
                    help='learning rate [1e-4]')
parser.add_argument('--lr-sch', dest='lr_sch', type=str, default=None,
                    help='learning rate shceduler '+
                    '([None], step, exp, cosine, plateau)')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                    help='batch size [128]')
parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100,
                    help='total number of training epochs [100]')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    default=False, help='Early stoping')

parser.add_argument('--cond', dest='cond', type=str, default='F',
                    help='label conditional AE ([F],T)')
parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=32,
                    help='dimension of latent space [32]')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2,
                    help='dropout for all layers [0.2]')
parser.add_argument('--kernel-size', dest='kernel_size', type=int, default=3,
                    help='2D conv kernel size, encoder [3]')
parser.add_argument('--conv-blocks', dest='conv_blocks', type=int, default=5,
                    help='conv+actfx+pool blocks [5]')
parser.add_argument('--model_name', dest='model_name', type=str, default='',
                    help='name of model')
parser.add_argument('--comment', dest='comment', type=str, default='',
                    help='extra comments')
args = parser.parse_args()

# Initialize W&B project and save user defined flags
wandb.init(project="PPD-AE", tags=['AE'])
wandb.config.update(args)
wandb.config.rnd_seed = rnd_seed


# run main program
def run_code():
    # asses which device will be used, CPY or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    # Load Data #
    if args.data == 'PPD':
        dataset = ProtoPlanetaryDisks(machine=args.machine, transform=True,
                                      img_norm=True)
    elif args.data == 'MNIST':
        dataset = MNIST(args.machine)
    else:
        print('Error: Wrong dataset (MNIST, Proto Planetary Disk)...')
        raise

    if len(dataset) == 0:
        print('No items in training set...')
        print('Exiting!')
        sys.exit()
    print('Dataset size: ', len(dataset))
    
    # data loaders for training and testing
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size,
                                                       shuffle=True,
                                                       test_split=.2,
                                                       random_seed=rnd_seed)

    if args.data == 'PPD' and args.cond == 'T':
        wandb.config.physics_dim = len(dataset.meta_names)
    else:
        wandb.config.physics_dim = 0

    print('Physic dimension: ', wandb.config.physics_dim)

    # Define AE model, Ops, and Train #
    # To used other AE models change the following line,
    # different types of AE models are stored in src/ae_model.py
    model_name = args.model_name
      
    if model_name == 'ConvUpSamp_AE':
       model = ConvUpSamp_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
         
    elif model_name == 'ResNet_UpSamp_AE':
      model = ResNet_UpSamp_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
      
    elif model_name == 'ResNet_Linear_AE':
      model = ResNet_Linear_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
    elif model_name == 'ResNet_Tconv_AE':
      model = ResNet_Tconv_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
      
    elif model_name == 'ConvLin_AE':
      model = ConvLin_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
      
    elif model_name == 'TranConv_AE':
      model = TranConv_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
      
    elif model_name == 'Linear_AE':
      model = Linear_AE(latent_dim=args.latent_dim,
                               img_dim=dataset.img_dim,
                               in_ch=dataset.imgs.shape[1])
      
    # log model architecture and gradients to wandb
    wandb.watch(model, log='gradients')

    wandb.config.n_train_params = count_parameters(model)
    print('Summary:')
    print(model)
    print('Num of trainable params: ', wandb.config.n_train_params)
    print('\n')

    # Initialize optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning Rate scheduler
    if args.lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=20,
                                              gamma=0.5)
    elif args.lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=0.985)
    elif args.lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=50,
                                                         eta_min=1e-5)
    elif args.lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=.5,
                                                         verbose=True)
    else:
        scheduler = None

    print('Optimizer    :', optimizer)
    print('LR Scheduler :', scheduler.__class__.__name__)

    print('########################################')
    print('########  Running in %4s  #########' % (device))
    print('########################################')

    # initialize trainer
    trainer = Trainer(model, optimizer, args.batch_size, wandb,
                      scheduler=scheduler, print_every=500,
                      device=device)

    if args.dry_run:
        print('******** DRY RUN ******** ')
        return

    # run training/testing iterations
    trainer.train(train_loader, test_loader, args.num_epochs,
                  save=True, early_stop=args.early_stop)


if __name__ == "__main__":
    print('Running in: ', args.machine, '\n')
    for key, value in vars(args).items():
        print('%15s\t: %s' % (key, value))

    run_code()
