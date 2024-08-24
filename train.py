import sys
from Configurations import*
from torch.nn import functional as TorchFunc
from Configurations import seed_configrations
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from multiprocessing.spawn import freeze_support


#\\\\\\\\\\\\\\\\\\\\

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#|||||||||||||||||||||
from typing_extensions import dataclass_transform

from data import train_df

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings('ignore')
import os

import gc   # for garbage collection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_auroc
from torchvision import transforms



import joblib
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from copy import deepcopy

from loss import Loss
from Configurations import torch_configurations
from Configurations import seed_configrations
from Configurations import base_configurations

from models.cnn import cnn_metadata

from data import*

from torcheval.metrics.functional import binary_auroc

# Running training and validation
def prepare_loaders(df, fold):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset = ISICDataset(train_df, feat=num_feat, hdf=hdf,
                                transforms=data_transforms['train'], is_training=True)
    valid_dataset = ISICDataset(valid_df, feat=num_feat, hdf=hdf,
                                transforms=data_transforms['valid'], is_training=False)

    '''
    indication:We Use torch.utils.data.DataLoader with num_workers=0:
    Temporarily set the num_workers parameter of your DataLoader to 0. 
    This will force the data loading to be done in the main process, avoiding multiprocessing-related issues
    '''
    train_loader = DataLoader(train_dataset, batch_size=torch_configurations.train_batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=torch_configurations.valid_batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, valid_loader



def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        metadata = data['metadata'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images, metadata).squeeze()

        for param in model.parameters():
            if torch.isnan(param).any():
                print("NaN detected in model parameters")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if torch.isnan(outputs).any():
            print("NaN detected in model output")
        print('model outputs,', outputs)
        outputs=torch.clamp(outputs, min=0, max=1)
        loss = Loss(outputs, targets)
        loss = loss / torch_configurations.n_accumulate

        loss.backward()

        if (step + 1) % torch_configurations.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                #assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler) # Check for type
                scheduler.step()

        auroc = TorchFunc.binary_cross_entropy(input=outputs, target=targets).item()

        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_AUROC=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()

    return epoch_loss, epoch_auroc


def valid_one_epoch( model,
                     dataloader,
                     device,
                     epoch)->None:

    '''

    :param model:
    :param optimizer:
    :param scheduler:
    :param dataloader:
    :param epoch:
    :return: None
    '''

    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0

    bar = tqdm(enumerate( dataloader, total = len(dataloader)))

    for step, data in bar:
        images = data['image'].to( device=device, dtype = torch.float)
        metadata= data['metadatacsv'].to(device=device, dtype=torch.float)
        targets = data['targets'].to(device=device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images, metadata).squeeze()
        torch.clamp(outputs, min=0, max=1)
        print(outputs)
        loss = Loss(outputs, targets)

        auroc = TorchFunc.binary_cross_entropy(input=outputs, target=targets).item()
        print(auroc)

        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_AUROC=epoch_auroc)

    gc.collect()

    return epoch_loss, epoch_auroc

# Main training function
def run_training(model, optimizer, scheduler, device, num_epochs, fold):
    best_model_wts = deepcopy(model.state_dict())
    best_epoch_auroc = 0
    best_epoch_loss = 1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_loader, valid_loader = prepare_loaders(df, fold)
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer,
                                                              scheduler, dataloader=train_loader,
                                                              device=device, epoch=epoch)

        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, dataloader=valid_loader, device=device, epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid AUROC'].append(val_epoch_auroc)

        # deep copy the model (balance auroc and loss accuracy)
        if (best_epoch_auroc <= val_epoch_auroc) & (best_epoch_loss - val_epoch_loss >= -0.05):
            print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_epoch_loss = val_epoch_loss
            #best_model_wts = deepcopy(model.state_dict())

            # Save a model file from the current directory
            #PATH = f"best_AUROC_model_fold{fold}.bin"
            #torch.save(model.state_dict(), PATH)
            print(f"Model Saved -- epoch: {epoch:.0f}, AUROC: {val_epoch_auroc:.4f}, Loss: {val_epoch_loss:.4f}")
        del train_loader, valid_loader
    print('Best val AUROC: {:4f}'.format(best_epoch_auroc))

    # load best model weights
    #model.load_state_dict(best_model_wts)

    return model, history


def running_KFold( n_folds:int=base_configurations.n_folds,
                   optimizer:torch.optim='ADAM')->None:

    KFolds = StratifiedGroupKFold( n_splits=n_folds)
    for fold, (train_idex, valid_idx ) in enumerate ( KFolds.split( X=df, y=df.target, groups=df.patient_id)):
        df.loc[valid_idx, 'fold'] = fold
        print( f'training Fold {fold}')
        torch.cuda.empty_cache()

        # Set a model
        model = cnn_metadata( torch_configurations.backbone_name,
                     pretrained = torch_configurations.pretrained)

        # move model to the gpu ('cuda')
        model = model.to( torch_configurations.device)

        # Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=torch_configurations.learning_rate,
                              weight_decay=1e-6)

        #if base_configurations.cosine_scheduler['linear_lr'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,
                                                       eta_min=1e-10)
        #elif base_configurations.cosine_scheduler ['linear_warm_start']== 'CosineAnnealingWarmRestarts':
            #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer)
        #, T_0=CONFIG['T_0'],
                                                                 #eta_min=CONFIG['min_lr'])

        # Prepare Dataloaders
        train_loader, valid_loader = prepare_loaders(df, fold=fold)

        # Run training
        model, history = run_training(model, optimizer, scheduler,
                                     torch_configurations.device,torch_configurations.epochs, fold)
        return None


# Plot Loss and AUROC Curves
def plot_metrics(history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['Train Loss'], label='Train Loss')
        ax1.plot(history['Valid Loss'], label='Valid Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()

        ax2.plot(history['Train AUROC'], label='Train AUROC')
        ax2.plot(history['Valid AUROC'], label='Valid AUROC')
        ax2.set_title('AUROC Curves')
        ax2.legend()

        plt.show()

# plot_metrics(history)

################## ============ ########### ============ ############## ============ ################
'''
       WHAT DOES NOT COME WITH EFFORT COME BY MORE EFFORT 
       
'''


# Preparing stratified k-fold
def KFold(n_fold):
        skf = StratifiedGroupKFold(n_splits=base_configurations.n_folds)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df.target, groups=df.patient_id)):
            df.loc[val_idx, 'fold'] = fold

        # Get best model for multiple folds
        for fold in range(base_configurations.n_folds):

              print(f'Training Fold {fold}')
              print(f'------------------------------')
              torch.cuda.empty_cache()

              # Setup model
              model = cnn_metadata(torch_configurations.backbone_name, pretrained=True)
              model.to(torch_configurations.device)

              # Optimizer & Scheduler
              optimizer = optim.Adam(model.parameters(), lr=torch_configurations.learning_rate,
                                     weight_decay=1e-6)

              # if base_configurations.cosine_scheduler['linear_lr'] == 'CosineAnnealingLR':
              scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=13016,
                                                         eta_min=1e-10)
              # Prepare Dataloaders
              train_loader, valid_loader = prepare_loaders(df, fold)

              # Run training
              model, history = run_training(model, optimizer, scheduler,
                                  torch_configurations.device, base_configurations.n_folds, fold)

              del model
              plot_metrics(history)
              del history

from multiprocessing import Process
if __name__ == '__main__':
    freeze_support()
    p = Process(target=KFold, args=(5,))
    p.start()
    p.join()
    #KFold(5)






