# data: https://drive.google.com/drive/folders/1XRyf21Kdda2mQAqrbqApGGthU0sIDqMe?usp=sharing

# Don't erase the template code, except "Your code here" comments.

from logging import raiseExceptions
import subprocess
import sys
from tkinter import W
import tqdm 
from tqdm import tqdm


########################################## PACKAGES INSTALATION ##########################################
# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
#PACKAGES_TO_INSTALL = ["gdown==4.4.0",]
#subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

"""
name: aim_hw2
channels:
  - conda-forge
dependencies:
  - matplotlib +
  - numpy +
  - omegaconf +
  - pandas +
  - pip +
  - python=3.12 +
  - pytorch=2.5.1 +
  - torchvision=0.20.1 + 
  - wandb +
  - seaborn +
  - tqdm +
  - scikit-learn +
  """



import torch
# Your code here... ######## IMPORTS


######################################## MAKE TIMERS WHICH WILL BE USED TO SEE PROCESSES #################################


from time import time

class Timer():
    def __init__(self, name: str, notify: bool = True):
        self.name = name
        self.notify = notify
        self.start_time = time()
    def __enter__(self):
        self.start_time = time()
        if self.notify:
            print(f'> "{self.name}" started')
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time
        if self.notify:
            print(f'< {str(self)}\n')
    def __str__(self):
        return f'"{self.name}" took {self.elapsed_time:.2f} seconds'




######################################## HERE WE WORK WITH DATA -- MAKE DATASETS, DATALOADERS #################################


# first of all, we are to make transformations for data augmentation
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Normalize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, GaussianBlur
from torch.utils.data import Dataset, ConcatDataset
from omegaconf import DictConfig

import torch.nn
import pandas as pd
import os
from os import path

# which data aug we want to be able to make?

AUGMENTATION_REGISTRY = {
    "flip_hor": RandomHorizontalFlip(p=1),
    "flip_ver": RandomVerticalFlip(p=1),
    "blur": GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
    "noise": ColorJitter(brightness=.5, hue=.3),
    "rotation": RandomRotation(degrees=(0, 180))
}

# now we can make transforms with our image
def make_transformations(aug_names, device):
    sequence = []      
    for aug_name in aug_names:
        sequence.append(AUGMENTATION_REGISTRY[aug_name])
    sequence.append(transforms.ToTensor())
    sequence.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))      # ССЫЛКА
    return sequence

import torch


# now we should make datasets. i will use a special function for it - ImageFolder, which lets us to parse imagenet dataset properly
def get_dataloader(path, kind, cfg):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    train_transforms = transforms.Compose(make_transformations(cfg.data.data_augmentation, cfg.device))
    check_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if kind == 'train':

        train_ok_dataset = ImageFolder(root=os.path.join(path, 'train'), transform=check_transforms)  # normal images
        custom_one = ImageFolder(root=os.path.join(path, 'train'), transform=transforms.Compose([ColorJitter(brightness=.8, hue=.5),
                                                    RandomRotation(degrees=(0, 180)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))  # normal images

        custom_two = ImageFolder(root=os.path.join(path, 'train'), transform=transforms.Compose([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
                                                    RandomRotation(degrees=(0, 360)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

        custom_three = ImageFolder(root=os.path.join(path, 'train'), transform=transforms.Compose([GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
                                                    ColorJitter(brightness=.8, hue=.5),
                                                    RandomRotation(degrees=(-90, 90)),
                                                    RandomHorizontalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    
        datasets = [train_ok_dataset, custom_one, custom_two, custom_three]
        for augmethod in cfg.data.data_augmentation:   # add pictures with wanted augmentaion methods
            train_add_dataset = ImageFolder(root=os.path.join(path, 'train'), transform=transforms.Compose([AUGMENTATION_REGISTRY[augmethod],
                                                                                                    transforms.ToTensor(),
                                                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
            datasets.append(train_add_dataset)

        train_dataset = ConcatDataset(datasets)

        return DataLoader(train_dataset, 
                          batch_size=cfg.data.batch_size,
                          **cfg.datamodule,
                          shuffle=True,
                          drop_last=True)
    
    elif kind == 'val':
        val_dataset = ImageFolder(root=os.path.join(path, 'val'), transform=check_transforms)
        return DataLoader(val_dataset, 
                          batch_size=cfg.data.batch_size,
                          **cfg.datamodule,
                          shuffle=False,
                          drop_last=False)

    elif kind == 'test_for_tuning':
        test_for_tuning_dataset = ImageFolder(root=os.path.join(path, 'test_for_tuning'), transform=check_transforms)
        return DataLoader(test_for_tuning_dataset, 
                          batch_size=cfg.data.batch_size,
                          **cfg.datamodule,
                          shuffle=False,
                          drop_last=False)

    elif kind == 'test':
        test_dataset = ImageFolder(root=os.path.join(path, 'test'), transform=check_transforms)
        return DataLoader(test_dataset, 
                          batch_size=cfg.data.batch_size,
                          **cfg.datamodule,
                          shuffle=False,
                          drop_last=False)
    else:
        raise Exception('Unknown sample format')






######################################## MAKE OUR MODEL AS A NN.MODULE CLASS #################################

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer


from torch.optim import AdamW, SGD, Adam
OPTIMIZER_REGISTRY= {'AdamW': AdamW, 
                    'SGD': SGD,
                    'Adam': Adam}

def get_optimizer(model, cfg):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    optim_kwargs = {k:v for k, v in cfg.optimizer.items() if k!='name'}
    optimizer = OPTIMIZER_REGISTRY[cfg.optimizer.name](model.parameters(), **optim_kwargs)
    return optimizer


def load_scheduler(optimizer, cfg):
    """
    Create an scheduler object for `optimizer`.

    return:
    schedular:
        `torch.optim.lr_scheduler`
    """
    kwargs = {'optimizer': optimizer, **cfg.scheduler}
    return torch.optim.lr_scheduler.ConstantLR(optimizer = optimizer, factor = 0.5)




######################################### MAKE TRAINER STEPS #########################################





# INIT WEIGHTS
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn



def set_global_seed(seed):
    """
    Sets random seeds everywhere 
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)




def parameter_count(model):       # from https://wandb.ai/wandb_fc/tips/reports/How-To-Calculate-Number-of-Model-Parameters-for-PyTorch-and-TensorFlow-Models--VmlldzoyMDYyNzIx
    total_params = sum(param.numel() for param in model.parameters())
    return total_params


################################################################# TRAINER STEPS ##############################################################

# HERE WE INIT TRAINER STEPS

import warnings
from omegaconf import DictConfig
import wandb
from pathlib import Path

import torch

 

# from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC
import torch
from torch.nn import ModuleDict

from torch.nn import ModuleList
# from torchmetrics import Metric

from typing import Optional
from torch import dtype 





################################_LOSSES_#########################
import torch
from torch.nn import Module

    
LOSS_REGISTRY = {'CrossEntropyLoss': nn.CrossEntropyLoss}
def load_loss(loss_cfg, device):
    loss_kwargs = {k: v for k, v in loss_cfg.items() if k not in ['loss']}
    #return  LOSS_REGISTRY[loss_cfg.loss](**loss_kwargs)
    return nn.CrossEntropyLoss(**loss_kwargs).to(device)






###################################################### TRAINER STEPS ##################################
def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    y_pred = model(batch)
    y_pred = nn.Softmax(dim = 1)(y_pred)
    return y_pred


def validate(dataloader, model, cfg):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    model.eval()
    val_acc = 0
    val_loss = 0
    loss = load_loss(cfg.loss, cfg.device)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            X, y = batch
            X, y = X.to(cfg.device), y.to(cfg.device)
            y_pred = model(X)
            val_loss += loss(y_pred, y)
            val_acc += (torch.argmax(y_pred, dim = 1) == y).sum()
    val_acc = val_acc / len(dataloader.dataset)       # total / num of objects
    val_loss = val_loss / len(dataloader.dataset)             # mean through all batches
    return val_acc, val_loss


def test(model, test_dataloader, cfg):
    test_loss = 0
    test_acc = 0
    model.eval()
    loss = load_loss(cfg.loss, cfg.device)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            X, y = batch
            X, y = X.to(cfg.device), y.to(cfg.device)
            y_pred = model(X)
            curr_loss = loss(y_pred, y)
            test_loss += curr_loss
            test_acc += (torch.argmax(y_pred, dim = 1) == y).sum()
    test_acc = test_acc / len(test_dataloader.dataset)
    test_loss = test_loss / len(test_dataloader.dataset)
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
    print(test_acc, test_loss)

    


def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer, scheduler, cfg, epoch, logger):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    logger.watch(model)
    train_acc = 0

    model.train()
    loss = load_loss(cfg.loss, cfg.device)

    with Timer('Training step'):
        for _, batch in enumerate(tqdm(train_dataloader)):
            X, y = batch
            X, y = X.to(cfg.device), y.to(cfg.device)
            y_pred = model(X)
            train_loss = loss(y_pred, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc += (torch.argmax(y_pred, dim = 1) == y).sum()
            wandb.log({'train_loss': train_loss / cfg.data.batch_size})                       # mean (in batch) loss -> ~ for 1 object

    wandb.log({'train_acc': train_acc / len(train_dataloader.dataset)})     # mean

    with Timer('Validation step'):
        model.eval()
        val_acc = 0
        val_loss = 0
        loss = load_loss(cfg.loss, cfg.device)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader)):
                X, y = batch
                X, y = X.to(cfg.device), y.to(cfg.device)
                y_pred = model(X)
                val_loss += loss(y_pred, y)
                val_acc += (torch.argmax(y_pred, dim = 1) == y).sum()
    val_acc = val_acc / len(val_dataloader.dataset)       # total / num of objects
    val_loss = val_loss / len(val_dataloader.dataset)             # mean through all batches
    wandb.log({'val_loss': val_loss, 'val_acc': val_acc})
    
    
    if epoch % 10 == 0:
        print('saving checkpoint...')
        ckpt_path = os.path.join(cfg.model_dir, f'checkpoints/epoch_num={epoch};val_acc={val_acc:.3f}') 
        save_weights(model, ckpt_path)
    
    return val_acc







#################################################### SAVE CHECKPOINTS ####################################################


from omegaconf import DictConfig
from pathlib import Path

from typing import TYPE_CHECKING




def save_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    torch.save(model.state_dict(), checkpoint_path)


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


####################################################### I DONT KNOW WHATS THAT ######################################
def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here; md5_checksum = "abcd"
    # Your code here; google_drive_link = "https://drive.google.com/file/d/abcd/view?usp=sharing"
    md5_checksum = "2fba692926005ac449129cb8ee05a498"
    # server link = /mnt/calc/lizzzafomenko/aim/results/ANSWER/checkpoints/BEST_epoch_num_48_val_acc_0537
    google_drive_link = "https://drive.google.com/file/d/1oQDcl8D2jKy7euvKjb_6pRRl5OAUyFDy/view?usp=sharing"


    return md5_checksum, google_drive_link






############################################## OTHER THINGS #########################################
# я люблю OmegaConf + hydra, но здесь это излишне + не понимаю как правильно сдать такое решение.
# поэтому при сдаче я буду импортировать из обычного словаря
from omegaconf import DictConfig, OmegaConf


cfg = OmegaConf.create(
    {
    'data': {
        'batch_size': 256,
        'image_size': 64,
        'data_augmentation': ['noise', 'rotation', "flip_hor", "flip_ver", "blur"]
    },
    'datamodule': {
        'num_workers': 2,
        'persistent_workers': True,
        'pin_memory': False
    },
    'loss': {
        'loss': 'CrossEntropyLoss',
        'reduction': 'sum' 
    },
    'scheduler': {               # OneCycleLR
        #'max_lr': 2.5e-3,
        #'pct_start': 0.01,
        #'div_factor': 100,
        #'final_div_factor': 1
        'T_max': 100
    },
    'optimizer': {
        'name': 'AdamW',
        'lr': 4e-4,                  # ССЫЛКУ
        'weight_decay': 0.001              # ССЫЛКУ
    },
    'model': {
        'num_layers': 2
    },
    'server': {
    # Server configuration (paths)
        'repo_root': '/mnt/calc/lizzzafomenko/aim',
        'data_root': '/mnt/calc/lizzzafomenko/aim/tiny-imagenet-200'
    },
    'seed': 57,
    'model_dir': 'results/ANSWER',  
    'epoch_num': 50,
    'device': 1,
    'logger': {
        'project': 'aim_imagenet_classification',
        'log_model': 'all',
        'name': 'ANSWER'
    }
    },

)

import torch.nn as nn




import models as models
from models import *


def get_model(cfg):
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    model = FINAL_MODEL(cfg.model)
    return model.to(cfg.device)




# а здесь уже начинаем обучение

# 1) set seed

def main(cfg: DictConfig):


    # assert good scheduler behaviour 
    cfg.scheduler.total_steps = cfg.epoch_num

    # global seed and make everyting to one format
    set_global_seed(cfg.seed)
    torch.set_float32_matmul_precision('medium')

    # load paths for checkpoints and save used config (to load it later)
    checkpoint_dir = Path(cfg.model_dir) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    config_save_path = Path(cfg.model_dir) / "core_config.yaml"
    OmegaConf.save(cfg, config_save_path)

    # here we init the model from another file - had to try some architectures
    with Timer('Model initialization'):
        model = get_model(cfg)
        print(f'Number of parameters: {parameter_count(model)}')
        print(model)
    

    # load loss - Entropy loss
    with Timer('Get loss'):
        loss = load_loss(cfg.loss, cfg.device)

    # init wandb logging
    def init_wandb():
        wandb.login(key='local-f57ccc8fdf497fe140ac4b5e383f9492de46ed93', host="http://localhost:8385")
        return wandb.init(project=cfg.logger.project, name=cfg.logger.name)
    logger = init_wandb()


    # log number of parameters
    wandb.log({'param_number': parameter_count(model)})

    # train dataloaders with LOTS of augmentation
    with Timer('Train dataloader initialization'):
        train_dataloader = get_dataloader(f"{cfg.server.data_root}", 'train', cfg)

    # val dataloader - from /tiny-imegenet-200/val but removed all _0 and _1 to another sample - 'test_for_training'
    cfg.scheduler.total_steps = len(train_dataloader)*cfg.epoch_num
    with Timer('Validation dataloader initialization'):
        val_dataloader = get_dataloader(f"{cfg.server.data_root}", 'val', cfg)

    # load optimizer and scheduler
    optimizer = get_optimizer(model, cfg)
    scheduler = load_scheduler(optimizer, cfg)

    # start training + plus validation
    with Timer('Training'):
        max_val_acc = float('-inf')
        best_path = ''
        for epoch in range(cfg.epoch_num):
            with Timer(f'Epoch number {epoch}'):
                train_dataloader = get_dataloader(f"{cfg.server.data_root}", 'train', cfg)
                curr_val_acc = train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer, scheduler, cfg, epoch, logger)
                if curr_val_acc > max_val_acc:
                    max_val_acc = curr_val_acc
                    best_model = model
                    best_path =  os.path.join(cfg.model_dir, f'checkpoints/BEST_epoch_num={epoch};val_acc={max_val_acc:.3f}')
        
        # SAVE best checkpoint for evaluation
        print('loading best checkpoint...')
        save_weights(model, best_path)


   
    # here we load pseudotest dataloader - 'test_for_tuning'
    with Timer('Testing'):
        with Timer('Load test dataloader'):
            test_dataloader = get_dataloader(f"{cfg.server.data_root}", 'test_for_tuning', cfg)
        with Timer(f'Load best performing checkpoint from {best_path}'):
            model = load_weights(model, best_path)
        test(model, test_dataloader, cfg)

    # here we upload index to class dictionary from val dataset 
    map_classes = {class_idx: class_name for class_name, class_idx in val_dataloader.dataset.class_to_idx.items()}

    # get new model and load weights from best path.
    # best_path = 'BEST_epoch_num=48;val_acc=0.537'
    model = get_model(cfg)
    model = load_weights(model, best_path)

    # here i predict labels for pseudotest - just to check if everything works correct. save it to test_to_val_checking.csv - just in case
    with Timer('Predicting valid labels'):
        # example real submission
        pred_dict = {}
        pred_labels = []
        model.to(cfg.device)
        model.eval()
        for batch, _ in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(cfg.device)
                predicted_labels = predict(model, batch).argmax(1)
            pred_labels.extend(predicted_labels.tolist())
        for i, (img_name, _) in enumerate(test_dataloader.dataset.imgs):
            pred_dict[img_name.split("/")[-1]] = map_classes[pred_labels[i]]

        print(pred_dict)

        submission_df = pd.DataFrame(pred_dict.items(), columns=["id", "pred"])
        submission_df.to_csv(f"{cfg.model_dir}/test_to_val_checking.csv", index=False)
    
    # load real test data (with no labels provided) and predict labels. make a df to submission.csv and submit in kaggle. got 0.5305
    test_dataloader = get_dataloader(f"{cfg.server.data_root}", 'test', cfg)
    with Timer('Predicting labels'):
        # example real submission
        pred_dict = {}
        pred_labels = []
        model.to(cfg.device)
        model.eval()
        for batch, _ in tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(cfg.device)
                predicted_labels = predict(model, batch).argmax(1)
            pred_labels.extend(predicted_labels.tolist())
        for i, (img_name, _) in enumerate(test_dataloader.dataset.imgs):
            pred_dict[img_name.split("/")[-1]] = map_classes[pred_labels[i]]

        submission_df = pd.DataFrame(pred_dict.items(), columns=["id", "pred"])
        submission_df.to_csv(f"{cfg.model_dir}/submission.csv", index=False)
    
    # finish everything!
    print('finished :)')

    def finish_wandb():
        wandb.finish()
    
    finish_wandb()


if __name__ == "__main__":
    main(cfg)

