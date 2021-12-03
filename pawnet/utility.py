import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
import os
import tqdm

import seaborn as sns
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision.utils import make_grid
from attrdict import AttrDict
import yaml
from sklearn.model_selection import StratifiedKFold
import copy
import pickle
# from tqdm import tqdm_notebook

# additional lightning 

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import LightningDataModule, LightningModule


import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

from timm import create_model

# def seed_everything(seed=1234):
#     """
#     Utility function to seed everything
#     source: https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch
#     """
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

def read_yaml(filename):
    """
    Read yaml configuation and returns the dict

    Parameters
    ----------
    filename: string
        Path including yaml file name
    """

    with open(filename) as f:
        config = yaml.safe_load(f)

    return config


    
# configs
class BaseConfigLoader:
    
    def __init__(self,config_path):
        self.config = read_yaml(config_path)
            
    def load_config(self):
        return AttrDict(self.config)


class pawnetDataset(torch.utils.data.Dataset):
    """
    Dataset
    Based on template https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(self,annotation_df, img_dir,transform=None,target_transform=None,test=False,custom_len=None):
        self.annotation_df = annotation_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.test=test # if dataset contains labels
        self.custom_len=custom_len # if we want to define our own epoch
        
        
    def __len__(self):
        """Define 1 epoch"""
        if self.custom_len is None:
            return len(self.annotation_df)
        else:
            return self.custom_len
    
    def __getitem__(self,idx):
        """called batch num of times"""
        img_path = os.path.join(self.img_dir, self.annotation_df.iloc[idx, 0]) # ID is column index 0
        image = read_image(img_path+".jpg")
        if self.test:
            label = 0
        else:
            label = self.annotation_df.iloc[idx, 13] # Pawpularity is column index 13
            
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class PetfinderDataModule(LightningDataModule):
    """
    Lightning datamodule to handle all loaders
    """
    def __init__(
        self,
        img_dir, # os.path.join(file_path,"train")
        train_df,
        val_df,
        train_transformation,
        test_transformation,
        batch_size = 64,
        num_workers = 2
    ):
        super().__init__()
        self.img_dir = img_dir
        self._train_df = train_df
        self._val_df = val_df
        self.train_transformation = train_transformation
        self.test_transformation = test_transformation
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self):
        train_data = pawnetDataset(annotation_df=self._train_df,img_dir = self.img_dir ,transform = self.train_transformation) # can set custom len to let model exceed training size (since we are augmenting)
        return torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,num_workers =self.num_workers, shuffle=True, pin_memory=False)

    def val_dataloader(self):
        val_data = pawnetDataset(annotation_df=self._val_df,img_dir = self.img_dir, transform = self.test_transformation)
        return torch.utils.data.DataLoader(val_data,batch_size=self.batch_size,num_workers =self.num_workers, shuffle=False, pin_memory=False)



class pawNetBasic(pl.LightningModule):
    """
    First cut basic pawNet model
    we will improve on this - this serves as skeleton code
    for other models
    
    timm contains collection of several pretrained models
    
    This is a lightning variant *
    
    
    lightning model requires the following methods:
    1. forward 
    2. training_step (logic inside the iteration loop) , validation_step, test_step (not stable on tpu)
    3. training_epoch_end, validation_epoch_end
    4. configure_optimizers 
    
    other configurable list here https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    
    """
    
    def __init__(self,criterion, model_config):
        """

        Parameters
        ----------
        criterion : [type]
            torch loss criterion
        model_config : 
            attrDict object from base_config_manager.load_config().XXmodel
        dropout : float, optional
            dropout, by default 0.5
        lr : float, optional
            learning rate, by default 0.00001
        """
        super().__init__()
        self.dropout = model_config.dropout
        self.lr = model_config.learning_rate
        self._criterion = criterion        
        # initialize layers
        # https://fastai.github.io/timmdocs/tutorial_feature_extractor
        # remove FCL by setting num_classes=0
        self.pretrained = create_model(
            model_config.pretrained, 
            pretrained=True, 
            num_classes=0, 
            in_chans=3
        )
        self.model_config = model_config
        # get mixup parameter
        if self.model_config.mixup is not None:
            print("will perform mixup")
            self.mixup_alpha = self.model_config.mixup.alpha
        else:
            self.mixup_alpha = None

        
        # create layers based on pretrained model selected (this affects the feature map size)
        # self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d(1) # timm pretrain comes with pooling    
        self.linear_1 = torch.nn.Linear(in_features=self.model_config.feature_map_out_size,out_features=1000)
        self.prelu = torch.nn.PReLU()
        self.linear_2 = torch.nn.Linear(in_features=1000,out_features=1)
        
    def forward(self,x):
        out = self.pretrained(x)
#         out = out.view(out.size(0), -1) # reshape for linear. timm already returns with CHW flatten
        out = torch.nn.Dropout(self.dropout)(out)
        out = self.linear_1(out)
        out = self.prelu(out)
        out = self.linear_2(out)
        
        
        
        return out
    
    
    def training_step(self, batch, batch_idx):
        """
        logic instead batch loop
        """
        loss, pred, labels = self.__share_step(batch, 'train')
        
        return {'loss': loss, 'pred': pred, 'labels': labels}
        
    def validation_step(self, batch, batch_idx):
        """
        logic instead batch loop for validation
        """
        
        loss, pred, labels = self.__share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, 'labels': labels}
    
    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        
        # check for mixup and only for train
        # autocast for more effective training / memory usage
        # https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        # https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/291159

        if (self.mixup_alpha is not None) & (mode == "train"):
            # perform mixup
            x,y = mixup(images,labels,alpha=self.mixup_alpha,device= self.device)
            with torch.cuda.amp.autocast(enabled=True):
                logits = self.forward(x).squeeze(1)
                loss = self._criterion(logits, y)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                logits = self.forward(images).squeeze(1)
                loss = self._criterion(logits, labels)
        
        # return logloss for training mode, scaled for others
        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().cpu() * 100.
        return loss, pred, labels
        
    def training_epoch_end(self, outputs):
        """
        called every end of epoch, contains logic
        at end of epoch
        """
        self.__share_epoch_end(outputs, mode = 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, mode = 'val')    
        
    def __share_epoch_end(self, outputs, mode):
        """
        output is a list of output defined in
        `training_step` as well as `validation_step`.
        Need to iterate through each iteration's output.
        the output was a dict
        """
        preds = []
        labels = []
        losses = []
        for out in outputs:
            pred, label, loss = out['pred'], out['labels'], out["loss"]
            preds.append(pred)
            labels.append(label)
            losses.append(loss.view(-1,1))
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        losses = torch.cat(losses)
        if mode == "train":
            loglogss_metrics = losses.mean() # average logloss across iterations
            self.log(f'{mode}_logloss', loglogss_metrics, prog_bar=True)
        else:
            print(f"{mode}: skip logging for logloss")
            
        # RMSE
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        # automatic accumulation at end of epoch for training, true always for test,validation loops
        self.log(f'{mode}_RMSE_loss', metrics, prog_bar=True)
        
        
    def configure_optimizers(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        
        Any of these 6 options.

        Single optimizer.

        List or Tuple of optimizers.

        Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).

        Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.

        Tuple of dictionaries as described above, with an optional "frequency" key.

        None - Fit will run without any optimizer.
        """
        #opt = torch.optim.Adam(self.parameters())
        # TODO: add learning rate to config
        opt = torch.optim.AdamW(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,eta_min=1e-4)
  
        return [opt]


def mixup(images,labels,alpha=0.2, device = "cuda"):
    """
    Mixup - to be applied in training loop
    on batches of images and labels

    Parameters
    ----------
    images : torch tensor
        batch of images (N,C,H,W)
    label : torch tensor
        batch of labels (N)
    alpha : float, optional
        beta distribution concentration, by default 0.2
        between 0 and 1
    """
    # shuffle images
    # https://discuss.pytorch.org/t/shuffling-a-tensor/25422/9
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html type_as to allow multi gpu
    indices = torch.randperm(len(images)).to(device)
    new_images = images[indices].view(images.size())
    
    # shuffle target
    new_labels = labels[indices].view(labels.size())

    # sample from beta distribution
    beta_distribution = torch.distributions.beta.Beta(alpha,alpha)
    t = beta_distribution.sample(sample_shape=torch.Size([len(images)])).to(device)

    # create beta weights
    tx = t.view(-1,1,1,1)
    ty = t.view(-1)
    x = (images * tx) + (new_images * (1-tx))
    y = labels * ty + new_labels * (1-ty)

    return x,y


class pawNetAdvance(pl.LightningModule):
    """
    Advance version with manual optimization.
    It uses manual optimization to allow custom
    optimzer and scheduler step. This is needed in order
    to perform model weights averaging

    Custom optimizer/scheduler step:
    https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#gradient-accumulation


    SWA / EMW weights
    https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
    https://forums.pytorchlightning.ai/t/stochastic-weight-averaging/400

    Current epoch
    https://github.com/PyTorchLightning/pytorch-lightning/issues/1424

    """
    
    
    def __init__(self,criterion, model_config):
        """

        Parameters
        ----------
        criterion : [type]
            torch loss criterion
        model_config : 
            attrDict object from base_config_manager.load_config().XXmodel
        dropout : float, optional
            dropout, by default 0.5
        lr : float, optional
            learning rate, by default 0.00001
        """
        super().__init__()
        self.dropout = model_config.dropout
        self.lr = model_config.learning_rate
        self._criterion = criterion        
        # initialize layers
        # https://fastai.github.io/timmdocs/tutorial_feature_extractor
        # remove FCL by setting num_classes=0
        self.pretrained = create_model(
            model_config.pretrained, 
            pretrained=True, 
            num_classes=0, 
            in_chans=3
        )
        self.model_config = model_config

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # get mixup parameter
        if self.model_config.mixup is not None:
            print("will perform mixup")
            self.mixup_alpha = self.model_config.mixup.alpha
        else:
            self.mixup_alpha = None
        
        # determine if to average =====
        # https://forums.pytorchlightning.ai/t/stochastic-weight-averaging/400
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        if self.model_config.average.type == "swa":
            self.wa_model = AveragedModel(self)
            self.wa_start = self.model_config.average.start
        elif self.model_config.average.type == "ema":
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.01 * averaged_model_parameter + 0.99 * model_parameter
            self.wa_model = AveragedModel(self, avg_fn=ema_avg)
            self.wa_start = self.model_config.average.start
        elif self.model_config.average is None:
            self.wa_model = None
            self.wa_start = None
        
        
        
        # create layers based on pretrained model selected (this affects the feature map size)
        # self.global_avg_pooling = torch.nn.AdaptiveAvgPool2d(1) # timm pretrain comes with pooling    
        self.linear_1 = torch.nn.Linear(in_features=self.model_config.feature_map_out_size,out_features=1000)
        self.prelu = torch.nn.PReLU()
        self.linear_2 = torch.nn.Linear(in_features=1000,out_features=1)
        
    def forward(self,x):
        out = self.pretrained(x)
#         out = out.view(out.size(0), -1) # reshape for linear. timm already returns with CHW flatten
        out = torch.nn.Dropout(self.dropout)(out)
        out = self.linear_1(out)
        out = self.prelu(out)
        out = self.linear_2(out)
        
        return out
    
    
    def training_step(self, batch, batch_idx):
        """
        logic instead batch loop
        """
        # retrieve optimizers and schedulers
        opt = self.optimizers()
        opt.zero_grad()
        
        
        loss, pred, labels = self.__share_step(batch, 'train')
        self.manual_backward(loss)
        opt.step()
        
        return {'loss': loss, 'pred': pred, 'labels': labels}
        
    def validation_step(self, batch, batch_idx):
        """
        logic instead batch loop for validation
        """
        
        loss, pred, labels = self.__share_step(batch, 'val')
        return {'loss': loss, 'pred': pred, 'labels': labels}
    
    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        
        # check for mixup and only for train
        # autocast for more effective training / memory usage
        # https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        # https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/291159

        if (self.mixup_alpha is not None) & (mode == "train"):
            # perform mixup
            x,y = mixup(images,labels,alpha=self.mixup_alpha,device= self.device)
            with torch.cuda.amp.autocast(enabled=True):
                logits = self.forward(x).squeeze(1)
                loss = self._criterion(logits, y)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                logits = self.forward(images).squeeze(1)
                loss = self._criterion(logits, labels)
        
        # return logloss for training mode, scaled for others
        pred = logits.sigmoid().detach().cpu() * 100.
        labels = labels.detach().cpu() * 100.
        return loss, pred, labels
        
    def training_epoch_end(self, outputs):
        """
        called every end of epoch, contains logic
        at end of epoch
        """
        

        # update scheduler , parameters
        
        # sch = self.lr_schedulers()[0]
        swa_sch = self.lr_schedulers()
        if self.wa_model is not None:
            if self.current_epoch > self.wa_start:
                self.wa_model.update_parameters(self)
                swa_sch.step()
        # else:
        #     sch.step()

        self.__share_epoch_end(outputs, mode = 'train')


    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, mode = 'val')    
        
    def __share_epoch_end(self, outputs, mode):
        """
        output is a list of output defined in
        `training_step` as well as `validation_step`.
        Need to iterate through each iteration's output.
        the output was a dict
        """
        preds = []
        labels = []
        losses = []
        for out in outputs:
            pred, label, loss = out['pred'], out['labels'], out["loss"]
            preds.append(pred)
            labels.append(label)
            losses.append(loss.view(-1,1))
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        losses = torch.cat(losses)
        if mode == "train":
            loglogss_metrics = losses.mean() # average logloss across iterations
            self.log(f'{mode}_logloss', loglogss_metrics, prog_bar=True)
        else:
            print(f"{mode}: skip logging for logloss")
            
        # RMSE
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        # automatic accumulation at end of epoch for training, true always for test,validation loops
        self.log(f'{mode}_RMSE_loss', metrics, prog_bar=True)
        
        
    def configure_optimizers(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        
        Any of these 6 options.

        Single optimizer.

        List or Tuple of optimizers.

        Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).

        Dictionary, with an "optimizer" key, and (optionally) a "lr_scheduler" key whose value is a single LR scheduler or lr_scheduler_config.

        Tuple of dictionaries as described above, with an optional "frequency" key.

        None - Fit will run without any optimizer.
        """
        #opt = torch.optim.Adam(self.parameters())
        # TODO: add learning rate to config
        opt = torch.optim.AdamW(self.parameters(),lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,eta_min=1e-4)
        swa_scheduler = SWALR(opt,anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)
  
        return [opt], [swa_scheduler]