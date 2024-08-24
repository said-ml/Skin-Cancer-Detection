
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.optim import lr_scheduler

from typing import Union
from typing import Optional
from typing import Tuple
from typing import Dict

from albumentations import Compose
from albumentations import Resize
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2

class seed_configrations:
    seed:int=42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed=seed_configrations()

class torch_configurations:

               learning_rate=1e-5
               optimizer='adam'
               epochs=50
               train_batch_size=16
               valid_batch_size=32



               if not torch.cuda.is_available():
                    import warnings
                    warnings.warn('run the code with the power if gpu')
                    device='cpu'

               else:
                     device='cuda'

               backbone_name:str ='swin_large_patch4_window7_224'
               pretrained:bool=True     # False

               n_accumulate:int=1
               cosine_scheduler= {'linear_lr' : 'CosineAnnealingLR',
                                   'linear_warm_start' : 'CosineAnnealingWarmRestarts'
                                    }

               n_workers:int=0


device=torch_configurations().device
print(f'device{device}')




class lightgbm_configurations:

      def __init__(self,
                   learning_rate:float=1e-4,
                   n_iters:int=200,
                   device:  Union['gpu', 'cpu']='gpu', # default gpu
                   )->None:

                  self.learning_rate=learning_rate
                  self.n_iters=n_iters
                  self.device=device


class base_configurations:
      n_folds: int = 5
      is_training:bool= True # False
      on_kaggle: bool =True  # False

