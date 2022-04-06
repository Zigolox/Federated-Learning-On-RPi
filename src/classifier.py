
"""PyTorch MNIST image classifier"""


import timeit
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms

import flwr as fl

def partitioner(
    dataset:torch.utils.data.Dataset,
    batch_size: int,
    user_id: int,
    nb_users: int
    ) -> torch.utils.data.DataLoader:
    """Function that partitions datasets"""
    