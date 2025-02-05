import matplotlib.pyplot as plt
import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter


def one_hot_encode_recording_date(inputs, num_classes=10):
    """
    Converts (N, 2, 28, 28) input tensor into (N, 11, 28, 28),
    where recording_date is expanded to 10 one-hot channels.
    """
    # Extract density and recording_date channels
    density = inputs[:, 0, :, :].unsqueeze(1)
    recording_date = inputs[:, 1, :, :].long()  

    # Perform one-hot encoding (N, 28, 28, 10)
    one_hot = F.one_hot(recording_date, num_classes=num_classes)  
    
    one_hot = one_hot.permute(0, 3, 1, 2)  

    # Concatenate density and one-hot encoded recording_date
    transformed_inputs = torch.cat([density, one_hot], dim=1)

    return transformed_inputs

def augment_recording_date(train_data, val_data, replace_ratio=0.5):
    """
    Augments the 'recording_date' channel in the training data by partially replacing 0-1 values
    with values sampled from the validation dataset's distribution, keeping part of the original distribution.
    """
    train_augmented = train_data.clone()  
    
    val_recording_dates = val_data[:, 1, :, :].flatten()

    # Compute unique values and their probabilities in validation set
    unique_vals, counts = torch.unique(val_recording_dates, return_counts=True)
    probabilities = counts.float() / counts.sum()

    # Get mask for elements to be replaced (randomly select pixels to change)
    mask = torch.rand_like(train_augmented[:, 1, :, :]) < replace_ratio 

    # Sample new values for masked elements using validation distribution
    sampled_values = torch.multinomial(probabilities, mask.sum().item(), replacement=True)
    sampled_values = unique_vals[sampled_values].reshape(mask.sum().item())

    train_augmented[:, 1, :, :][mask] = sampled_values

    return train_augmented