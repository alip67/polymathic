import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNNClassifier(torch.nn.Module):
    """ Basic CNN classifier with 2 convolutional layers, batch_norm, dropout, and 2 fully connected layers"""
    def __init__(self, 
                input_dim:int=2,
                proj_dim:int=16,
                mlp_dim:int=128,
                num_classes:int=1,
                input_shape:int=28,
                dropout:float=0.1):
        super(BasicCNNClassifier, self).__init__()
        self.input_shape = input_shape
        self.proj_dim = proj_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # First convolutional block
        self.conv1 = nn.Conv2d(input_dim, proj_dim, kernel_size=3, padding=1,)
        self.bn1 = nn.BatchNorm2d(proj_dim)
        self.dropout1 = nn.Dropout2d(p=dropout)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(proj_dim, proj_dim*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(proj_dim*2)
        self.dropout2 = nn.Dropout2d(p=dropout)
        
        # MLP classifier head
        self.flatten = nn.Flatten()
        # Note: The actual size will depend on your input image dimensions
        # Assuming 28x28 input images, after 2 conv layers with stride 1 and maxpool
        self.fc1 = nn.Linear(proj_dim*2*(input_shape//4)**2, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        
        # MLP classifier
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return x