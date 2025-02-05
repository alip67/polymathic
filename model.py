import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=2, in_channels=2, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2 

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn_proj = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn_proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TinyViT(nn.Module):
    def __init__(self, input_dim:int=2, 
                 input_shape:int=28, 
                 patch_size:int=4, 
                 embed_dim:int=64, 
                 num_heads:int=4, 
                 num_layers:int=4, 
                 num_classes:int=1,
                 dropout:float=0.3):
        super(TinyViT, self).__init__()
        
        assert input_shape % patch_size == 0
        


        self.patch_size = patch_size
        self.num_patches = (input_shape // patch_size) ** 2
        self.embed_dim = embed_dim


        self.bn_patch = nn.BatchNorm2d(input_dim) # BatchNorm after embedding
        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.dropout1 = nn.Dropout2d(p=dropout)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4,
                dropout = dropout,
                activation="gelu", 
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Convert image to patches and embed
        x = self.bn_patch(x)
        x = self.patch_embed(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = x.flatten(2).transpose(1, 2) 

        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Global Average Pooling
        x = x.mean(dim=1)  

        x = self.fc_out(x)  
        return x.squeeze(1)