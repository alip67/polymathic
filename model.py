import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=2, in_channels=2, embed_dim=64):  # 🚀 Change patch_size=2
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # Update number of patches

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 🚀 Add MaxPooling
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 🚀 Add MaxPooling
        )

        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.bn_proj = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        x = self.bn_proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# class TinyViT(nn.Module):
#     def __init__(self, image_size=28, patch_size=4, embed_dim=64, num_heads=4, num_layers=4, num_classes=1):
#         super(TinyViT, self).__init__()

#         assert image_size % patch_size == 0, "Image size must be divisible by patch size"

#         self.patch_size = patch_size
#         self.num_patches = (image_size // patch_size) ** 2
#         self.embed_dim = embed_dim

#         # Patch Embedding (Conv2d + BatchNorm2d)
#         self.patch_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim, 
#                                      kernel_size=patch_size, stride=patch_size)
#         self.bn_patch = nn.BatchNorm2d(embed_dim)  # BatchNorm after embedding

#         # Transformer Encoder
#         self.encoder_layers = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
#         ])

#         # Global Average Pooling (No CLS Token)
#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

#         # Single-Neuron Output Layer
#         self.fc_out = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # Convert image to patches and embed
#         x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, H/P, W/P)
#         x = self.bn_patch(x)  # Apply Batch Normalization
#         x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

#         # Pass through Transformer Encoder Layers
#         for layer in self.encoder_layers:
#             x = layer(x)

#         # Global Average Pooling (Removes CLS Token)
#         x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)  # Shape: (batch_size, embed_dim)

#         # Single Neuron Output (Regression or Binary Classification)
#         x = self.fc_out(x)  # Shape: (batch_size, 1)
#         return x.squeeze(1)


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(TransformerBlock, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.BatchNorm1d(embed_dim * 4),  # BatchNorm for stability
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.BatchNorm1d(embed_dim)  # BatchNorm after final projection
#         )

#     def forward(self, x):
#         # Self-Attention Block
#         attn_output, _ = self.attention(x, x, x)
#         x = self.norm1(x + attn_output)

#         # Feed-Forward Network
#         ff_output = self.feed_forward(x.transpose(1, 2)).transpose(1, 2)  # Apply BatchNorm1d correctly
#         x = self.norm2(x + ff_output)
#         return x



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
        
        assert input_shape % patch_size == 0, "Image size must be divisible by patch size"
        


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

        # Output layer (single neuron, no classification token)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Convert image to patches and embed
        x = self.bn_patch(x)
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, H/P, W/P)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Global Average Pooling (removes CLS token need)
        x = x.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Output single neuron prediction
        x = self.fc_out(x)  # Shape: (batch_size, 1)
        return x.squeeze(1)



class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64, num_heads=4,
                 depth=4, mlp_dim=64, num_classes=1, dropout=0.2):  # 🚀 Reduce depth & heads
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)  # 🚀 Increase dropout

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)  

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(mlp_dim, num_classes),
            nn.Tanh()  # 🚀 Prevents extreme logits
        )

        # nn.init.xavier_uniform_(self.mlp_head[1].weight)

    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # 🚀 This should have correct shape now
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)

        # 🚀 Dynamically adjust `pos_embed` to match `x.shape[1]`
        pos_embed = self.pos_embed[:, :x.shape[1], :]
        
        x = x + pos_embed  # ✅ Now it will not throw an error
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])  # Take CLS token output
        return self.mlp_head(x).squeeze(1)