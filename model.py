import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

from mamba_ssm.modules.mamba2 import Mamba2
from plan_mamba.cnn import InceptionBlock

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=384, patch_size=5, emb_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, emb_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(x)  # shape: (B, num_patches, emb_size)
    
class MambaBlock(nn.Module):
    def __init__(self, emb_size=16, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.mamba = Mamba2(
            d_model=emb_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Linear(emb_size * 2, emb_size)
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = residual + x
        x = x + self.mlp(self.norm(x))
        return self.dropout(x)
    
class PlantXMamba(nn.Module):
    def __init__(self, num_classes=4, patch_size=5, emb_size=16, num_blocks=4, dropout=0.1):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg_block = nn.Sequential(*list(vgg.features[:10]))
        self.inception = InceptionBlock(in_channels=128)
        self.patch_embed = PatchEmbedding(in_channels=384, patch_size=patch_size, emb_size=emb_size)
        self.transformer = nn.Sequential(*[MambaBlock(emb_size=emb_size, d_state=64, d_conv=4, expand=2) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(emb_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.vgg_block(x)    # (B, 128, 56, 56)
        x = self.inception(x)    # (B, 384, 56, 56)
        x = self.patch_embed(x)  # (B, 121, 16)
        x = self.transformer(x)  # (B, 121, 16)
        x = self.norm(x)         # (B, 121, 16)
        x = x.permute(0, 2, 1)   # (B, 16, 121)
        x = self.global_pool(x).squeeze(-1)  # (B, 16)
        return self.classifier(x)  # (B, num_classes)

# Kiểm tra mô hình
if __name__ == "__main__":
    model = PlantXMamba(num_classes=4, patch_size=5, emb_size=16, num_blocks=4, dropout=0.1)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(output.shape)  # ([2, 4])