import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Определяем кастомный датасет
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(os.listdir(root_dir))}

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, not_fin=True, kernel_size=3, stride=2):
        super(UNetBlock, self).__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            if not_fin:
                self.block = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1),
                    nn.ReLU(inplace=True)
                )
            else:
                self.block = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1),
                    )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x


class Diffusion_Model(nn.Module):
    def __init__(self, img_channels=3, img_size=64, embed_dim=128, temb_dim=4, hidden_dim=256, class_label=None, device='mps', depth=4):
        super(Diffusion_Model, self).__init__()
        self.img_channels = img_channels
        self.embed_dim = embed_dim
        self.temb_dim = temb_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_size

        # Adjust the number of channels (3 channels)
        self.down1 = UNetBlock(img_channels, embed_dim, down=True, use_dropout=True, kernel_size=3, stride=2)
        self.down2 = UNetBlock(embed_dim, embed_dim*2, down=True, use_dropout=True, kernel_size=3, stride=2)
        self.down3 = UNetBlock(embed_dim*2, embed_dim*4, down=True, use_dropout=True, kernel_size=3, stride=2)
        self.down4 = UNetBlock(embed_dim*4, embed_dim*8, down=True, use_dropout=True, kernel_size=3, stride=2)

        self.up1 = UNetBlock(embed_dim*8 + embed_dim, embed_dim*4, down=False, use_dropout=False, kernel_size=3, stride=2)
        self.up2 = UNetBlock((embed_dim + embed_dim)*4, embed_dim*2, down=False, use_dropout=False, kernel_size=3, stride=2)
        self.up3 = UNetBlock((embed_dim + embed_dim)*2, embed_dim, down=False, use_dropout=False, kernel_size=3, stride=2)
        self.up4 = UNetBlock((embed_dim + embed_dim), img_channels, down=False, use_dropout=False, not_fin=False, kernel_size=3, stride=2)

        self.temb_proj = nn.Sequential(
            nn.Linear(temb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Attention and MLP layers
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=(img_size//16)**2, num_heads=1, batch_first=True, dropout=0.) for _ in range(depth)])

        self.device = device

    def forward(self, x, t, class_label=None):
        t_emb = self.get_time_embedding(t, self.temb_dim).to(x.device)
        t_emb = self.temb_proj(t_emb)

        # Encoder path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Reshape and add time embedding
        x4 = x4.reshape(x4.size(0), x4.size(1), -1)
        t_emb = t_emb.unsqueeze(-1).repeat(1, 1, x4.size(2))
        x4 = torch.cat([x4, t_emb], dim=1)

        # Attention and MLP layers
        for attn in self.attention_layers:
            residual = x4
            x4, _ = attn(x4, x4, x4)
            x4 += residual

        # Reshape back to image format
        x4 = x4.transpose(1, 2).reshape(x4.size(0), -1, int(math.sqrt(x4.size(2))), int(math.sqrt(x4.size(2))))

        # Decoder path
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)

        return x4, x

    @staticmethod
    def get_time_embedding(timesteps, embedding_dim):
        device = timesteps.device
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb

    def classify(self, x, t, class_label):
        features, _ = self.forward(x, t, class_label)
        logits = self.classifier(features)
        return logits
