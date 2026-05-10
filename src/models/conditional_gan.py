import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalGenerator(nn.Module):
    """Conditional GAN Generator.
    Takes latent vector z + class label (chromosome type + anomaly) and outputs image.
    Labels are one-hot encoded and concatenated with the latent vector.
    """
    def init(self, latent_dim=128, n_classes=28, img_channels=1, img_size=128):
        super().init()
        self.img_size = img_size
        # Embed label space into a vector
        self.label_embedding = nn.Embedding(n_classes, latent_dim)
        
        # Build generator as a series of transposed convolutions
        self.init_size = img_size // 16  # 128/16=8
        self.fc = nn.Linear(latent_dim * 2, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),   # 8 -> 16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),   # 16 -> 32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),   # 32 -> 64
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),   # 64 -> 128
            nn.Conv2d(32, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z shape: (batch, latent_dim)
        # labels shape: (batch,) with class indices
        label_emb = self.label_embedding(labels)   # (batch, latent_dim)
        gen_input = torch.cat([z, label_emb], dim=1)   # (batch, 2*latent_dim)
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator.
    Takes image and class label, outputs a single probability (real/fake).
    Label is projected and added to intermediate features.
    """
    def init(self, img_channels=1, img_size=128, n_classes=28):
        super().init()
        self.label_embedding = nn.Embedding(n_classes, img_size * img_size)
        
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, 64, 3, stride=2, padding=1),   # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 64 -> 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # 32 -> 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),   # 16 -> 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Linear(512 * 8 * 8, 1)

    def forward(self, img, labels):
        # Embed labels to a 2D map and concatenate with image
        label_map = self.label_embedding(labels).view(img.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat([img, label_map], dim=1)   # (batch, channels+1, H, W)
        features = self.model(d_in)
        features = features.view(features.size(0), -1)
        validity = self.fc(features)
        return validity
