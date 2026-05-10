import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
from tqdm import tqdm

# Local modules
from models.conditional_gan import ConditionalGenerator, ConditionalDiscriminator
from data_loader import ChromosomeDataset   # This returns (img, label)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Conditional GAN for synthetic karyotype generation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(epoch, generator, discriminator, g_optim, d_optim, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optim.state_dict(),
        'd_optimizer_state_dict': d_optim.state_dict(),
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

def generate_and_save_samples(generator, fixed_z, fixed_labels, epoch, sample_dir, nrow=8):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_z, fixed_labels)
        save_image(samples, os.path.join(sample_dir, f'samples_epoch_{epoch}.png'), 
                   nrow=nrow, normalize=True)
    generator.train()

def train(cfg, resume_ckpt=None):
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # DataLoader
    dataset = ChromosomeDataset(
        root_dir=cfg['data']['raw_path'],
        img_size=(cfg['data']['image_size'], cfg['data']['image_size'])
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data'].get('num_workers', 4),
        pin_memory=True
    )
    print(f"Loaded {len(dataset)} images from {cfg['data']['raw_path']}")
    
    # Models
    n_classes = cfg['model']['n_classes']
    latent_dim = cfg['model']['latent_dim']
    img_size = cfg['data']['image_size']
    
    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        n_classes=n_classes,
        img_channels=1,      # grayscale
        img_size=img_size
    ).to(device)
    
    discriminator = ConditionalDiscriminator(
        img_channels=1,
        img_size=img_size,
        n_classes=n_classes
    ).to(device)
    
    # Loss and optimizers
    adversarial_loss = nn.BCEWithLogitsLoss()
    g_optim = optim.Adam(generator.parameters(), lr=cfg['training']['learning_rate'], betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=cfg['training']['learning_rate'], betas=(0.5, 0.999))
    
    start_epoch = 0
    if resume_ckpt is not None:
        checkpoint = torch.load(resume_ckpt, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optim.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch} (checkpoint: {resume_ckpt})")
    
    # Fixed noise and labels for sample visualization during training
fixed_z = torch.randn(64, latent_dim, device=device)
    fixed_labels = torch.randint(0, n_classes, (64,), device=device)
    
    # Training loop
    for epoch in range(start_epoch, cfg['training']['epochs']):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        
        for batch_idx, (imgs, real_labels) in enumerate(progress_bar):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            real_labels = real_labels.to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optim.zero_grad()
            
            # Real images
            real_validity = discriminator(imgs, real_labels)
            d_real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
            
            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels)
            fake_validity = discriminator(fake_imgs.detach(), fake_labels)
            d_fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optim.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            g_optim.zero_grad()
            
            # Generate new fakes (with new noise) and try to fool discriminator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels)
            fake_validity = discriminator(fake_imgs, fake_labels)
            g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
            
            g_loss.backward()
            g_optim.step()
            
            # Store losses for logging
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f"{d_loss.item():.4f}",
                'G_loss': f"{g_loss.item():.4f}"
            })
        
        # End of epoch - print average losses
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished | Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}")
        
        # Save samples and checkpoint every N epochs
        if (epoch + 1) % cfg['training']['save_interval'] == 0:
            generate_and_save_samples(generator, fixed_z, fixed_labels, epoch+1, 'samples')
            save_checkpoint(epoch+1, generator, discriminator, g_optim, d_optim, 'checkpoints')
            print(f"Checkpoint and samples saved at epoch {epoch+1}")
    
    # Save final model
    save_checkpoint(cfg['training']['epochs'], generator, discriminator, g_optim, d_optim, 'checkpoints')
    print("Training completed. Final model saved.")

if name == 'main':
    args = parse_args()
    config = load_config(args.config)
    train(config, resume_ckpt=args.resume)
