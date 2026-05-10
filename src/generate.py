
import argparse
import torch
import yaml
from models.conditional_gan import ConditionalGenerator
from torchvision.utils import save_image
import os

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create generator
    generator = ConditionalGenerator(latent_dim=cfg['model']['latent_dim'],
                                     n_classes=cfg['model']['n_classes'],
                                     img_size=cfg['data']['image_size']).to(device)
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Define anomaly classes: e.g., label 0 = normal chr1, label 1 = trisomy21, etc.
    # You need to map your own encoding.
    anomaly_label = args.anomaly_label   # integer
    num_images = args.num
    z = torch.randn(num_images, cfg['model']['latent_dim'], device=device)
    labels = torch.full((num_images,), anomaly_label, dtype=torch.long, device=device)
    
    with torch.no_grad():
        imgs = generator(z, labels)
    
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(num_images):
        save_image(imgs[i], os.path.join(args.output_dir, f'gen_{anomaly_label}_{i}.png'), normalize=True)
    print(f"Generated {num_images} images with anomaly label {anomaly_label} in {args.output_dir}")

if name == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--anomaly_label', type=int, required=True, help='class index for desired anomaly')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='data/synthetic')
    args = parser.parse_args()
    generate(args)
