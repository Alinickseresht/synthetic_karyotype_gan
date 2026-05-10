fake_validity = discriminator(fake_imgs.detach(), fake_labels)
            d_fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optim.step()
            
            # Train Generator
            g_optim.zero_grad()
            z = torch.randn(batch_size, cfg['model']['latent_dim'], device=device)
            fake_labels = torch.randint(0, cfg['model']['n_classes'], (batch_size,), device=device)
            fake_imgs = generator(z, fake_labels)
            fake_validity = discriminator(fake_imgs, fake_labels)
            g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
            g_loss.backward()
            g_optim.step()
            
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())
            
        # Save samples every 'save_interval' epochs
        if (epoch+1) % cfg['training']['save_interval'] == 0:
            with torch.no_grad():
                samples = generator(fixed_z, fixed_labels)
                save_image(samples, f'samples/epoch_{epoch+1}.png', nrow=8, normalize=True)
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
            }, f'models/checkpoints/ckpt_epoch_{epoch+1}.pth')
    
    print("Training finished.")

if name == 'main':
    args = parse_args()
    config = load_config(args.config)
    train(config, resume_ckpt=args.resume)


