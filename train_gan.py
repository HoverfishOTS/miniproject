import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid

from model_gan import GeneratorResNet, DiscriminatorPatchGAN
from xbd_dataset import XBDDataset

def train_gan(epochs=100, batch_size=16, lr=2e-4, lambda_pixel=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training GAN on device: {device}")

    # 1. Initialize W&B
    wandb.init(
        project="xbd-gan-post2pre",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "lambda_pixel": lambda_pixel,
            "architecture": "Pix2Pix",
            "dataset": "tier3"
        }
    )

    # 2. Setup Tier3 Dataset
    print("Loading exclusive tier3 dataset...")
    train_dataset = XBDDataset(split='tier3', img_size=256)
    
    # Setup Balanced Sampler
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # WSL Optimizations: num_workers=0 and pin_memory=False to prevent crashes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, # Must be False when using sampler
        sampler=sampler,
        num_workers=0, 
        pin_memory=False
    )

    # 3. Models
    generator = GeneratorResNet().to(device)
    discriminator = DiscriminatorPatchGAN().to(device)

    # 4. Optimizers and Losses
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()

    optimizer_G = optim.AdamW(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Mixed precision scalers for RTX 4090 optimization
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    os.makedirs("checkpoints_GAN_tier3", exist_ok=True)

    print("Starting GAN training loop...")
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        train_loss_G = 0
        train_loss_D = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (pre_img, post_img, masks) in enumerate(pbar):
            # Scale images from [0, 1] to [-1, 1] to match Generator's Tanh output
            real_pre = (pre_img.to(device) * 2.0) - 1.0
            real_post = (post_img.to(device) * 2.0) - 1.0
            
            # Create Mask Weights for L1 Loss
            masks = masks.to(device)
            # Add channel dimension: (B, H, W) -> (B, 1, H, W)
            weights = torch.ones_like(masks, dtype=torch.float32).unsqueeze(1)
            # Apply 100x multiplier where masks > 0 (damaged pixels)
            weights[masks.unsqueeze(1) > 0] = 100.0
            
            # Real and Fake labels for Discriminator
            # Output of Discriminator is 16x16 grid
            b_size = real_pre.size(0)
            valid = torch.ones((b_size, 1, 16, 16), device=device, requires_grad=False)
            fake = torch.zeros((b_size, 1, 16, 16), device=device, requires_grad=False)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            with torch.amp.autocast('cuda'):
                # GAN takes in Post-Disaster and generates Pre-Disaster
                fake_pre = generator(real_post)
                
                # Discriminator evaluates (Post, Fake_Pre)
                pred_fake = discriminator(real_post, fake_pre)
                
                loss_GAN = criterion_GAN(pred_fake, valid)
                # Pixel-wise L1 loss to encourage structural similarity (Mask-Weighted)
                loss_pixel = torch.mean(weights * torch.abs(fake_pre - real_pre))
                
                loss_G = loss_GAN + lambda_pixel * loss_pixel
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            with torch.amp.autocast('cuda'):
                # Real pair
                pred_real = discriminator(real_post, real_pre)
                loss_real = criterion_GAN(pred_real, valid)
                
                # Fake pair
                pred_fake2 = discriminator(real_post, fake_pre.detach())
                loss_fake = criterion_GAN(pred_fake2, fake)
                
                loss_D = 0.5 * (loss_real + loss_fake)
                
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            # Logging
            train_loss_G += loss_G.item()
            train_loss_D += loss_D.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'D_Loss': f"{loss_D.item():.4f}", 
                    'G_Loss': f"{loss_G.item():.4f}"
                })
                wandb.log({
                    "batch_D_loss": loss_D.item(),
                    "batch_G_loss": loss_G.item()
                })

        # Epoch Summaries
        avg_loss_G = train_loss_G / len(train_loader)
        avg_loss_D = train_loss_D / len(train_loader)
        
        print(f"Epoch {epoch} - D_loss: {avg_loss_D:.4f} | G_loss: {avg_loss_G:.4f}")
        wandb.log({
            "epoch": epoch,
            "epoch_G_loss": avg_loss_G,
            "epoch_D_loss": avg_loss_D
        })

        # Visualization Logging at end of epoch
        generator.eval()
        with torch.no_grad():
            # Grab last batch to visualize
            sample_post = real_post[:4]
            sample_real_pre = real_pre[:4]
            sample_fake_pre = generator(sample_post)
            
            # Convert back from [-1, 1] to [0, 1] for visual plotting
            sample_post_vis = (sample_post * 0.5) + 0.5
            sample_real_pre_vis = (sample_real_pre * 0.5) + 0.5
            sample_fake_pre_vis = (sample_fake_pre * 0.5) + 0.5
            
            # Create a grid: Row 1 = Post, Row 2 = Generated Pre, Row 3 = Real Pre
            grid_img = torch.cat([sample_post_vis, sample_fake_pre_vis, sample_real_pre_vis], dim=0)
            grid = make_grid(grid_img, nrow=4, normalize=False)
            
            # Log to W&B
            wandb.log({"GAN Translation (Top=Post, Mid=FakePre, Bot=RealPre)": wandb.Image(grid.cpu())})

        # Checkpointing
        if epoch % 10 == 0 or epoch == epochs:
            checkpoint_path = f"checkpoints_GAN_tier3/gan_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, checkpoint_path)
            print(f"[*] Checkpoint saved at {checkpoint_path}")

    wandb.finish()
    print("Training Complete!")

if __name__ == "__main__":
    # Optimal settings for RTX 4090 (24GB VRAM)
    # Using batch size of 16 for Pix2Pix 256x256
    train_gan(epochs=100, batch_size=16, lr=2e-4, lambda_pixel=50)
