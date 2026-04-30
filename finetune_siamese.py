import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from model_siamese import SiameseUNetTransformer
from model_gan import GeneratorResNet
from xbd_dataset import XBDDataset
from train_siamese import FocalDiceLoss, LovaszSoftmaxLoss

def finetune_siamese(epochs=30, batch_size=24, lr=5e-5, gan_checkpoint=None, siamese_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-Tuning on device: {device}")

    # Explicitly requested: A brand new project/run on W&B
    wandb.init(
        project="xbd-siamese-damage-assessment",
        job_type="finetuning_gan_outputs",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "Siamese U-Net Transformer",
            "finetune": True
        }
    )

    print("Initializing Phase 3 Datasets...")
    train_dataset = XBDDataset(split='train')
    
    # WSL Optimizations: num_workers=0 and pin_memory=False to prevent crashes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )

    print("Loading Models...")
    # Load Frozen GAN
    generator = GeneratorResNet().to(device)
    if gan_checkpoint and os.path.exists(gan_checkpoint):
        print(f"[*] Loading GAN weights from {gan_checkpoint} ...")
        gan_state = torch.load(gan_checkpoint, map_location=device, weights_only=True)
        # Handle dict or pure model state
        if 'generator_state_dict' in gan_state:
            generator.load_state_dict(gan_state['generator_state_dict'])
        else:
            generator.load_state_dict(gan_state)
    generator.eval() # Freeze GAN
    for param in generator.parameters():
        param.requires_grad = False

    # Load Siamese Transformer to fine-tune
    model = SiameseUNetTransformer(num_classes=4).to(device)
    if siamese_checkpoint and os.path.exists(siamese_checkpoint):
        print(f"[*] Loading Siamese weights from {siamese_checkpoint} ...")
        model.load_state_dict(torch.load(siamese_checkpoint, map_location=device, weights_only=True))
    
    # Loss, Optimizer, Scheduler
    class_weights = torch.tensor([0.01, 2.0, 3.0, 5.0], dtype=torch.float32).to(device)
    criterion = LovaszSoftmaxLoss(weight=class_weights)
    
    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Use a new directory to strictly avoid overwriting v5 weights!
    checkpoint_dir = "checkpoints_siamese_finetuned"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting Siamese Transformer Fine-Tuning for {epochs} epochs...")
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for pre_img, post_img, masks in pbar:
            # We don't need real pre_img, we will generate fake_pre!
            post_img, masks = post_img.to(device), masks.to(device)
            
            # 1. Generate Fake Pre
            # Scale real_post to [-1, 1] for GAN
            post_img_scaled = (post_img * 2.0) - 1.0
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_pre_scaled = generator(post_img_scaled)
                    # Scale back to [0, 1] for Siamese
                    fake_pre = (fake_pre_scaled * 0.5) + 0.5
            
            optimizer.zero_grad()
            
            # 2. Train Siamese on (Fake Pre, Real Post)
            with torch.amp.autocast('cuda'):
                logits = model(fake_pre, post_img) # (B, 4, H, W)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if pbar.n % 10 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                wandb.log({"finetune_batch_loss": loss.item()})
                
        avg_loss = train_loss / len(train_loader)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch}/{epochs}] - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
        wandb.log({
            "epoch": epoch, 
            "epoch_avg_loss": avg_loss,
            "learning_rate": current_lr
        })
        
        # Checkpointing
        if epoch % 5 == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"siamese_finetuned_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            
            # Log as W&B Artifact
            artifact = wandb.Artifact(name=f"siamese-finetuned-{wandb.run.id}", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Fine-Tuning Complete!")

if __name__ == "__main__":
    finetune_siamese(
        epochs=100, 
        batch_size=24, 
        lr=5e-5, # Smaller LR for fine-tuning
        gan_checkpoint="checkpoints_GAN_tier3/gan_epoch_100.pth",
        siamese_checkpoint="checkpoints_siamese/siamese_epoch_V5_200.pth"
    )
