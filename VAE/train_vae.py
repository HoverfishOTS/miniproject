import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from dataset import LADIDataset
from VAE.model_vae import VAE

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE loss function.
    BCE: Binary Cross Entropy (Reconstruction Loss). 
         We use MSE here instead of BCE to match image pixels natively since BCE expects probabilities.
    KLD: Kullback-Leibler Divergence.
    """
    # Reconstruction loss (Mean Squared Error)
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss = Reconstruction + Beta * KL_Divergence
    return MSE + beta * KLD, MSE, KLD

def train_vae(epochs=50, start_epoch=1, batch_size=32, lr=1e-4, latent_dim=256, beta=1.0, resume_checkpoint=None):
    # 1. Initialize Weights & Biases for MLOps tracking
    wandb.init(
        project="miniproject-ladi-vae",
        config={
            "epochs": epochs,
            "start_epoch": start_epoch,
            "batch_size": batch_size,
            "learning_rate": lr,
            "latent_dim": latent_dim,
            "beta_kld_weight": beta,
            "resumed_from": resume_checkpoint
        }
    )

    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 3. Load Dataset
    print("Initializing Dataset...")
    train_dataset = LADIDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 4. Initialize Model & Optimizer
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Resume from Checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}' to continue training...")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Successfully loaded checkpoint.")
    elif resume_checkpoint:
        print(f"WARNING: Checkpoint '{resume_checkpoint}' not found. Starting from scratch.")

    # 6. Training Loop
    print(f"Starting Training from epoch {start_epoch} to {epochs}...")
    model.train()
    
    for epoch in range(start_epoch, epochs + 1):
        train_loss = 0
        train_mse = 0
        train_kld = 0
        
        # tqdm progress bar for neat console output
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            # Calculate Loss
            loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=beta)
            
            # Backward pass & Optimizer Step
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item() / len(data):.4f}", 
                    'MSE': f"{mse.item() / len(data):.4f}",
                    'KLD': f"{kld.item() / len(data):.4f}"
                })
                
                # Log to W&B
                wandb.log({
                    "batch_loss": loss.item() / len(data),
                    "batch_mse": mse.item() / len(data),
                    "batch_kld": kld.item() / len(data),
                    "epoch": epoch
                })

        # Epoch summaries
        avg_loss = train_loss / len(train_loader.dataset)
        avg_mse = train_mse / len(train_loader.dataset)
        avg_kld = train_kld / len(train_loader.dataset)
        
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, KLD: {avg_kld:.4f})")
        
        # Log summary metrics to W&B
        wandb.log({
            "epoch": epoch,
            "epoch_avg_loss": avg_loss,
            "epoch_avg_mse": avg_mse,
            "epoch_avg_kld": avg_kld
        })

        # W&B ALERTS: Alert us if loss explodes (NaN or extreme values meaning gradient explosion)
        if avg_loss > 10000 or torch.isnan(torch.tensor(avg_loss)):
            wandb.alert(
                title=f"Loss Explosion at Epoch {epoch}", 
                text=f"Average loss spiked to {avg_loss:.4f}. You may need to lower your learning rate.",
                level=wandb.AlertLevel.ERROR
            )
        
        # Log Visual Reconstructions to W&B (Logging the last batch of the epoch)
        with torch.no_grad():
            n = min(data.size(0), 8) # Plot up to 8 images
            # Generate a grid: Original images on top, Reconstructions on bottom
            comparison_images = []
            for i in range(n):
                comparison_images.append(wandb.Image(data[i].cpu(), caption="Original"))
                comparison_images.append(wandb.Image(recon_batch[i].cpu(), caption="Reconstruction"))
            
            wandb.log({"reconstructions": comparison_images, "epoch": epoch})

        # Save model checkpoint every 200 epochs (or on the final epoch)
        os.makedirs("checkpoints", exist_ok=True)
        if epoch % 200 == 0 or epoch == epochs:
            checkpoint_path = f"checkpoints/vae_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # W&B ARTIFACTS: Upload the model directly to the W&B cloud!
            # This logs your weights so you can download them anywhere without needing the physical file
            artifact = wandb.Artifact(name=f"vae-model-{wandb.run.id}", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Training Complete. Models saved to ./checkpoints/ and W&B Cloud.")

if __name__ == "__main__":
    # Resuming from the previous 150 epochs run, and continuing until epoch 2000
    # The program will now start counting from 151.
    train_vae(
        epochs=2000, 
        start_epoch=151, 
        batch_size=32, 
        lr=1e-4, 
        latent_dim=256, 
        beta=1.0, 
        resume_checkpoint="checkpoints/vae_epoch_150.pth"
    )
