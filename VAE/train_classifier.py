import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from dataset import LADIDataset
from VAE.model_vae import VAE

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_classes=12):
        super(LatentClassifier, self).__init__()
        # A simple Multi-Layer Perceptron (MLP) 
        # Maps the VAE Latent Distribution Means to the multi-label classes
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        # We process the latent representations z through the MLP
        x = torch.relu(self.fc1(z))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)  # No sigmoid here, handled by BCEWithLogitsLoss
        return logits


def train_stage2(vae_weights_path="checkpoints/vae_epoch_150.pth", epochs=50, batch_size=32, lr=1e-4, latent_dim=256, num_classes=12):
    # 1. Initialize W&B
    wandb.init(
        project="miniproject-ladi-classifier",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "latent_dim": latent_dim
        }
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 2. Load the Pretrained VAE and FREEZE its weights
    print(f"Loading VAE weights from {vae_weights_path}...")
    vae = VAE(latent_dim=latent_dim).to(device)
    
    if os.path.exists(vae_weights_path):
        vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    else:
        print(f"WARNING: VAE weights {vae_weights_path} not found. Ensure Stage 1 training has finished. Using un-pretrained weights.")
    
    vae.eval() # Set VAE to evaluation mode (no dropout, batchnorm updates)
    for param in vae.parameters():
        param.requires_grad = False # Freeze VAE parameters!

    # 3. Initialize the Classifier Head
    classifier = LatentClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)
    # Using BCEWithLogitsLoss for Multi-Label Classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # 4. Load Datasets
    print("Initializing Datasets...")
    train_dataset = LADIDataset(split='train')
    val_dataset = LADIDataset(split='validation')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 5. Training Loop
    print("Starting Classifier Training...")
    
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        classifier.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (Train)")
        
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Step 1: Extract Latent Representations z (specifically mu) from frozen VAE
            with torch.no_grad():
                mu, _ = vae.encoder(data)
                
            # Step 2: Pass MU through the classifier
            logits = classifier(mu)
            
            # Step 3: Compute Loss & Update
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if pbar.n % 10 == 0:
                 wandb.log({"train_batch_loss": loss.item() / len(data)})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        classifier.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} (Val)")
        with torch.no_grad():
            for data, labels in pbar_val:
                data, labels = data.to(device), labels.to(device)
                
                # Extract latents
                mu, _ = vae.encoder(data)
                
                # Inference
                logits = classifier(mu)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Convert logits to binary predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Metrics
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro') # Macro averages across all damage classes
        
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_accuracy:.4f} | Val F1 {val_f1:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1
        })

        # Save best model
        os.makedirs("checkpoints", exist_ok=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(classifier.state_dict(), "checkpoints/classifier_best.pth")
            print(f"[*] New best model saved with F1: {best_f1:.4f}")
            
            # W&B ARTIFACTS: Upload the best classifier to the W&B cloud
            artifact = wandb.Artifact(name=f"classifier-best-{wandb.run.id}", type="model")
            artifact.add_file("checkpoints/classifier_best.pth")
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Stage 2 Training Complete. Best model saved to ./checkpoints/classifier_best.pth and W&B Cloud.")

if __name__ == "__main__":
    train_stage2(vae_weights_path="checkpoints/vae_epoch_150.pth", epochs=50)
