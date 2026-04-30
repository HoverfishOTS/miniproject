import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from model_siamese import SiameseUNetTransformer
from xbd_dataset import XBDDataset

class FocalDiceLoss(nn.Module):
    """
    Combines Focal Loss with Dice Loss.
    Focal Loss helps with well-classified pixels, while Dice Loss directly optimizes 
    for Intersection over Union (IoU), which severely penalizes the network if it 
    ignores the rare foreground classes (Damage).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Optional tensor of static class weights [C]
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 1. FOCAL LOSS
        # Calculate unweighted cross entropy to retrieve true probabilities
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        
        # Calculate weighted cross entropy for the actual loss magnitude
        ce_loss_weighted = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted
        focal_loss = focal_loss.mean()

        # 2. DICE LOSS
        # Convert logits to probabilities (Softmax over classes)
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding: (B, H, W) -> (B, C, H, W)
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and union (summing over batch, height, and width)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        
        # Apply alpha weights to dice loss
        if self.alpha is not None:
            dice_loss = torch.sum((1. - dice_score) * self.alpha) / torch.sum(self.alpha)
        else:
            dice_loss = torch.mean(1. - dice_score)

        return focal_loss + dice_loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels, weights=None):
    """
    Multi-class Lovasz-Softmax loss
    """
    if probs.numel() == 0:
        return probs * 0.
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if fg.sum() == 0:
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss_c = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if weights is not None:
            loss_c = loss_c * weights[c]
        losses.append(loss_c)
    if len(losses) == 0:
        return probs.sum() * 0.
    return sum(losses) / len(losses)

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, weight=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        # flatten
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, probs.size(1))
        targets = targets.view(-1)
        
        loss_ce = F.cross_entropy(logits, targets.view_as(logits[:,0,:,:]), weight=self.weight)
        loss_lovasz = lovasz_softmax_flat(probs, targets, weights=self.weight)
        return loss_ce + loss_lovasz


def train_siamese(epochs=200, start_epoch=1, batch_size=24, lr=1e-4, resume_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Explicitly requested: A brand new project on W&B
    wandb.init(
        project="xbd-siamese-damage-assessment",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "architecture": "Siamese U-Net Transformer"
        }
    )

    print("Initializing Phase 3 Datasets...")
    train_dataset = XBDDataset(split='train')
    
    # Because of the large model, batch size ought to be kept relatively small to avoid OOM
    # NOTE: Set pin_memory=False by default to avoid the WSL VRAM crash we diagnosed earlier!
    # Set num_workers=0 because WSL has severe shared memory corruption issues leading to CUDA AcceleratorErrors
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )

    model = SiameseUNetTransformer(num_classes=4).to(device)
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[*] Resuming weights from {resume_checkpoint} ...")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=device, weights_only=True))
    
    # We output 4 classes per pixel.
    # The confusion matrix showed the model is terrified of predicting damage! It predicts "None/BG"
    # for 30-50% of inherently damaged pixels to play it safe due to class imbalance.
    # We drop None/BG to 0.1 and bump Destroyed to 3.5 to force learning.
    class_weights = torch.tensor([0.1, 1.5, 2.0, 3.5], dtype=torch.float32).to(device)
    # Bump Gamma from 2.0 to 3.0 to aggressively target "Hard" misclassified examples in the Loss
    criterion = FocalDiceLoss(alpha=class_weights, gamma=3.0)
    # Use AdamW for better weight decay regularization along with our scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Ramps the learning rate down smoothly to 1% of the original lr across the REMAINING epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - start_epoch + 1), eta_min=1e-6)

    os.makedirs("checkpoints_siamese", exist_ok=True)

    print(f"Starting Siamese Transformer training for {epochs} epochs...")
    
    # Initialize AMP Scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for pre_img, post_img, masks in pbar:
            pre_img, post_img, masks = pre_img.to(device), post_img.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # 1. Pipeline execution: Pre and Post injected simultaneously with Mixed Precision
            with torch.amp.autocast('cuda'):
                logits = model(pre_img, post_img) # (B, 4, H, W)
                loss = criterion(logits, masks)
            
            # 3. Backprop with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if pbar.n % 10 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                wandb.log({"train_batch_loss": loss.item()})
                
        avg_loss = train_loss / len(train_loader)
        
        # Step the learning rate scheduler and log it!
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch}/{epochs}] - Avg Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
        wandb.log({
            "epoch": epoch, 
            "epoch_avg_loss": avg_loss,
            "learning_rate": current_lr
        })
        
        # Checkpointing
        if epoch % 50 == 0 or epoch == epochs:
            checkpoint_path = f"checkpoints_siamese/siamese_epoch_V5_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Log as W&B Artifact to the new Project
            artifact = wandb.Artifact(name=f"siamese-model-{wandb.run.id}", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

    wandb.finish()
    print("Training Complete!")

if __name__ == "__main__":
    # We use a batch size of 24 because of the 24GB on the RTX 4090 + Mixed Precision.
    # Continuing from epoch 100 for an additional 100 epochs.
    train_siamese(
        epochs=200, 
        start_epoch=101, 
        batch_size=24, 
        lr=1e-4, 
        resume_checkpoint="checkpoints_siamese/siamese_epoch_V5_100.pth"
    )
