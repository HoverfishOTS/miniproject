import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import wandb

from model_siamese import SiameseUNetTransformer
from xbd_dataset import XBDDataset

def calculate_iou(preds, labels, num_classes=4):
    """
    Computes Intersection over Union (IoU) for each class.
    preds: (B, H, W) integer indices
    labels: (B, H, W) integer indices
    """
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            ious.append(float('nan'))  # Ignore class if it doesn't appear in either
        else:
            ious.append(intersection / union)
    return ious

def test_siamese(checkpoint_path, batch_size=4, num_visualizations=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Initialize W&B run for testing
    wandb.init(
        project="xbd-siamese-damage-assessment",
        job_type="evaluation",
        name=f"test-eval-{os.path.basename(checkpoint_path)}"
    )

    print("Initializing Test Dataset...")
    test_dataset = XBDDataset(split='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=False
    )

    print("Loading Siamese U-Net Transformer...")
    model = SiameseUNetTransformer(num_classes=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs("test_outputs", exist_ok=True)
    
    total_samples = 0
    total_correct = 0
    total_pixels = 0
    all_ious = {0: [], 1: [], 2: [], 3: []}
    conf_matrix = torch.zeros((4, 4), dtype=torch.int64, device=device)
    
    print("\nStarting evaluation over Test Set...")
    
    vis_count = 0
    with torch.no_grad():
        for pre_img, post_img, masks in tqdm(test_loader, desc="Testing"):
            pre_img, post_img, masks = pre_img.to(device), post_img.to(device), masks.to(device)
            
            # Forward pass with Test-Time Augmentation (TTA)
            # This provides a completely free IoU boost by averaging predictions across spatial orientations
            
            # 1. Base Prediction
            logits_base = model(pre_img, post_img)
            
            # 2. Horizontal Flip Prediction
            logits_hflip = model(TF.hflip(pre_img), TF.hflip(post_img))
            logits_hflip = TF.hflip(logits_hflip) # Flip it back to align with base!
            
            # 3. Vertical Flip Prediction
            logits_vflip = model(TF.vflip(pre_img), TF.vflip(post_img))
            logits_vflip = TF.vflip(logits_vflip) # Flip it back!
            
            # Average the logits for an ensembled prediction mask
            logits = (logits_base + logits_hflip + logits_vflip) / 3.0
            preds = torch.argmax(logits, dim=1) # (B, H, W)
            
            # Accuracy metric
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            total_samples += pre_img.size(0)
            
            # IoU metric
            batch_ious = calculate_iou(preds, masks)
            for c in range(4):
                if not np.isnan(batch_ious[c]):
                    all_ious[c].append(batch_ious[c])
                    
            # Confusion Matrix
            preds_f = preds.flatten()
            masks_f = masks.flatten()
            valid = (masks_f >= 0) & (masks_f < 4)
            cm_batch = torch.bincount(4 * masks_f[valid] + preds_f[valid], minlength=16)
            conf_matrix += cm_batch.reshape(4, 4)
            
            # Visualizations
            if vis_count < num_visualizations:
                for i in range(pre_img.size(0)):
                    if vis_count >= num_visualizations: 
                        break
                    
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # Convert tensors to numpy for plotting
                    pre_disp = pre_img[i].cpu().permute(1, 2, 0).numpy()
                    post_disp = post_img[i].cpu().permute(1, 2, 0).numpy()
                    mask_disp = masks[i].cpu().numpy()
                    pred_disp = preds[i].cpu().numpy()
                    
                    # Add standard damage colormap matching indices
                    cmap = plt.cm.get_cmap('tab10', 4)
                    
                    axes[0].imshow(pre_disp)
                    axes[0].set_title("Pre-Disaster")
                    axes[0].axis('off')
                    
                    axes[1].imshow(post_disp)
                    axes[1].set_title("Post-Disaster")
                    axes[1].axis('off')
                    
                    axes[2].imshow(mask_disp, clim=(0, 3), cmap=cmap)
                    axes[2].set_title("Ground Truth Mask")
                    axes[2].axis('off')
                    
                    im = axes[3].imshow(pred_disp, clim=(0, 3), cmap=cmap)
                    axes[3].set_title("Model Prediction")
                    axes[3].axis('off')
                    
                    # Colorbar
                    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], orientation='vertical', fraction=0.015, pad=0.04)
                    cbar.ax.set_yticklabels(['None/BG', 'Minor', 'Major', 'Destroyed']) 
                    
                    fig_path = f"test_outputs/eval_vis_{vis_count}.png"
                    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
                    
                    # Log figure to wandb
                    wandb.log({f"Test Visualizations/Sample_{vis_count}": wandb.Image(fig_path, caption=f"Sample {vis_count}")})
                    
                    plt.close()
                    vis_count += 1

    # Aggregate metrics
    pixel_acc = total_correct / total_pixels
    mean_ious = {c: np.mean(all_ious[c]) if all_ious[c] else 0.0 for c in range(4)}
    mIoU = np.mean(list(mean_ious.values()))

    print("\n" + "="*40)
    print("Test Results:")
    print("="*40)
    print(f"Overall Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print("\nClass-wise IoU:")
    print(f"  Class 0 (No Damage / Background): {mean_ious[0]:.4f}")
    print(f"  Class 1 (Minor Damage):         {mean_ious[1]:.4f}")
    print(f"  Class 2 (Major Damage):         {mean_ious[2]:.4f}")
    print(f"  Class 3 (Destroyed):            {mean_ious[3]:.4f}")
    print("="*40)
    print(f"Sample Visualizations saved to 'test_outputs/' directory.")

    # Log metrics to W&B
    wandb.log({
        "test_pixel_accuracy": pixel_acc,
        "test_mIoU": mIoU,
        "test_iou_class_0_nodamage": mean_ious[0],
        "test_iou_class_1_minor": mean_ious[1],
        "test_iou_class_2_major": mean_ious[2],
        "test_iou_class_3_destroyed": mean_ious[3],
    })

    # Plot and save confusion matrix (Normalized by row/True Label)
    cm_cpu = conf_matrix.cpu().numpy()
    cm_normalized = cm_cpu.astype('float') / cm_cpu.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    class_names = ['None/BG', 'Minor', 'Major', 'Destroyed']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = "test_outputs/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()

    wandb.log({"Test Confusion Matrix": wandb.Image(cm_path, caption="Normalized Confusion Matrix")})
    print(f"Confusion Matrix saved to {cm_path}")
    
    wandb.finish()

if __name__ == "__main__":
    test_siamese("checkpoints_siamese/siamese_epoch_V5_200.pth")
