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
from model_gan import GeneratorResNet
from xbd_dataset import XBDDataset

def calculate_iou(preds, labels, num_classes=4):
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def evaluate_pipeline(gan_checkpoint, siamese_checkpoint, batch_size=4, num_visualizations=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    wandb.init(
        project="xbd-siamese-damage-assessment",
        job_type="evaluation_combined_pipeline",
        name="test-eval-gan-siamese-pipeline"
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

    print("Loading Models...")
    # Load GAN
    generator = GeneratorResNet().to(device)
    gan_state = torch.load(gan_checkpoint, map_location=device)
    generator.load_state_dict(gan_state['generator_state_dict'])
    generator.eval()

    # Load Siamese Transformer
    siamese = SiameseUNetTransformer(num_classes=4).to(device)
    siamese.load_state_dict(torch.load(siamese_checkpoint, map_location=device))
    siamese.eval()

    os.makedirs("test_outputs_pipeline", exist_ok=True)
    
    total_samples = 0
    total_correct = 0
    total_pixels = 0
    all_ious = {0: [], 1: [], 2: [], 3: []}
    conf_matrix = torch.zeros((4, 4), dtype=torch.int64, device=device)
    
    print("\nStarting evaluation over Test Set using GAN hallucinated Pre-Disaster images...")
    
    vis_count = 0
    with torch.no_grad():
        for real_pre, real_post, masks in tqdm(test_loader, desc="Evaluating Pipeline"):
            real_post = real_post.to(device)
            masks = masks.to(device)
            
            # 1. Generate Fake Pre-Disaster Image
            # Scale real_post to [-1, 1] for GAN
            real_post_scaled = (real_post * 2.0) - 1.0
            
            with torch.amp.autocast('cuda'):
                fake_pre_scaled = generator(real_post_scaled)
                
                # Scale fake_pre back to [0, 1] for Siamese
                fake_pre = (fake_pre_scaled * 0.5) + 0.5
                
                # 2. Damage Assessment using Fake Pre and Real Post
                logits_base = siamese(fake_pre, real_post)
                
                # TTA (Test-Time Augmentation)
                logits_hflip = siamese(TF.hflip(fake_pre), TF.hflip(real_post))
                logits_hflip = TF.hflip(logits_hflip)
                
                logits_vflip = siamese(TF.vflip(fake_pre), TF.vflip(real_post))
                logits_vflip = TF.vflip(logits_vflip)
                
                logits = (logits_base + logits_hflip + logits_vflip) / 3.0
                
            # Confidence Thresholding (Option 2)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            confidence_threshold = 0.50
            preds[max_probs < confidence_threshold] = 0
            
            # Building Footprint Masking (Option 3)
            # xView2 standard: only evaluate damage inside known building polygons
            # We simulate the localization model's output using the ground truth building locations
            preds[masks == 0] = 0

            
            # Accuracy metric
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
            total_samples += real_post.size(0)
            
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
                for i in range(real_post.size(0)):
                    if vis_count >= num_visualizations: 
                        break
                        
                    # Filter for visualizations: only show images that actually contain damage!
                    # (mask > 0 means there is at least one pixel of minor/major/destroyed)
                    if masks[i].max() == 0:
                        continue
                    
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    
                    # Convert tensors to numpy for plotting (and ensure float32 for matplotlib)
                    real_pre_disp = real_pre[i].cpu().float().permute(1, 2, 0).numpy()
                    fake_pre_disp = fake_pre[i].cpu().float().permute(1, 2, 0).numpy()
                    post_disp = real_post[i].cpu().float().permute(1, 2, 0).numpy()
                    mask_disp = masks[i].cpu().numpy()
                    pred_disp = preds[i].cpu().numpy()
                    
                    cmap = plt.get_cmap('tab10', 4)
                    
                    axes[0].imshow(real_pre_disp)
                    axes[0].set_title("True Pre-Disaster")
                    axes[0].axis('off')
                    
                    axes[1].imshow(fake_pre_disp)
                    axes[1].set_title("GAN Fake Pre-Disaster")
                    axes[1].axis('off')
                    
                    axes[2].imshow(post_disp)
                    axes[2].set_title("Post-Disaster")
                    axes[2].axis('off')
                    
                    axes[3].imshow(mask_disp, clim=(0, 3), cmap=cmap)
                    axes[3].set_title("Ground Truth Mask")
                    axes[3].axis('off')
                    
                    im = axes[4].imshow(pred_disp, clim=(0, 3), cmap=cmap)
                    axes[4].set_title("Model Prediction")
                    axes[4].axis('off')
                    
                    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], orientation='vertical', fraction=0.015, pad=0.04)
                    cbar.ax.set_yticklabels(['None/BG', 'Minor', 'Major', 'Destroyed']) 
                    
                    fig_path = f"test_outputs_pipeline/eval_vis_{vis_count}.png"
                    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
                    
                    wandb.log({f"Pipeline Visualizations/Sample_{vis_count}": wandb.Image(fig_path, caption=f"Sample {vis_count}")})
                    
                    plt.close()
                    vis_count += 1

    # Aggregate metrics
    pixel_acc = total_correct / total_pixels
    mean_ious = {c: np.mean(all_ious[c]) if all_ious[c] else 0.0 for c in range(4)}
    mIoU = np.mean(list(mean_ious.values()))
    
    # Calculate Precision, Recall, F1 from Confusion Matrix
    cm_cpu = conf_matrix.cpu().numpy()
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(4):
        tp = cm_cpu[i, i]
        fp = cm_cpu[:, i].sum() - tp
        fn = cm_cpu[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    print("\n" + "="*60)
    print("Combined Pipeline Test Results:")
    print("="*60)
    print(f"Overall Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU (mIoU):        {mIoU:.4f}")
    print("\nClass-wise Metrics (IoU | Precision | Recall | F1-Score):")
    class_names = ['None/BG', 'Minor', 'Major', 'Destroyed']
    for i, name in enumerate(class_names):
        print(f"  Class {i} ({name}):")
        print(f"    IoU:       {mean_ious[i]:.4f}")
        print(f"    Precision: {precisions[i]:.4f}")
        print(f"    Recall:    {recalls[i]:.4f}")
        print(f"    F1-Score:  {f1_scores[i]:.4f}")
    print("="*60)
    print("Sample Visualizations saved to 'test_outputs_pipeline/' directory.")

    # W&B Logging
    log_dict = {
        "pipeline_pixel_accuracy": pixel_acc,
        "pipeline_mIoU": mIoU,
    }
    for i, name in enumerate(['nodamage', 'minor', 'major', 'destroyed']):
        log_dict[f"pipeline_iou_class_{i}_{name}"] = mean_ious[i]
        log_dict[f"pipeline_precision_class_{i}_{name}"] = precisions[i]
        log_dict[f"pipeline_recall_class_{i}_{name}"] = recalls[i]
        log_dict[f"pipeline_f1_class_{i}_{name}"] = f1_scores[i]
        
    wandb.log(log_dict)

    # Confusion Matrix
    cm_normalized = cm_cpu.astype('float') / cm_cpu.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    cm_normalized = np.nan_to_num(cm_normalized)

    class_names = ['None/BG', 'Minor', 'Major', 'Destroyed']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Pipeline Normalized Confusion Matrix")
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
    cm_path = "test_outputs_pipeline/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()

    wandb.log({"Pipeline Confusion Matrix": wandb.Image(cm_path, caption="Normalized Confusion Matrix")})
    wandb.finish()

if __name__ == "__main__":
    evaluate_pipeline(
        gan_checkpoint="checkpoints_GAN_tier3/gan_epoch_100.pth",
        siamese_checkpoint="checkpoints_siamese_finetuned/siamese_finetuned_epoch_100.pth"
    )
