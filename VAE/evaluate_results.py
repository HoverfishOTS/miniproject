import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dataset import LADIDataset
from VAE.model_vae import VAE
from VAE.train_classifier import LatentClassifier

def evaluate(vae_weights="checkpoints/vae_epoch_50.pth", classifier_weights="checkpoints/classifier_best.pth", latent_dim=256, num_classes=12):
    # Initialize W&B purely for logging final results
    wandb.init(project="miniproject-ladi-evaluation", name="test-set-evaluation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load Stage 1 Model (VAE)
    vae = VAE(latent_dim=latent_dim).to(device)
    if os.path.exists(vae_weights):
        vae.load_state_dict(torch.load(vae_weights, map_location=device))
        print("Loaded VAE weights.")
    else:
        print(f"WARNING: Could not find VAE weights at {vae_weights}")
    vae.eval()

    # Load Stage 2 Model (Classifier)
    classifier = LatentClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)
    if os.path.exists(classifier_weights):
        classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
        print("Loaded Classifier weights.")
    else:
        print(f"WARNING: Could not find Classifier weights at {classifier_weights}")
    classifier.eval()

    # Load TEST dataset (2023 disaster events - out of distribution)
    # This evaluates how well the model generalizes to new events.
    print("Loading Test Dataset...")
    test_dataset = LADIDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            
            # Step 1: LVM Generates Latent Representation
            mu, logvar = vae.encoder(data)
            
            # Step 2: Discriminative Head Predicts Damage
            logits = classifier(mu)
            
            # Convert logits to binary predictions (Threshold 0.5)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # For the Word Document: Save a grid of original vs. reconstructed images from the first batch
            if batch_idx == 0:
                print("Generating Reconstruction Grid for Word Document...")
                # Reparameterize and Decode to get Full Reconstructions
                z = vae.reparameterize(mu, logvar)
                recon = vae.decoder(z)
                
                # Pick 4 images to showcase
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                plt.suptitle("LVM Generative Reconstructions (Top: Original, Bottom: VAE)", fontsize=16)
                
                for i in range(4):
                    # Original
                    orig_img = data[i].cpu().permute(1, 2, 0).numpy()
                    axes[0, i].imshow(orig_img)
                    axes[0, i].axis('off')
                    
                    # Reconstruction
                    recon_img = recon[i].cpu().permute(1, 2, 0).numpy()
                    axes[1, i].imshow(recon_img)
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                plt.savefig('reconstruction_grid.png', dpi=300)
                print("Saved 'reconstruction_grid.png' to disk.")
                
                wandb.log({"Reconstruction Grid": wandb.Image('reconstruction_grid.png')})

    # Calculate Global Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n--- TEST SET EVALUATION REPORT ---")
    print(f"Overall Accuracy: {test_accuracy*100:.2f}%")
    print(f"Overall F1-Score (Macro): {test_f1:.4f}")
    
    wandb.log({
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1
    })
    
    # Detailed per-class report
    print("\nDetailed Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=test_dataset.label_cols, zero_division=0)
    print(report)
    
    # Save the text report to disk so it's easy to paste into the Word doc
    with open("classification_report.txt", "w") as f:
        f.write(f"TEST SET EVALUATION REPORT\n")
        f.write(f"Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"F1-Score: {test_f1:.4f}\n\n")
        f.write(report)
        
    print("Saved 'classification_report.txt' to disk.")
    wandb.finish()

if __name__ == "__main__":
    evaluate()
