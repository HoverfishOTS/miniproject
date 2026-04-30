import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random

class XBDDataset(Dataset):
    """
    A Dataloader for the paired xBD Dataset.
    Loads Pre-disaster imagery, Post-disaster imagery, and generates 
    a 4-class damage segmentation mask on the fly from the JSON polygons.
    """
    def __init__(self, root_dir="xbd_data", split='train', img_size=256):
        self.split = split
        self.img_size = img_size
        self.split_dir = os.path.join(root_dir, split)
        
        # We find all POST disaster images, because the damage labels are tied directly to the POST image files
        self.image_paths = sorted(glob.glob(os.path.join(self.split_dir, "images", "*_post_disaster.png")))
        self.length = len(self.image_paths)
        print(f"[*] Loaded XBD {split} dataset mapping ({self.length} images).")

        # Official xView2 challenge damage metrics
        self.damage_dict = {
            "un-classified": 0,
            "no-damage": 0, 
            "minor-damage": 1, 
            "major-damage": 2, 
            "destroyed": 3
        }

    def __len__(self):
        return self.length

    def get_sample_weights(self):
        """
        Computes sampling weights for each image based on damage severity.
        Used by WeightedRandomSampler to balance the dataset.
        """
        weights_cache_path = os.path.join(self.split_dir, "sample_weights.pt")
        if os.path.exists(weights_cache_path):
            print("[*] Loading cached sample weights...")
            return torch.load(weights_cache_path, weights_only=True)
            
        print("[*] Calculating sample weights based on damage severity (this might take a minute)...")
        weights = []
        for img_path in self.image_paths:
            label_path = img_path.replace("images", "labels").replace(".png", ".json")
            weight = 1.0 # Base weight for images with no damage or un-classified
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    try:
                        label_data = json.load(f)
                        if "features" in label_data and "xy" in label_data["features"]:
                            for feature in label_data["features"]["xy"]:
                                props = feature.get("properties", {})
                                subtype = props.get("subtype", "no-damage")
                                class_id = self.damage_dict.get(subtype, 0)
                                if class_id == 3: # Destroyed
                                    weight = 50.0
                                    break # Maximum weight achieved
                                elif class_id == 2: # Major damage
                                    weight = max(weight, 20.0)
                                elif class_id == 1: # Minor damage
                                    weight = max(weight, 5.0)
                    except json.JSONDecodeError:
                        pass
            weights.append(weight)
            
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        torch.save(weights_tensor, weights_cache_path)
        return weights_tensor

    def parse_wkt_polygon(self, wkt_str):
        # Extract string format: "POLYGON ((x y, x y, ...))" -> [(x,y), (x,y)]
        wkt_str = wkt_str.replace("POLYGON", "").replace("(", "").replace(")", "").strip()
        coords_str = wkt_str.split(",")
        coords = []
        for c in coords_str:
            x, y = map(float, c.strip().split())
            coords.append((x, y))
        return coords

    def __getitem__(self, idx):
        post_img_path = self.image_paths[idx]
        
        # Create a cache directory that encapsulates the resolution
        cache_dir = os.path.join(self.split_dir, f"cache_tensors_{self.img_size}")
        cache_path = os.path.join(cache_dir, os.path.basename(post_img_path).replace(".png", ".pt"))
        
        # 1. Load Pre-Computed Resized Tensors if they exist
        # This completely bypasses massive 1024x1024 PNG decoding and Bilinear resizing!
        if os.path.exists(cache_path):
            pre_tensor, post_tensor, mask_tensor = torch.load(cache_path, weights_only=True)
        else:
            # 2. Base Paths & Image Loading (Initial Cache Miss)
            pre_img_path = post_img_path.replace("_post_disaster.png", "_pre_disaster.png")
            post_label_path = post_img_path.replace("images", "labels").replace(".png", ".json")

            pre_img = Image.open(pre_img_path).convert("RGB")
            post_img = Image.open(post_img_path).convert("RGB")
            w, h = post_img.size

            # 3. Parse JSON Annotations
            mask_img = Image.new('L', (w, h), color=0)
            draw = ImageDraw.Draw(mask_img)

            if os.path.exists(post_label_path):
                with open(post_label_path, 'r') as f:
                    label_data = json.load(f)

                if "features" in label_data and "xy" in label_data["features"]:
                    for feature in label_data["features"]["xy"]:
                        props = feature.get("properties", {})
                        subtype = props.get("subtype", "no-damage")
                        class_id = self.damage_dict.get(subtype, 0)
                        
                        wkt_poly = feature.get("wkt", "")
                        if "POLYGON" in wkt_poly:
                            poly_coords = self.parse_wkt_polygon(wkt_poly)
                            draw.polygon(poly_coords, outline=class_id, fill=class_id)

            # 4. Resize Operations (HUGE CPU Bottleneck)
            pre_img = TF.resize(pre_img, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
            post_img = TF.resize(post_img, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
            mask_img = TF.resize(mask_img, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST)

            # 5. Convert to PyTorch Tensors
            pre_tensor = TF.to_tensor(pre_img)
            post_tensor = TF.to_tensor(post_img)
            
            mask_np = np.array(mask_img, dtype=np.int64)
            mask_tensor = torch.from_numpy(mask_np)
            mask_tensor = torch.clamp(mask_tensor, min=0, max=3)
            
            # 6. Save Tensors exactly as they are to NVMe/SSD for instant access natively in torch
            os.makedirs(cache_dir, exist_ok=True)
            torch.save((pre_tensor, post_tensor, mask_tensor), cache_path)

        # 6. Paired Data Augmentations (Train only)
        if self.split == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                pre_tensor = TF.hflip(pre_tensor)
                post_tensor = TF.hflip(post_tensor)
                mask_tensor = TF.hflip(mask_tensor)
                
            # Random Vertical Flip
            if random.random() > 0.5:
                pre_tensor = TF.vflip(pre_tensor)
                post_tensor = TF.vflip(post_tensor)
                mask_tensor = TF.vflip(mask_tensor)
                
            # Random 90-degree Rotations
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                pre_tensor = torch.rot90(pre_tensor, k_rot, dims=[1, 2])
                post_tensor = torch.rot90(post_tensor, k_rot, dims=[1, 2])
                # Mask tensor is 2D (H, W), so we rotate across dims 0 and 1!
                mask_tensor = torch.rot90(mask_tensor, k_rot, dims=[0, 1])
                
            # Independent Slight Color Jitter 
            # (Crucial: The pre-disaster and post-disaster images are often taken weeks apart 
            # with completely different lighting, seasons, or camera sensors. Independent jitter
            # prevents the Siamese network from just comparing basic pixel brightness.)
            if random.random() > 0.5:
                pre_tensor = TF.adjust_brightness(pre_tensor, random.uniform(0.8, 1.2))
                pre_tensor = TF.adjust_contrast(pre_tensor, random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                post_tensor = TF.adjust_brightness(post_tensor, random.uniform(0.8, 1.2))
                post_tensor = TF.adjust_contrast(post_tensor, random.uniform(0.8, 1.2))

        return pre_tensor, post_tensor, mask_tensor
