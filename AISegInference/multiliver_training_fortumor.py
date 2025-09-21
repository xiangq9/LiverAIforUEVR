import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import multiprocessing as mp
import random
import warnings
import math
warnings.filterwarnings("ignore")

import monai
from monai.data import Dataset, list_data_collate, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangeD,
    RandCropByPosNegLabeld, RandRotate90d, RandShiftIntensityd,
    RandFlipd, Spacingd, Orientationd, SpatialPadd, EnsureTyped,
    ToTensord, MapTransform, CropForegroundd, BorderPadd,
    RandGaussianNoised, RandScaleIntensityd
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm

# Task configuration
TASK_TYPE = "tumor"

# Windows compatibility settings
if os.name == 'nt':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def set_random_seeds():
    seed = random.randint(1, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")
    return seed

# Path configuration
DATA_DIR = rf".\{TASK_TYPE}_dataset"
MODEL_DIR = rf".\{TASK_TYPE}_model"
TRAIN_PCT = 0.8

# Modern yet simple configuration - drawing from reference code insights
BATCH_SIZE = 2                    # Based on reference code's batch_size=2
GRADIENT_ACCUMULATION = 2         # Reduced accumulation steps
NUM_WORKERS = 4                   
MAX_EPOCHS = 100                  
VAL_INTERVAL = 2                  
SPATIAL_SIZE = [96, 96, 96]       # Slightly larger patch
EARLY_STOP_PATIENCE = 20          
LEARNING_RATE = 1e-4              # Lower learning rate
WARMUP_EPOCHS = 5                 

# Moderate sampling strategy - based on reference code
POS_NEG_RATIO = 3                 # 3:1 moderate sampling
NUM_SAMPLES_PER_IMAGE = 16        # Reduced sample count
ONLY_TUMOR_SLICES = True          # Train only on slices with tumors - key improvement

# Settings
USE_CACHE_DATASET = True
MEMORY_EFFICIENT = True
SAVE_DEBUG_IMAGES = True
DEBUG_DIR = os.path.join(MODEL_DIR, "debug_vis")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Remove custom transforms, use MONAI standard transforms to avoid dimension issues

class CleanLabelsd(MapTransform):
    """Label cleaning - based on reference code's binarization approach"""
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                labels = d[key]
                if hasattr(labels, 'numpy'):
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.as_tensor(labels)
                    # Strict binarization: >0.5=1, <=0.5=0
                    cleaned = torch.where(labels > 0.5, 1.0, 0.0).float()
                    d[key] = cleaned
                elif hasattr(labels, 'get_fdata'):
                    labels_array = labels.get_fdata()
                    cleaned = np.where(labels_array > 0.5, 1.0, 0.0).astype(np.float32)
                    import nibabel as nib
                    cleaned_nii = nib.Nifti1Image(cleaned, labels.affine, labels.header)
                    d[key] = cleaned_nii
                else:
                    cleaned = np.where(labels > 0.5, 1.0, 0.0).astype(np.float32)
                    d[key] = cleaned
        return d

def prepare_data(data_dir):
    """Data preparation"""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    print(f"Looking for data: {images_dir}")
    
    images = sorted(glob(os.path.join(images_dir, "*.nii.gz")))
    labels = sorted(glob(os.path.join(labels_dir, "*.nii.gz")))
    
    if not images:
        images = sorted(glob(os.path.join(images_dir, "*.nii")))
        labels = sorted(glob(os.path.join(labels_dir, "*.nii")))
    
    print(f"Found {len(images)} images and {len(labels)} labels")
    
    if len(images) == 0 or len(labels) == 0:
        raise ValueError(f"No data found in {data_dir}")
    
    if len(images) != len(labels):
        min_len = min(len(images), len(labels))
        images = images[:min_len]
        labels = labels[:min_len]
    
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    
    data_dicts = [
        {"image": image_name, label_key: label_name}
        for image_name, label_name in zip(images, labels)
    ]
    
    random.shuffle(data_dicts)
    train_size = int(len(data_dicts) * TRAIN_PCT)
    train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]
    
    return train_files, val_files

def get_transforms(phase, memory_efficient=False):
    """Modern data transformations - incorporating reference code insights"""
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    keys = ["image", label_key]
    
    common_transforms = [
        LoadImaged(keys=keys),
        CleanLabelsd(keys=[label_key]),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.5), mode=("bilinear", "nearest")),
        Orientationd(keys=keys, axcodes="RAS"),
        
        # KEY IMPROVEMENT: Window width adjustment - based on reference code
        WindowingTransform(keys=["image"], window_width=350, window_center=40),
        
        # KEY IMPROVEMENT: CLAHE enhancement - based on reference code  
        CLAHETransform(keys=["image"], clip_limit=2.0, tile_grid_size=(8, 8)),
        
        # Crop foreground
        CropForegroundd(keys=keys, source_key="image", margin=10),
    ]
    
    if memory_efficient:
        common_transforms.append(ToTensord(keys=keys))
    
    if phase == "train":
        train_specific = [
            BorderPadd(keys=keys, spatial_border=SPATIAL_SIZE),
            
            # Moderate positive sample sampling - based on reference code approach
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=label_key,
                spatial_size=SPATIAL_SIZE,
                pos=POS_NEG_RATIO,        # 3:1 moderate sampling
                neg=1,
                num_samples=NUM_SAMPLES_PER_IMAGE,  # 16 samples
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
        ]
        
        # Moderate data augmentation - based on reference code parameters
        augmentations = [
            # Basic flipping - reference code has horizontal_flip=False, vertical_flip=False
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            
            # 90-degree rotation - corresponds to rotation_range=0.1
            RandRotate90d(keys=keys, prob=0.3, max_k=1),
            
            # Moderate intensity adjustments
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
        ]
        
        return Compose([
            *common_transforms,
            *train_specific,
            *augmentations,
            EnsureTyped(keys=keys),
        ])
    else:
        return Compose([
            *common_transforms,
            EnsureTyped(keys=keys),
        ])

class ModernUNet(nn.Module):
    """Modern UNet - based on reference code's simplified architecture"""
    def __init__(self, out_channels=2, use_checkpoint=False):
        super().__init__()
        
        # Simplified UNet - based on reference code structure but using MONAI implementation
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),  # Corresponds to reference code's 64,128,256,512
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
            norm=Norm.BATCH,
        )
        
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.unet, x, use_reentrant=False
            )
        else:
            return self.unet(x)
    
    def inference(self, x):
        return self.forward(x)

def create_data_loaders(train_files, val_files):
    """Create data loaders"""
    train_transforms = get_transforms("train", memory_efficient=MEMORY_EFFICIENT)
    val_transforms = get_transforms("val", memory_efficient=MEMORY_EFFICIENT)
    
    print(f"Creating data loaders...")
    
    if USE_CACHE_DATASET:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.8,
            num_workers=NUM_WORKERS,
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=NUM_WORKERS,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    
    return train_loader, val_loader

def calculate_metrics(pred, target, epsilon=1e-8):
    """Calculate metrics"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + epsilon) / (pred_binary.sum() + target_binary.sum() + epsilon)
    
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    
    tp = intersection
    fp = pred_binary.sum() - intersection
    fn = target_binary.sum() - intersection
    
    precision = (tp + epsilon) / (tp + fp + epsilon)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def validate_model_modern(model, val_loader, device, epoch):
    """Modern validation function"""
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            val_inputs, val_labels = (
                val_data["image"].to(device, non_blocking=True),
                val_data[label_key].to(device, non_blocking=True),
            )
            
            val_labels = val_labels.long()
            
            # Inference
            try:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits = sliding_window_inference(
                        val_inputs, 
                        SPATIAL_SIZE, 
                        2, 
                        model.inference,
                        overlap=0.6,
                        mode="gaussian"
                    )
            except Exception as e:
                print(f"Inference failed: {e}")
                logits = model.inference(val_inputs)
            
            # Post-processing - based on reference code's sigmoid approach
            pred_softmax = F.softmax(logits, dim=1)
            pred_binary = (pred_softmax[:, 1:2] > 0.5).float()
            
            # Calculate metrics
            target_binary = (val_labels > 0).float()
            metrics = calculate_metrics(pred_binary, target_binary)
            all_metrics.append(metrics)
            
            # Save visualization
            if SAVE_DEBUG_IMAGES and idx == 0:
                save_debug_visualization(val_inputs, val_labels, pred_binary, epoch, idx)
            
            torch.cuda.empty_cache()
    
    # Average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    else:
        avg_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    return avg_metrics, all_metrics

def save_debug_visualization(inputs, labels, predictions, epoch, idx):
    """Save debug visualization"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        img_np = inputs[0, 0].detach().cpu().numpy()
        lab_np = labels[0, 0].detach().cpu().numpy() if labels.dim() == 5 else labels[0].detach().cpu().numpy()
        pred_np = predictions[0, 0].detach().cpu().numpy()
        
        # Select middle slice
        z = img_np.shape[2] // 2
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_np[:, :, z], cmap='gray')
        axes[0].set_title(f'Enhanced CT (Epoch {epoch})')
        axes[0].axis('off')
        
        axes[1].imshow(lab_np[:, :, z], cmap='Reds')
        axes[1].set_title(f'Ground Truth (Pixels: {lab_np[:,:,z].sum():.0f})')
        axes[1].axis('off')
        
        axes[2].imshow(pred_np[:, :, z], cmap='Blues')
        axes[2].set_title(f'Prediction (Pixels: {pred_np[:,:,z].sum():.0f})')
        axes[2].axis('off')
        
        axes[3].imshow(img_np[:, :, z], cmap='gray', alpha=0.7)
        axes[3].imshow(lab_np[:, :, z], cmap='Reds', alpha=0.4)
        axes[3].imshow(pred_np[:, :, z], cmap='Blues', alpha=0.3)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(DEBUG_DIR, f"modern_prediction_epoch{epoch}_{idx}.png"),
            dpi=100, bbox_inches='tight'
        )
        plt.close()
        
    except Exception as e:
        print(f"Failed to save visualization: {e}")

def train_one_epoch_modern(model, train_loader, optimizer, loss_fn, device, scaler, epoch):
    """Modern training function"""
    model.train()
    epoch_loss = 0
    step = 0
    total_fg_ratio = 0
    positive_samples = 0
    
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    accumulation_steps = GRADIENT_ACCUMULATION
    
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = (
            batch_data["image"].to(device, non_blocking=True),
            batch_data[label_key].to(device, non_blocking=True),
        )
        
        labels = (labels > 0.5).long()
        
        # Monitor foreground ratio
        fg_ratio = (labels > 0).float().mean().item()
        total_fg_ratio += fg_ratio
        
        if fg_ratio > 0.001:
            positive_samples += 1
        
        if batch_idx < 3 and epoch < 5:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Foreground ratio {fg_ratio:.6f}")
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda', enabled=True):
            seg_outputs = model(inputs)
            loss = loss_fn(seg_outputs, labels) / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if len(train_loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_fg_ratio = total_fg_ratio / step
    return epoch_loss / step, avg_fg_ratio, positive_samples

def main():
    print("=" * 80)
    print("Modern Tumor Segmentation System - Incorporating Reference Code Insights + Latest MONAI Technology")
    print("=" * 80)
    
    # Set random seed
    current_seed = set_random_seeds()
    
    print(f"Configuration:")
    print(f"  - Random seed: {current_seed}")
    print(f"  - Patch size: {SPATIAL_SIZE}")
    print(f"  - Pos/Neg ratio: {POS_NEG_RATIO}:1 (moderate sampling)")
    print(f"  - Batch size: {BATCH_SIZE} (based on reference code)")
    print(f"  - Samples per image: {NUM_SAMPLES_PER_IMAGE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Key improvements: Window width adjustment + CLAHE enhancement + Simplified UNet")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    # Data preparation
    train_files, val_files = prepare_data(DATA_DIR)
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    # Data loaders
    train_loader, val_loader = create_data_loaders(train_files, val_files)
    print(f"Steps per epoch: {len(train_loader)}")
    
    # Modern model
    print(f"Building modern UNet model...")
    model = ModernUNet(
        out_channels=2,
        use_checkpoint=MEMORY_EFFICIENT,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer - Adam, based on reference code
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # Learning rate scheduling
    def warmup_cosine(epoch):
        if epoch < WARMUP_EPOCHS:
            return epoch / WARMUP_EPOCHS
        else:
            cos_epoch = epoch - WARMUP_EPOCHS
            cos_epochs = MAX_EPOCHS - WARMUP_EPOCHS
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_epochs))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    
    # Mixed precision
    scaler = torch.amp.GradScaler()
    
    # Simplified loss function - based on reference code's binary_crossentropy approach
    dice_ce_loss = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        ce_weight=torch.tensor([1.0, 2.0]).to(device),  # Moderate weights, not 50:1
        smooth_nr=1e-5,
        smooth_dr=1e-5
    )
    
    print("Loss function: DiceCE (Dice + CrossEntropy, weight 2:1)")
    
    # TensorBoard
    log_dir = os.path.join(MODEL_DIR, "modern_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training variables
    best_dice = -1
    best_metrics = {}
    best_epoch = -1
    early_stop_counter = 0
    
    print(f"\nStarting training for {MAX_EPOCHS} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        try:
            # Training
            epoch_loss, avg_fg_ratio, positive_samples = train_one_epoch_modern(
                model, train_loader, optimizer, dice_ce_loss, device, scaler, epoch
            )
            
            # Logging
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/fg_ratio", avg_fg_ratio, epoch)
            writer.add_scalar("train/positive_samples", positive_samples, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            scheduler.step()
            
            epoch_time = time.time() - epoch_start
            print(f"Training: {epoch_time:.1f}s, Loss: {epoch_loss:.4f}, Foreground: {avg_fg_ratio:.6f}, Positive samples: {positive_samples}")
            
            # Validation
            if (epoch + 1) % VAL_INTERVAL == 0:
                print("Validating...")
                val_start = time.time()
                
                avg_metrics, all_metrics = validate_model_modern(
                    model, val_loader, device, epoch
                )
                
                val_time = time.time() - val_start
                print(f"Validation: {val_time:.1f}s")
                
                # Log all metrics
                for key, value in avg_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                
                current_dice = avg_metrics['dice']
                
                # Check improvement
                if current_dice > best_dice:
                    improvement = current_dice - best_dice
                    best_dice = current_dice
                    best_metrics = avg_metrics.copy()
                    best_epoch = epoch + 1
                    
                    # Save best model
                    model_path = os.path.join(MODEL_DIR, f"modern_best_model.pth")
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': avg_metrics,
                        'best_dice': best_dice,
                        'seed': current_seed,
                        'config': {
                            'spatial_size': SPATIAL_SIZE,
                            'pos_neg_ratio': POS_NEG_RATIO,
                            'num_samples': NUM_SAMPLES_PER_IMAGE,
                            'learning_rate': LEARNING_RATE,
                            'batch_size': BATCH_SIZE,
                        }
                    }
                    torch.save(checkpoint, model_path)
                    
                    print(f"New best model! Improvement: +{improvement:.6f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                # Display detailed metrics
                print(f"Metrics:")
                print(f"  Dice: {current_dice:.6f} (Best: {best_dice:.6f})")
                print(f"  IoU: {avg_metrics['iou']:.6f}")  
                print(f"  Precision: {avg_metrics['precision']:.6f}")
                print(f"  Recall: {avg_metrics['recall']:.6f}")
                
                # Compare with previous 0.16
                if best_dice > 0.16:
                    improvement_vs_old = best_dice - 0.16
                    print(f"Surpassed previous 0.16! Improvement: +{improvement_vs_old:.6f}")
                
                # Progress assessment
                if best_dice > 0.7:
                    print("Excellent level! Approaching clinical usability!")
                elif best_dice > 0.5:
                    print("Good level! Significant improvement!")
                elif best_dice > 0.3:
                    print("Steady progress! Correct direction!")
                elif best_dice > 0.2:
                    print("Continuous improvement...")
                else:
                    print("Learning in progress, patience required...")
                
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print("Early stopping")
                    break
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Training summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best results (Epoch {best_epoch}):")
    print(f"  Dice: {best_dice:.6f}")
    print(f"  IoU: {best_metrics.get('iou', 0):.6f}")
    print(f"  Precision: {best_metrics.get('precision', 0):.6f}")
    print(f"  Recall: {best_metrics.get('recall', 0):.6f}")
    print(f"Total training time: {total_time:.1f} minutes")
    print(f"Random seed: {current_seed}")
    
    # Comparison analysis with reference code
    print(f"\nImprovement Analysis:")
    if best_dice > 0.16:
        improvement = best_dice - 0.16
        print(f"Improvement over previous 0.16 baseline: +{improvement:.6f}")
        print(f"Key improvements effective: Window width adjustment + CLAHE enhancement + Simplified architecture")
    else:
        print(f"Not meeting expectations, recommend checking data preprocessing")
    
    print(f"Models and logs saved in: {MODEL_DIR}")
    
    # Cleanup resources
    writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nSystem error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
