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
warnings.filterwarnings("ignore")

import monai
from monai.data import Dataset, list_data_collate, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRangeD,
    RandCropByPosNegLabeld, RandRotate90d, RandShiftIntensityd,
    RandFlipd, Spacingd, Orientationd, SpatialPadd, EnsureTyped,
    ToTensord, MapTransform, Activations, AsDiscrete, RandGaussianNoised,
    RandAdjustContrastd, RandScaleIntensityd, RandRotated, CropForegroundd,
    RandSpatialCropSamplesd, BorderPadd
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, TverskyLoss, DiceLoss, FocalLoss, GeneralizedDiceLoss
from monai.networks.nets import SegResNet, UNet, UNETR, SwinUNETR
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
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Set random seeds
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

# 🎯 Aggressive optimization configuration - specifically for small tumors
BATCH_SIZE = 1                    
GRADIENT_ACCUMULATION = 6         # Further increase gradient accumulation
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION
NUM_WORKERS = 4                   
MAX_EPOCHS = 200                  # Increase training epochs
VAL_INTERVAL = 2                  
SPATIAL_SIZE = [80, 80, 80]       # Slightly reduce patch to ensure complete tumor inclusion
EARLY_STOP_PATIENCE = 30          
LEARNING_RATE = 5e-4              # Increase learning rate
WARMUP_EPOCHS = 15                

# 🔥 Aggressive sampling strategy - focus on positive samples
POS_NEG_RATIO = 15                # 15:1 aggressive positive sample sampling
NUM_SAMPLES_PER_IMAGE = 30        # Significantly increase sampling
USE_MULTI_SCALE = False           # Disable multi-scale first to avoid errors

# Settings
USE_CACHE_DATASET = True
MEMORY_EFFICIENT = True
SAVE_DEBUG_IMAGES = True
DEBUG_DIR = os.path.join(MODEL_DIR, "debug_vis")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

class CleanLabelsd(MapTransform):
    """Stricter label cleaning"""
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
                    # Very strict binarization
                    cleaned = torch.where(labels > 0.01, 1.0, 0.0).float()
                    d[key] = cleaned
                elif hasattr(labels, 'get_fdata'):
                    labels_array = labels.get_fdata()
                    cleaned = np.where(labels_array > 0.01, 1.0, 0.0).astype(np.float32)
                    import nibabel as nib
                    cleaned_nii = nib.Nifti1Image(cleaned, labels.affine, labels.header)
                    d[key] = cleaned_nii
                else:
                    cleaned = np.where(labels > 0.01, 1.0, 0.0).astype(np.float32)
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
    """Aggressively optimized data transforms - specifically for small tumors"""
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    keys = ["image", label_key]
    
    common_transforms = [
        LoadImaged(keys=keys),
        CleanLabelsd(keys=[label_key]),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.2), mode=("bilinear", "nearest")),  # More refined
        Orientationd(keys=keys, axcodes="RAS"),
        # Wider window range
        ScaleIntensityRangeD(
            keys=["image"],
            a_min=-250, a_max=400,
            b_min=0.0, b_max=1.0, clip=True
        ),
        # Crop foreground but preserve more edges
        CropForegroundd(keys=keys, source_key="image", margin=20),
    ]
    
    if memory_efficient:
        common_transforms.append(ToTensord(keys=keys))
    
    if phase == "train":
        train_specific = [
            # Ensure patch size
            BorderPadd(keys=keys, spatial_border=SPATIAL_SIZE),
            
            # 🔥 Ultra-aggressive positive sample sampling
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=label_key,
                spatial_size=SPATIAL_SIZE,
                pos=POS_NEG_RATIO,        # 15:1 extreme positive sample sampling
                neg=1,
                num_samples=NUM_SAMPLES_PER_IMAGE,  # 30 samples
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
        ]
        
        # Extremely gentle augmentation - avoid destroying small tumors
        augmentations = [
            # Keep only the safest augmentations
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            
            # Very small rotation
            RandRotated(
                keys=keys,
                range_x=(-0.15, 0.15),  # ±8.6 degrees
                range_y=(-0.15, 0.15),
                range_z=(-0.1, 0.1),    # ±5.7 degrees
                prob=0.3,
                mode=("bilinear", "nearest"),
                align_corners=True
            ),
            
            # Minimal intensity adjustment
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.3),
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

class FocusedTumorModel(nn.Module):
    """Model specifically designed for small tumors"""
    def __init__(self, out_channels=2, use_checkpoint=False):
        super().__init__()
        
        # Use architecture more suitable for small targets
        self.segmentation_model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),  # Moderate number of channels
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
            norm=Norm.BATCH,
        )
        
        # Dedicated small target attention
        self.tumor_attention = nn.Sequential(
            nn.Conv3d(out_channels, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid()
        )
        
        self.use_checkpoint = use_checkpoint
        self.inference_mode = False
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            seg_output = torch.utils.checkpoint.checkpoint(
                self.segmentation_model, x, use_reentrant=False
            )
        else:
            seg_output = self.segmentation_model(x)
        
        # Apply dedicated tumor attention
        attention_weights = self.tumor_attention(seg_output)
        # Only enhance foreground channel
        enhanced_output = seg_output.clone()
        enhanced_output[:, 1:2] = seg_output[:, 1:2] * (1 + 2.0 * attention_weights)
        
        return enhanced_output
    
    def inference(self, x):
        self.inference_mode = True
        result = self.forward(x)
        self.inference_mode = False
        return result

class AggressiveFocalTverskyLoss(nn.Module):
    """Aggressive Focal Tversky Loss specifically for small targets"""
    def __init__(self, alpha=0.2, beta=0.8, gamma=3.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tversky = TverskyLoss(
            to_onehot_y=True, softmax=True,
            alpha=alpha, beta=beta,
            smooth_nr=1e-8, smooth_dr=1e-8
        )
    
    def forward(self, pred, target):
        tversky = self.tversky(pred, target)
        # More aggressive focal term
        focal_tversky = torch.pow(tversky, self.gamma)
        return focal_tversky

def create_data_loaders(train_files, val_files):
    """Create data loaders"""
    train_transforms = get_transforms("train", memory_efficient=MEMORY_EFFICIENT)
    val_transforms = get_transforms("val", memory_efficient=MEMORY_EFFICIENT)
    
    print(f"Creating data loaders...")
    
    if USE_CACHE_DATASET:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=0.9,  # Maximum caching
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
    """Calculate multiple metrics"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    # Dice
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + epsilon) / (pred_binary.sum() + target_binary.sum() + epsilon)
    
    # IoU
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    
    # Precision & Recall
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

def validate_model_fixed(model, val_loader, device, epoch):
    """Fixed validation function - single-scale inference"""
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
            
            # 🔧 Fix: Single-scale inference to avoid parameter conflicts
            try:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits = sliding_window_inference(
                        val_inputs, 
                        SPATIAL_SIZE, 
                        2, 
                        model.inference,
                        overlap=0.6, 
                        mode="gaussian"
                        # Remove sw_batch_size parameter to avoid conflicts
                    )
            except Exception as e:
                print(f"Inference failed: {e}")
                # If sliding window fails, use direct forward pass
                logits = model.inference(val_inputs)
            
            # Post-processing
            pred_softmax = F.softmax(logits, dim=1)
            pred_binary = (pred_softmax[:, 1:2] > 0.5).float()
            
            # Calculate metrics
            target_binary = (val_labels > 0).float()
            metrics = calculate_metrics(pred_binary, target_binary)
            all_metrics.append(metrics)
            
            # Save visualization
            if SAVE_DEBUG_IMAGES and idx == 0:
                save_enhanced_visualization(val_inputs, val_labels, pred_binary, epoch, idx)
            
            torch.cuda.empty_cache()
    
    # Average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    else:
        avg_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    return avg_metrics, all_metrics

def save_enhanced_visualization(inputs, labels, predictions, epoch, idx):
    """Improved visualization saving"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        img_np = inputs[0, 0].detach().cpu().numpy()
        lab_np = labels[0, 0].detach().cpu().numpy() if labels.dim() == 5 else labels[0].detach().cpu().numpy()
        pred_np = predictions[0, 0].detach().cpu().numpy()
        
        # Find slices with labels
        label_slices = np.where(lab_np.sum(axis=(0,1)) > 0)[0]
        if len(label_slices) == 0:
            # If no labels, select middle slice
            z_slices = [img_np.shape[2] // 2]
        else:
            # Select slices with labels
            z_slices = [label_slices[len(label_slices)//2]]
        
        # Add middle slice for comparison
        middle_z = img_np.shape[2] // 2
        if middle_z not in z_slices:
            z_slices.append(middle_z)
        
        fig, axes = plt.subplots(len(z_slices), 4, figsize=(16, 4*len(z_slices)))
        if len(z_slices) == 1:
            axes = axes.reshape(1, -1)
        
        for i, z in enumerate(z_slices):
            if z >= img_np.shape[2]:
                z = img_np.shape[2] - 1
                
            axes[i, 0].imshow(img_np[:, :, z], cmap='gray')
            axes[i, 0].set_title(f'Input Slice {z} (Epoch {epoch})')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(lab_np[:, :, z], cmap='Reds', alpha=0.8)
            axes[i, 1].set_title(f'Ground Truth (Sum: {lab_np[:,:,z].sum():.0f})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_np[:, :, z], cmap='Blues', alpha=0.8)
            axes[i, 2].set_title(f'Prediction (Sum: {pred_np[:,:,z].sum():.0f})')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(img_np[:, :, z], cmap='gray', alpha=0.7)
            axes[i, 3].imshow(lab_np[:, :, z], cmap='Reds', alpha=0.4)
            axes[i, 3].imshow(pred_np[:, :, z], cmap='Blues', alpha=0.3)
            axes[i, 3].set_title('Overlay')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(DEBUG_DIR, f"focused_prediction_epoch{epoch}_{idx}.png"),
            dpi=100, bbox_inches='tight'
        )
        plt.close()
        
    except Exception as e:
        print(f"Failed to save visualization: {e}")

def train_one_epoch_focused(model, train_loader, optimizer, loss_functions, device, scaler, epoch):
    """Training function focused on small tumors"""
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
        
        if fg_ratio > 0.001:  # Record meaningful positive samples
            positive_samples += 1
        
        if batch_idx < 5 and epoch < 10:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Foreground ratio {fg_ratio:.6f}")
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda', enabled=True):
            seg_outputs = model(inputs)
            
            # Calculate loss
            main_loss = loss_functions['main'](seg_outputs, labels)
            
            # If current batch has positive samples, increase weight
            if fg_ratio > 0.001:
                main_loss = main_loss * 2.0  # Weight patches containing tumors
            
            loss = main_loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        
        # Memory cleanup
        if batch_idx % 3 == 0:
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
    print("=" * 70)
    print("🎯 Focused Small Tumor Segmentation Training System - Aggressive Optimization")
    print("=" * 70)
    
    # Set random seeds
    current_seed = set_random_seeds()
    
    print(f"Configuration:")
    print(f"  - Random seed: {current_seed}")
    print(f"  - Patch size: {SPATIAL_SIZE}")
    print(f"  - Pos/Neg ratio: {POS_NEG_RATIO}:1 (aggressive positive sampling)")
    print(f"  - Samples per image: {NUM_SAMPLES_PER_IMAGE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Gradient accumulation steps: {GRADIENT_ACCUMULATION}")
    print(f"  - Multi-scale inference: {USE_MULTI_SCALE} (temporarily disabled)")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    # Data preparation
    train_files, val_files = prepare_data(DATA_DIR)
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    # Data loaders
    train_loader, val_loader = create_data_loaders(train_files, val_files)
    print(f"Steps per epoch: {len(train_loader)}")
    
    # Model focused on small tumors
    print(f"Building focused small tumor segmentation model...")
    model = FocusedTumorModel(
        out_channels=2,
        use_checkpoint=MEMORY_EFFICIENT,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer - more aggressive settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-6,  # Reduce regularization
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Improved learning rate schedule - longer warmup
    def improved_cosine_annealing_with_warmup(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            cos_epoch = epoch - WARMUP_EPOCHS
            cos_epochs = MAX_EPOCHS - WARMUP_EPOCHS
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * cos_epoch / cos_epochs))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=improved_cosine_annealing_with_warmup)
    
    # Mixed precision
    scaler = torch.amp.GradScaler(
        init_scale=2.**14,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    
    # 🔥 Loss function specifically for small tumors
    dice_loss = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=False,
        smooth_nr=1e-8,
        smooth_dr=1e-8
    )
    
    aggressive_focal_tversky = AggressiveFocalTverskyLoss(
        alpha=0.2, beta=0.8, gamma=3.0  # Aggressive parameters
    )
    
    # Extremely high weight cross entropy
    ce_loss = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 50.0]).to(device)  # 50:1 weight ratio
    )
    
    def aggressive_tumor_loss(pred, target):
        """Aggressive loss function specifically for small tumors"""
        dice = dice_loss(pred, target)
        focal_tversky = aggressive_focal_tversky(pred, target)
        ce = ce_loss(pred, target.squeeze(1))
        
        # Aggressive combination: more focus on recall
        return 0.3 * dice + 0.5 * focal_tversky + 0.2 * ce
    
    loss_functions = {
        'main': aggressive_tumor_loss
    }
    
    print("Loss function: 0.3*Dice + 0.5*AggressiveFocalTversky + 0.2*CE (50:1 weight)")
    
    # TensorBoard
    log_dir = os.path.join(MODEL_DIR, "focused_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training variables
    best_dice = -1
    best_recall = -1  # Also track recall
    best_metrics = {}
    best_epoch = -1
    early_stop_counter = 0
    
    print(f"\nStarting training for {MAX_EPOCHS} epochs...")
    print("=" * 70)
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        try:
            # Training
            epoch_loss, avg_fg_ratio, positive_samples = train_one_epoch_focused(
                model, train_loader, optimizer, loss_functions, device, scaler, epoch
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
                
                avg_metrics, all_metrics = validate_model_fixed(
                    model, val_loader, device, epoch
                )
                
                val_time = time.time() - val_start
                print(f"Validation: {val_time:.1f}s")
                
                # Log all metrics
                for key, value in avg_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                
                current_dice = avg_metrics['dice']
                current_recall = avg_metrics['recall']
                
                # Update best results - consider both Dice and Recall
                improvement = False
                if current_dice > best_dice:
                    improvement = True
                elif current_dice == best_dice and current_recall > best_recall:
                    improvement = True
                
                if improvement:
                    best_dice = current_dice
                    best_recall = current_recall
                    best_metrics = avg_metrics.copy()
                    best_epoch = epoch + 1
                    
                    # Save best model
                    model_path = os.path.join(MODEL_DIR, f"focused_best_model.pth")
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
                        }
                    }
                    torch.save(checkpoint, model_path)
                    
                    print(f"🎉 New best model! (Dice improved or equal Dice with better Recall)")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                # Display detailed metrics
                print(f"Metrics:")
                print(f"  Dice: {current_dice:.6f} (Best: {best_dice:.6f})")
                print(f"  IoU: {avg_metrics['iou']:.6f}")
                print(f"  Precision: {avg_metrics['precision']:.6f}")
                print(f"  Recall: {current_recall:.6f} (Best: {best_recall:.6f})")
                
                # Progress assessment
                if best_dice > 0.5:
                    print("🚀 Excellent performance! Model is doing great!")
                elif best_dice > 0.3:
                    print("✅ Good performance! Continuing optimization...")
                elif best_dice > 0.15:
                    print("📈 Significant progress! On the right track!")
                elif best_dice > 0.05:
                    print("🔄 Some improvement, continue training...")
                elif best_dice > 0.01:
                    print("💪 Starting to learn, be patient...")
                else:
                    print("⚠️  Model still adapting, needs more time...")
                
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print("⏹️  Early stopping")
                    break
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Training summary
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "=" * 70)
    print("🏁 Training completed!")
    print("=" * 70)
    print(f"Best results (Epoch {best_epoch}):")
    print(f"  Dice: {best_dice:.6f}")
    print(f"  IoU: {best_metrics.get('iou', 0):.6f}")
    print(f"  Precision: {best_metrics.get('precision', 0):.6f}")
    print(f"  Recall: {best_metrics.get('recall', 0):.6f}")
    print(f"Total training time: {total_time:.1f} minutes")
    print(f"Random seed: {current_seed}")
    
    # Performance assessment
    if best_dice > 0.5:
        print("🎊 Congratulations! Excellent performance achieved!")
    elif best_dice > 0.3:
        print("👍 Good performance achieved! Model is usable!")
    elif best_dice > 0.15:
        print("📊 Significant progress made, recommend further optimization!")
    elif best_dice > 0.05:
        print("🔧 Some improvement, can adjust hyperparameters for further optimization")
    else:
        print("🔍 Need to check data quality or adjust strategy")
    
    print(f"📁 Models and logs saved in: {MODEL_DIR}")
    
    # Cleanup resources
    writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n💥 System error: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
