import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import multiprocessing as mp
from contextlib import nullcontext

import monai
from monai.data import Dataset, list_data_collate, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandCropByPosNegLabeld, RandRotate90d, 
    RandShiftIntensityd, RandFlipd, Spacingd, Orientationd,
    SpatialPadd, EnsureTyped, ToTensord, MapTransform
)
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet

# Task configuration
TASK_TYPE = "vessel"  # Modify this during training: "liver", "vessel", "tumor"

# Windows compatibility settings
if os.name == 'nt':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# High-performance GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

monai.utils.set_determinism(seed=42)

# Dynamic path configuration
DATA_DIR = rf".\{TASK_TYPE}_dataset"
MODEL_DIR = rf".\{TASK_TYPE}_model"
TRAIN_PCT = 0.8

# Performance optimization parameters - maintain original settings
BATCH_SIZE = 2              
GRADIENT_ACCUMULATION = 1  
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION
NUM_WORKERS = 6 if os.name == 'nt' else 6
MAX_EPOCHS = 100
VAL_INTERVAL = 20
SPATIAL_SIZE = [96, 96, 96]
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-4

# Advanced optimization settings
USE_CACHE_DATASET = True    
USE_COMPILE = False         
ENABLE_PROFILING = False    
MEMORY_EFFICIENT = True     

os.makedirs(MODEL_DIR, exist_ok=True)

class CleanLabelsd(MapTransform):
    """Transform class for cleaning label data - converts all positive values to 1"""
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if key in d:
                labels = d[key]
                # Handle MetaTensor/Tensor/Array directly
                if hasattr(labels, 'numpy'):  # MetaTensor or torch.Tensor
                    # Process as torch tensor
                    if not isinstance(labels, torch.Tensor):
                        labels = torch.as_tensor(labels)
                    cleaned = torch.where(labels > 0.5, 1.0, 0.0)
                    d[key] = cleaned
                elif hasattr(labels, 'get_fdata'):  # nibabel object
                    labels_array = labels.get_fdata()
                    cleaned = np.where(labels_array > 0.5, 1.0, 0.0)
                    import nibabel as nib
                    cleaned_nii = nib.Nifti1Image(cleaned.astype(np.float32), labels.affine, labels.header)
                    d[key] = cleaned_nii
                else:
                    # Regular numpy array
                    cleaned = np.where(labels > 0.5, 1.0, 0.0)
                    d[key] = cleaned.astype(np.float32)
        return d

def prepare_data(data_dir):
    """Prepare dataset"""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    print(f"Looking for data at: {images_dir}")
    print(f"Current task type: {TASK_TYPE}")
    
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

    # Dynamic label key
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE

    data_dicts = [
        {"image": image_name, label_key: label_name}
        for image_name, label_name in zip(images, labels)
    ]

    np.random.shuffle(data_dicts)
    train_size = int(len(data_dicts) * TRAIN_PCT)
    train_files, val_files = data_dicts[:train_size], data_dicts[train_size:]

    return train_files, val_files

def get_transforms(phase, memory_efficient=False):
    """Get data transformations"""
    # Dynamic keys configuration
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    keys = ["image", label_key]
    
    if memory_efficient:
        common_transforms = [
            LoadImaged(keys=keys),
            CleanLabelsd(keys=[label_key]),  # Clean labels immediately
            EnsureChannelFirstd(keys=keys),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            ToTensord(keys=keys),
        ]
    else:
        common_transforms = [
            LoadImaged(keys=keys),
            CleanLabelsd(keys=[label_key]),  # Clean labels immediately
            EnsureChannelFirstd(keys=keys),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=keys, axcodes="RAS"),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ]

    if phase == "train":
        train_specific = [
            SpatialPadd(keys=keys, spatial_size=SPATIAL_SIZE),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=label_key,
                spatial_size=SPATIAL_SIZE,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
        
        # Adjust data augmentation based on task type
        if TASK_TYPE == "vessel":
            if memory_efficient:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                    RandRotate90d(keys=keys, prob=0.4, max_k=1),
                ]
            else:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=keys, prob=0.5, max_k=3),
                    RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.6),
                ]
        elif TASK_TYPE == "tumor":
            if memory_efficient:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.4, spatial_axis=0),
                    RandRotate90d(keys=keys, prob=0.3, max_k=1),
                ]
            else:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=keys, prob=0.5, max_k=2),
                    RandShiftIntensityd(keys=["image"], offsets=0.12, prob=0.5),
                ]
        else:  # liver
            if memory_efficient:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.3, spatial_axis=0),
                    RandRotate90d(keys=keys, prob=0.3, max_k=1),
                ]
            else:
                augmentations = [
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                    RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=keys, prob=0.5, max_k=3),
                    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
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

class OptimizedLiverSegClassifier(nn.Module):
    """Optimized segmentation + classification model"""
    def __init__(self, out_channels=2, use_checkpoint=False, task_type="liver"):
        super().__init__()
        
        # Adjust network based on task type
        if task_type == "vessel":
            self.segmentation_model = SegResNet(
                blocks_down=[2, 2, 4, 4],
                blocks_up=[1, 1, 1],
                init_filters=32,
                in_channels=1,
                out_channels=out_channels,
                dropout_prob=0.15,
                use_conv_final=True,
            )
        elif task_type == "tumor":
            self.segmentation_model = SegResNet(
                blocks_down=[1, 2, 3, 4],
                blocks_up=[1, 1, 1],
                init_filters=24,
                in_channels=1,
                out_channels=out_channels,
                dropout_prob=0.2,
                use_conv_final=True,
            )
        else:  # liver
            self.segmentation_model = SegResNet(
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=1,
                out_channels=out_channels,
                dropout_prob=0.2,
                use_conv_final=True,
            )

        # Only liver task has classification head
        if task_type == "liver":
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(out_channels, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
        else:
            self.classification_head = None

        self.use_checkpoint = use_checkpoint
        self.task_type = task_type
        self.inference_mode = False

    def forward(self, x):
        if self.use_checkpoint and self.training:
            seg_output = torch.utils.checkpoint.checkpoint(
                self.segmentation_model, x, use_reentrant=False
            )
        else:
            seg_output = self.segmentation_model(x)
        
        if hasattr(self, 'inference_mode') and self.inference_mode:
            return seg_output
            
        # Only liver task returns classification results
        if self.classification_head is not None:
            cls_output = self.classification_head(seg_output)
            return seg_output, cls_output
        else:
            return seg_output

    def inference(self, x):
        self.inference_mode = True
        if self.use_checkpoint:
            seg_output = torch.utils.checkpoint.checkpoint(
                self.segmentation_model, x, use_reentrant=False
            )
        else:
            seg_output = self.segmentation_model(x)
        self.inference_mode = False
        return seg_output

def create_optimized_data_loaders(train_files, val_files):
    """Create high-performance data loaders"""
    train_transforms = get_transforms("train", memory_efficient=MEMORY_EFFICIENT)
    val_transforms = get_transforms("val", memory_efficient=MEMORY_EFFICIENT)

    print(f"Creating data loaders...")
    
    if USE_CACHE_DATASET and len(train_files) < 1000:
        print("  Using CacheDataset for accelerated data loading")
        cache_dir = os.path.join(MODEL_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        train_ds = CacheDataset(
            data=train_files, 
            transform=train_transforms,
            cache_rate=0.5,
            num_workers=NUM_WORKERS,
        )
        val_ds = CacheDataset(
            data=val_files, 
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=NUM_WORKERS,
        )
    else:
        print("  Using standard Dataset")
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)

    # Maintain original DataLoader settings
    train_loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': NUM_WORKERS,
        'collate_fn': list_data_collate,
        'pin_memory': torch.cuda.is_available(),
        'drop_last': True,
    }
    
    if NUM_WORKERS > 0:
        train_loader_kwargs.update({
            'prefetch_factor': 2,
            'persistent_workers': True,
        })

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader

def train_one_epoch_clean(model, train_loader, optimizer, seg_loss_fn, cls_loss_fn, 
                         device, scaler, epoch):
    """Train one epoch"""
    model.train()
    epoch_loss = 0
    seg_loss_epoch = 0
    cls_loss_epoch = 0
    step = 0
    
    # Dynamic label key
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    
    accumulation_steps = GRADIENT_ACCUMULATION
    optimizer.zero_grad()

    for batch_idx, batch_data in enumerate(train_loader):
        step += 1
        
        inputs, labels = (
            batch_data["image"].to(device, non_blocking=True),
            batch_data[label_key].to(device, non_blocking=True),
        )

        # Check labels (should already be cleaned to 0 and 1)
        if batch_idx == 0 and epoch == 0:
            unique_labels = torch.unique(labels)
            print(f"Label values in training: {unique_labels.cpu().numpy()}")

        # Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', enabled=True):
            if TASK_TYPE == "liver":
                # Liver task: segmentation + classification
                seg_outputs, cls_outputs = model(inputs)
                cls_labels = (torch.sum(labels, dim=[1, 2, 3, 4]) > 0).float()
                seg_loss = seg_loss_fn(seg_outputs, labels)
                cls_loss = cls_loss_fn(cls_outputs.squeeze(), cls_labels)
                loss = (seg_loss + 0.5 * cls_loss) / accumulation_steps
                cls_loss_epoch += cls_loss.item()
            else:
                # Vessel/tumor task: segmentation only
                seg_outputs = model(inputs)
                seg_loss = seg_loss_fn(seg_outputs, labels)
                loss = seg_loss / accumulation_steps
                cls_loss_epoch = 0

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        seg_loss_epoch += seg_loss.item()

    if len(train_loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return epoch_loss / step, seg_loss_epoch / step, cls_loss_epoch / step

def validate_model_optimized(model, val_loader, device, dice_metric, hausdorff_metric):
    """Optimized model validation"""
    # Dynamic label key
    label_key = "label" if TASK_TYPE == "liver" else TASK_TYPE
    
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device, non_blocking=True),
                val_data[label_key].to(device, non_blocking=True),
            )

            roi_size = (96, 96, 96)
            sw_batch_size = 4
            overlap = 0.5

            with torch.amp.autocast(device_type='cuda', enabled=True):
                val_seg_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model.inference,
                    overlap=overlap, mode="gaussian"
                )

            val_seg_outputs_argmax = torch.argmax(val_seg_outputs, dim=1, keepdim=True)
            dice_metric(y_pred=val_seg_outputs_argmax, y=val_labels)
            hausdorff_metric(y_pred=val_seg_outputs_argmax, y=val_labels)

        metric_dice = dice_metric.aggregate().item()
        metric_hausdorff = hausdorff_metric.aggregate().item()

        dice_metric.reset()
        hausdorff_metric.reset()

    return metric_dice, metric_hausdorff

def main():
    print(f"Starting {TASK_TYPE} segmentation model training...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    # Prepare data
    train_files, val_files = prepare_data(DATA_DIR)
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    # Create data loaders
    train_loader, val_loader = create_optimized_data_loaders(train_files, val_files)
    print(f"Steps per epoch: {len(train_loader)}, Effective batch size: {EFFECTIVE_BATCH_SIZE}")

    # Create model
    print(f"Building {TASK_TYPE} segmentation model...")
    model = OptimizedLiverSegClassifier(
        out_channels=2, 
        use_checkpoint=MEMORY_EFFICIENT,
        task_type=TASK_TYPE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-5,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler(
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )

    # Adjust loss function based on task type
    if TASK_TYPE == "vessel":
        seg_loss_fn = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            weight=torch.tensor([1.0, 4.0]).to(device),
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    elif TASK_TYPE == "tumor":
        seg_loss_fn = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            weight=torch.tensor([1.0, 6.0]).to(device),
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    else:  # liver
        seg_loss_fn = DiceCELoss(
            to_onehot_y=True, 
            softmax=True, 
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
    
    # Classification loss only for liver
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0).to(device)) if TASK_TYPE == "liver" else None

    # Evaluation metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")

    # TensorBoard
    log_dir = os.path.join(MODEL_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    early_stop_counter = 0

    print(f"\nStarting training for {MAX_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        try:
            # Training
            epoch_loss, seg_loss_epoch, cls_loss_epoch = train_one_epoch_clean(
                model, train_loader, optimizer, seg_loss_fn, cls_loss_fn, 
                device, scaler, epoch
            )

            # Log training metrics
            writer.add_scalar("train/total_loss", epoch_loss, epoch)
            writer.add_scalar("train/segmentation_loss", seg_loss_epoch, epoch)
            if TASK_TYPE == "liver":
                writer.add_scalar("train/classification_loss", cls_loss_epoch, epoch)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], epoch)

            # Update learning rate
            scheduler.step()

            epoch_time = time.time() - epoch_start
            samples_per_second = len(train_loader) * BATCH_SIZE / epoch_time
            
            print(f"Training complete: {epoch_time:.1f}s, "
                  f"Average loss: {epoch_loss:.4f}, "
                  f"Throughput: {samples_per_second:.1f} samples/s")

            # Validation
            if (epoch + 1) % VAL_INTERVAL == 0:
                print("Validating...")
                val_start = time.time()
                
                metric_dice, metric_hausdorff = validate_model_optimized(
                    model, val_loader, device, dice_metric, hausdorff_metric
                )

                val_time = time.time() - val_start
                print(f"Validation complete: {val_time:.1f}s")

                writer.add_scalar("val/dice", metric_dice, epoch)
                writer.add_scalar("val/hausdorff", metric_hausdorff, epoch)

                if metric_dice > best_metric:
                    best_metric = metric_dice
                    best_metric_epoch = epoch + 1
                    best_model_path = os.path.join(MODEL_DIR, f"best_{TASK_TYPE}_model.pth")
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_metric': best_metric,
                        'task_type': TASK_TYPE,
                        'config': {
                            'batch_size': BATCH_SIZE,
                            'gradient_accumulation': GRADIENT_ACCUMULATION,
                            'learning_rate': LEARNING_RATE,
                            'spatial_size': SPATIAL_SIZE,
                        }
                    }
                    torch.save(checkpoint, best_model_path)
                    print(f"New best {TASK_TYPE} model! Dice: {best_metric:.4f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                print(f"Current Dice: {metric_dice:.4f}, Best: {best_metric:.4f} (Epoch {best_metric_epoch})")

                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory: {e}")
                print("Suggestion: Reduce BATCH_SIZE or enable MEMORY_EFFICIENT mode")
                break
            else:
                print(f"Error at epoch {epoch + 1}: {e}")
                break
        except Exception as e:
            print(f"Error at epoch {epoch + 1}: {e}")
            break

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, f"final_{TASK_TYPE}_model.pth")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'training_completed': True,
        'total_epochs': epoch + 1,
        'best_metric': best_metric,
        'task_type': TASK_TYPE,
        'config': {
            'batch_size': BATCH_SIZE,
            'gradient_accumulation': GRADIENT_ACCUMULATION,
            'effective_batch_size': EFFECTIVE_BATCH_SIZE,
        }
    }
    torch.save(final_checkpoint, final_model_path)

    # Training summary
    total_time = (time.time() - start_time) / 60
    avg_epoch_time = total_time * 60 / (epoch + 1)
    
    print(f"\n{TASK_TYPE} model training complete!")
    print(f"Best Dice: {best_metric:.4f} (Epoch {best_metric_epoch})")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Average per epoch: {avg_epoch_time:.1f} seconds")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"TensorBoard logs: {log_dir}")
    
    # Cleanup
    writer.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{TASK_TYPE} training interrupted by user")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n{TASK_TYPE} training error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
