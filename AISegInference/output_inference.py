import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import json
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, ToTensord, MapTransform
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm

# Use trimesh only for 3D reconstruction, not dependent on Open3D
try:
    import trimesh
    from trimesh import Trimesh
    HAS_TRIMESH = True
    print("Trimesh library loaded successfully")
except ImportError:
    HAS_TRIMESH = False
    print("Trimesh library not found. Please install with: pip install trimesh")

# PySide6 imports (optional)
try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
    from PySide6.QtCore import Qt
    HAS_PYSIDE6 = True
    print("PySide6 library loaded successfully")
except ImportError:
    HAS_PYSIDE6 = False
    print("PySide6 not found. Qt GUI features will be disabled.")

# Define all required model classes (same as before)
class OptimizedLiverSegClassifier(nn.Module):
    """Optimized segmentation + classification model (for liver and vessel)"""
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

class FocusedTumorModel(nn.Module):
    """Model specifically for small tumors (UNet architecture)"""
    def __init__(self, out_channels=2, use_checkpoint=False):
        super().__init__()
        
        # Using UNet architecture
        self.segmentation_model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
            norm=Norm.BATCH,
        )
        
        # Specialized small target attention
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
        
        # Apply specialized tumor attention
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

class Organ3DModel:
    """Independent organ 3D model class, based on Trimesh, compatible with Python 3.13"""
    
    def __init__(self, organ_name, mesh_data=None, color=None):
        self.organ_name = organ_name
        self.mesh_data = mesh_data  # Trimesh object
        self.color = color or [0.8, 0.8, 0.8]
        self.visible = True
        self.file_paths = {}
        self.statistics = {}
        
        # Qt object reference for PySide6
        self.qt_mesh_item = None
        
    def set_mesh(self, mesh):
        """Set mesh data"""
        self.mesh_data = mesh
        
    def set_color(self, color):
        """Set color"""
        self.color = color
        if self.mesh_data and hasattr(self.mesh_data, 'visual'):
            # Trimesh color setting
            self.mesh_data.visual.face_colors = [int(c*255) for c in color] + [255]
    
    def set_visibility(self, visible):
        """Set visibility - supports PySide6"""
        self.visible = visible
        
        # If Qt object exists, update its visibility too
        if self.qt_mesh_item and hasattr(self.qt_mesh_item, 'setVisible'):
            self.qt_mesh_item.setVisible(visible)
        
    def export_mesh(self, output_path, format='obj'):
        """Export mesh model"""
        if not self.mesh_data:
            return False
            
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Trimesh export
            self.mesh_data.export(str(output_path))
            self.file_paths[format] = str(output_path)
            return True
        except Exception as e:
            print(f"Error exporting {self.organ_name} mesh: {e}")
            return False
    
    def get_statistics(self):
        """Get model statistics"""
        if not self.mesh_data:
            return {"organ_name": self.organ_name}
            
        stats = {"organ_name": self.organ_name}
        
        try:
            if hasattr(self.mesh_data, 'vertices'):
                stats["vertices"] = len(self.mesh_data.vertices)
            if hasattr(self.mesh_data, 'faces'):
                stats["faces"] = len(self.mesh_data.faces)
            if hasattr(self.mesh_data, 'volume'):
                stats["volume_mm3"] = float(self.mesh_data.volume)
            if hasattr(self.mesh_data, 'area'):
                stats["surface_area_mm2"] = float(self.mesh_data.area)
            if hasattr(self.mesh_data, 'is_watertight'):
                stats["is_watertight"] = bool(self.mesh_data.is_watertight)
                
        except Exception as e:
            print(f"Error calculating statistics for {self.organ_name}: {e}")
            
        return stats

class TrimeshMask3DGenerator:
    """Generator class for creating 3D models from masks using Trimesh"""
    
    def __init__(self):
        self.class_colors = {
            "liver": [0.8, 0.4, 0.4],  # Liver: dark red
            "vessel": [0.2, 0.6, 0.8],  # Vessel: blue
            "tumor": [0.9, 0.7, 0.2]   # Tumor: yellow
        }
        
    def mask_to_mesh_marching_cubes(self, mask, spacing, origin=[0, 0, 0], step_size=1):
        """Convert mask directly to mesh using Marching Cubes algorithm"""
        if not HAS_TRIMESH:
            raise ImportError("Trimesh is required for 3D reconstruction")
            
        print(f"    Generating mesh using Marching Cubes algorithm...")
        
        try:
            # Use skimage's marching_cubes algorithm
            from skimage.measure import marching_cubes
            
            # Apply smoothing to mask
            from scipy import ndimage
            smoothed_mask = ndimage.gaussian_filter(mask.astype(float), sigma=0.5)
            
            # Execute marching cubes
            verts, faces, normals, values = marching_cubes(
                smoothed_mask, 
                level=0.5,
                step_size=step_size,
                allow_degenerate=False
            )
            
            # Apply spacing transformation to physical coordinates
            verts[:, 0] *= spacing[0]
            verts[:, 1] *= spacing[1] 
            verts[:, 2] *= spacing[2]
            
            # Apply origin offset
            verts += np.array(origin)
            
            # Create Trimesh object
            mesh = Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Fix mesh
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            # Try to fix inconsistent normals
            if not mesh.is_winding_consistent:
                mesh.fix_normals()
            
            print(f"    Marching Cubes complete: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangular faces")
            return mesh
            
        except Exception as e:
            print(f"    Marching Cubes error: {e}")
            return None
    
    def mask_to_mesh_voxel_grid(self, mask, spacing, origin=[0, 0, 0]):
        """Generate mesh using voxel grid method (backup method)"""
        if not HAS_TRIMESH:
            raise ImportError("Trimesh is required for 3D reconstruction")
            
        print(f"    Generating mesh using voxel grid method...")
        
        try:
            # Create voxel grid
            from trimesh.voxel import VoxelGrid
            
            # Find non-zero positions in mask
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
                
            # Convert to voxel coordinates
            voxel_coords = np.column_stack(coords)
            
            # Create voxel grid
            voxel_grid = VoxelGrid(
                encoding=voxel_coords,
                transform=np.eye(4)  # Identity transformation matrix
            )
            
            # Convert to mesh
            mesh = voxel_grid.marching_cubes
            
            if mesh and len(mesh.vertices) > 0:
                # Apply spacing scaling
                scale_matrix = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
                mesh.apply_transform(scale_matrix)
                
                # Apply origin translation
                translation_matrix = np.eye(4)
                translation_matrix[:3, 3] = origin
                mesh.apply_transform(translation_matrix)
                
                print(f"    Voxel grid complete: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangular faces")
                return mesh
            else:
                print("    Voxel grid method failed")
                return None
                
        except Exception as e:
            print(f"    Voxel grid error: {e}")
            return None
    
    def generate_organ_3d_model(self, mask, organ_name, spacing, origin=[0, 0, 0], 
                               method='marching_cubes', **kwargs):
        """Generate 3D model for a single organ"""
        print(f"  Generating 3D model for {organ_name}...")
        
        # Check if mask is empty
        voxel_count = np.sum(mask > 0)
        if voxel_count < 100:
            print(f"    Skipping {organ_name}: too few voxels ({voxel_count})")
            return None
        
        # Select reconstruction method
        if method == 'marching_cubes':
            mesh = self.mask_to_mesh_marching_cubes(
                mask, spacing, origin,
                step_size=kwargs.get('step_size', 1)
            )
        elif method == 'voxel_grid':
            mesh = self.mask_to_mesh_voxel_grid(mask, spacing, origin)
        else:
            print(f"    Unsupported reconstruction method: {method}")
            return None
            
        if mesh is None or len(mesh.vertices) == 0:
            print(f"    {organ_name} 3D reconstruction failed")
            return None
            
        # Create organ 3D model object
        organ_model = Organ3DModel(
            organ_name=organ_name,
            mesh_data=mesh,
            color=self.class_colors.get(organ_name.lower(), [0.8, 0.8, 0.8])
        )
        
        # Set color
        organ_model.set_color(organ_model.color)
        
        print(f"    Successfully generated {organ_name} 3D model")
        return organ_model

class LiverAnalysisPipelineWith3D:
    """Liver AI analysis pipeline with 3D model generation (using Trimesh, compatible with Python 3.13)"""
    
    def __init__(self, model_paths, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_paths = model_paths
        self.models = {}
        
        # Standardization parameters (consistent with training)
        self.target_spacing = (1.5, 1.5, 2.0)
        self.hu_range = (-200, 250)
        self.roi_size = (96, 96, 96)
        
        # Class definitions and color mapping
        self.class_names = {
            0: "Background",
            1: "Liver", 
            2: "Hepatic Vessel",
            3: "Hepatic Tumor"
        }
        
        self.class_colors = {
            1: [0.0, 1.0, 0.0, 0.6],  # Liver: green, alpha 0.6
            2: [1.0, 0.0, 0.0, 0.8],  # Vessel: red, alpha 0.8
            3: [1.0, 1.0, 0.0, 0.9]   # Tumor: yellow, alpha 0.9
        }
        
        # Diagnostic criteria
        self.diagnosis_criteria = {
            "liver_volume_normal": (800, 2200),
            "tumor_size_small": 2.0,
            "tumor_size_large": 20.0,
            "tumor_liver_ratio_high": 15.0,
            "tumor_liver_ratio_moderate": 5.0,
            "vessel_ratio_high": 8.0,
            "vessel_ratio_low": 3.0,
        }
        
        # 3D model generator (using Trimesh)
        self.mask_3d_generator = TrimeshMask3DGenerator()
        
        print(f"Initializing Liver AI Analysis Pipeline (Trimesh-based 3D reconstruction)...")
        print(f"Using device: {self.device}")
        print(f"3D reconstruction support: {'Yes' if HAS_TRIMESH else 'No'}")
        
        self._load_models()
    
    # Other methods remain the same, only need to modify _generate_3d_models method
    def _load_models(self):
        """Load the three trained models"""
        print("Loading models...")
        
        # Liver model - must load successfully
        if "liver" in self.model_paths and os.path.exists(self.model_paths["liver"]):
            print("  Loading liver segmentation model...")
            try:
                checkpoint = torch.load(self.model_paths["liver"], map_location=self.device, weights_only=False)
                liver_model = OptimizedLiverSegClassifier(
                    out_channels=2, 
                    use_checkpoint=False,
                    task_type="liver"
                )
                liver_model.load_state_dict(checkpoint['model_state_dict'])
                liver_model.to(self.device)
                liver_model.eval()
                self.models["liver"] = liver_model
                print(f"    Liver model loaded successfully")
            except Exception as e:
                print(f"    Liver model loading failed: {e}")
                raise RuntimeError("Liver model is required, loading failure will prevent any analysis")
                
        # Vessel model - must load successfully  
        if "vessel" in self.model_paths and os.path.exists(self.model_paths["vessel"]):
            print("  Loading vessel segmentation model...")
            try:
                checkpoint = torch.load(self.model_paths["vessel"], map_location=self.device, weights_only=False)
                vessel_model = OptimizedLiverSegClassifier(
                    out_channels=2, 
                    use_checkpoint=False,
                    task_type="vessel"
                )
                vessel_model.load_state_dict(checkpoint['model_state_dict'])
                vessel_model.to(self.device)
                vessel_model.eval()
                self.models["vessel"] = vessel_model
                print(f"    Vessel model loaded successfully")
            except Exception as e:
                print(f"    Vessel model loading failed: {e}")
                raise RuntimeError("Vessel model is required, loading failure will affect vessel analysis")
                
        # Tumor model - optional, failure won't affect main functionality
        if "tumor" in self.model_paths and os.path.exists(self.model_paths["tumor"]):
            print("  Loading tumor segmentation model...")
            try:
                checkpoint = torch.load(self.model_paths["tumor"], map_location=self.device, weights_only=False)
                tumor_model = FocusedTumorModel(
                    out_channels=2,
                    use_checkpoint=False
                )
                tumor_model.load_state_dict(checkpoint['model_state_dict'])
                tumor_model.to(self.device)
                tumor_model.eval()
                self.models["tumor"] = tumor_model
                print(f"    Tumor model loaded successfully")
            except Exception as e:
                print(f"    Tumor model loading failed: {e}")
                print(f"    Tumor detection will be skipped (this is normal, not everyone has tumors)")
        
        # Check required models
        required_models = ["liver", "vessel"]
        missing_models = [m for m in required_models if m not in self.models]
        
        if missing_models:
            raise RuntimeError(f"Missing required models: {missing_models}")
        
        print(f"Successfully loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def _preprocess_ct(self, ct_path):
        """Preprocess CT image"""
        print("Preprocessing CT image...")
        
        transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS", labels=None),
            Spacingd(keys=["image"], pixdim=self.target_spacing, mode="bilinear"),
            ScaleIntensityRanged(keys=["image"], 
                               a_min=self.hu_range[0], a_max=self.hu_range[1],
                               b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"]),
        ])
        
        data = {"image": str(ct_path)}
        processed_data = transforms(data)
        
        image_tensor = processed_data["image"].unsqueeze(0).to(self.device)
        
        # Fix metadata access issues
        try:
            if hasattr(processed_data["image"], 'meta') and processed_data["image"].meta is not None:
                meta_dict = processed_data["image"].meta
            elif "image_meta_dict" in processed_data:
                meta_dict = processed_data["image_meta_dict"]
            else:
                meta_dict = {
                    "spacing": list(self.target_spacing),
                    "origin": [0, 0, 0],
                    "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "spatial_shape": list(image_tensor.shape[2:])
                }
        except Exception as e:
            print(f"  Metadata access error, using default values: {e}")
            meta_dict = {
                "spacing": list(self.target_spacing),
                "origin": [0, 0, 0], 
                "direction": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "spatial_shape": list(image_tensor.shape[2:])
            }
        
        print(f"  Original size: {meta_dict.get('spatial_shape', 'Unknown')}")
        print(f"  Processed size: {image_tensor.shape}")
        print(f"  Spacing: {meta_dict.get('spacing', self.target_spacing)}")
        
        return image_tensor, meta_dict
    
    # Prediction methods remain unchanged...
    def _predict_liver(self, image_tensor):
        """Predict liver mask"""
        print("  Step 1: Liver segmentation...")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                if hasattr(self.models["liver"], 'inference'):
                    liver_output = sliding_window_inference(
                        image_tensor, self.roi_size, 4, 
                        self.models["liver"].inference,
                        overlap=0.5, mode="gaussian"
                    )
                else:
                    def seg_only_inference(x):
                        result = self.models["liver"](x)
                        if isinstance(result, tuple):
                            return result[0]
                        return result
                    
                    liver_output = sliding_window_inference(
                        image_tensor, self.roi_size, 4, 
                        seg_only_inference,
                        overlap=0.5, mode="gaussian"
                    )
                    
        liver_mask = torch.argmax(liver_output, dim=1, keepdim=True)
        liver_voxels = torch.sum(liver_mask > 0).item()
        print(f"    Liver voxels detected: {liver_voxels}")
        
        return liver_mask
    
    def _predict_vessel(self, image_tensor, liver_mask, roi_slice):
        """Predict vessel mask"""
        print("  Step 2: Vessel segmentation (within ROI)...")
        
        # Crop to ROI
        image_roi = image_tensor[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        liver_roi = liver_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        
        print(f"    Image size within ROI: {image_roi.shape}")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                roi_shape = image_roi.shape[2:]
                patch_size = (32, 32, 32) if min(roi_shape) < 64 else (64, 64, 64)
                sw_batch_size = 2 if min(roi_shape) < 64 else 4
                
                roi_output = sliding_window_inference(
                    image_roi, patch_size, sw_batch_size,
                    self.models["vessel"],
                    overlap=0.6, mode="gaussian"
                )
        
        roi_mask = torch.argmax(roi_output, dim=1, keepdim=True)
        roi_mask = roi_mask * liver_roi  # Keep only vessels within liver
        
        predicted_voxels = torch.sum(roi_mask > 0).item()
        print(f"    Vessel voxels detected: {predicted_voxels}")
        
        # Restore to original size
        full_mask = torch.zeros_like(liver_mask)
        full_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]] = roi_mask
        
        return full_mask
    
    def _predict_tumor(self, image_tensor, liver_mask, roi_slice):
        """Predict tumor mask (optional)"""
        print("  Step 3: Tumor segmentation (within ROI)...")
        
        if "tumor" not in self.models:
            print("    Skipping tumor segmentation (model not loaded)")
            return torch.zeros_like(liver_mask)
        
        # Crop to ROI
        image_roi = image_tensor[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        liver_roi = liver_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        
        print(f"    Image size within ROI: {image_roi.shape}")
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                roi_shape = image_roi.shape[2:]
                patch_size = (32, 32, 32) if min(roi_shape) < 64 else (64, 64, 64)
                sw_batch_size = 2 if min(roi_shape) < 64 else 4
                
                roi_output = sliding_window_inference(
                    image_roi, patch_size, sw_batch_size,
                    self.models["tumor"],
                    overlap=0.6, mode="gaussian"
                )
        
        roi_mask = torch.argmax(roi_output, dim=1, keepdim=True)
        roi_mask = roi_mask * liver_roi  # Keep only tumors within liver
        
        predicted_voxels = torch.sum(roi_mask > 0).item()
        print(f"    Tumor voxels detected: {predicted_voxels}")
        
        # Restore to original size
        full_mask = torch.zeros_like(liver_mask)
        full_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]] = roi_mask
        
        return full_mask
    
    def _get_liver_roi(self, liver_mask, margin=20):
        """Get ROI bounding box based on liver mask"""
        print("  Computing liver ROI...")
        
        liver_np = liver_mask.squeeze().cpu().numpy()
        coords = np.where(liver_np > 0)
        if len(coords[0]) == 0:
            return None, None
            
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        # Add margin
        shape = liver_np.shape
        z_min = max(0, z_min - margin)
        z_max = min(shape[0], z_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(shape[1], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(shape[2], x_max + margin)
        
        roi_slice = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
        roi_info = {
            "slice": roi_slice,
            "bounds": [(z_min, z_max), (y_min, y_max), (x_min, x_max)],
            "shape": (z_max-z_min, y_max-y_min, x_max-x_min),
        }
        roi_info["reduction_ratio"] = np.prod(roi_info["shape"]) / np.prod(shape)
        
        print(f"    ROI size: {roi_info['shape']} ({roi_info['reduction_ratio']:.1%} of original)")
        
        return roi_slice, roi_info
    
    def _fuse_masks(self, liver_mask, vessel_mask, tumor_mask):
        """Fuse three masks into multi-class labels"""
        print("  Step 4: Fusing segmentation results...")
        
        liver_np = liver_mask.squeeze().cpu().numpy().astype(np.uint8)
        vessel_np = vessel_mask.squeeze().cpu().numpy().astype(np.uint8)
        tumor_np = tumor_mask.squeeze().cpu().numpy().astype(np.uint8) if tumor_mask is not None else np.zeros_like(liver_np)
        
        # Ensure vessels and tumors are only within liver
        vessel_np = vessel_np * liver_np
        tumor_np = tumor_np * liver_np
        
        # Fuse by priority: tumor > vessel > liver
        fused = np.zeros_like(liver_np, dtype=np.uint8)
        fused[liver_np == 1] = 1      # Liver
        fused[vessel_np == 1] = 2     # Vessel (overrides liver)
        fused[tumor_np == 1] = 3      # Tumor (highest priority)
        
        # Statistics of fused results
        unique, counts = np.unique(fused, return_counts=True)
        for label, count in zip(unique, counts):
            if label > 0:
                print(f"    {self.class_names[label]}: {count} voxels")
        
        # Return separate mask dictionary for 3D model generation
        masks = {
            "liver": liver_np,
            "vessel": vessel_np, 
            "tumor": tumor_np
        }
        
        return fused, masks
    
    def _generate_3d_models(self, individual_masks, meta_dict, output_dir):
        """Generate 3D models (using Trimesh)"""
        print("Step 5: Generating 3D models (based on Trimesh)...")
        
        if not HAS_TRIMESH:
            print("  Trimesh not installed, skipping 3D model generation")
            return {}
            
        spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
        origin = meta_dict.get("origin", [0, 0, 0])
        
        organ_3d_models = {}
        
        # Generate 3D model for each organ
        organ_name_map = {
            "liver": "Liver",
            "vessel": "Vessel", 
            "tumor": "Tumor"
        }
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for organ_key, mask in individual_masks.items():
            if np.sum(mask > 0) < 50:  # Too few voxels
                print(f"  Skipping {organ_name_map.get(organ_key, organ_key)}: too few voxels")
                continue
                
            # Use Marching Cubes algorithm
            organ_model = self.mask_3d_generator.generate_organ_3d_model(
                mask=mask,
                organ_name=organ_key,
                spacing=spacing,
                origin=origin,
                method='marching_cubes',
                step_size=2  # Lower resolution for speed
            )
            
            if organ_model:
                organ_3d_models[organ_key] = organ_model
                
                # Export model files
                models_dir = output_dir / "3d_models"
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Export in multiple formats
                for fmt in ['obj', 'ply', 'stl']:
                    file_path = models_dir / f"{organ_key}_model.{fmt}"
                    success = organ_model.export_mesh(file_path, fmt)
                    if success:
                        print(f"    Exported {organ_key} {fmt.upper()}: {file_path}")
                    else:
                        print(f"    Failed to export {organ_key} {fmt.upper()}")
        
        print(f"  Successfully generated {len(organ_3d_models)} 3D models")
        return organ_3d_models

    # Other methods remain unchanged (statistical analysis, report generation, visualization, etc.)
    def _calculate_statistics(self, fused_mask, spacing):
        """Calculate detailed statistics"""
        print("  Step 6: Calculating statistics...")
        
        voxel_volume = np.prod(spacing)
        
        stats = {
            "analysis_time": datetime.now().isoformat(),
            "spacing_mm": list(spacing),
            "voxel_volume_mm3": float(voxel_volume),
            "classes": {}
        }
        
        for label, name in self.class_names.items():
            if label == 0:
                continue
                
            mask = (fused_mask == label)
            voxel_count = np.sum(mask)
            volume_mm3 = voxel_count * voxel_volume
            volume_ml = volume_mm3 / 1000
            
            # Connected component analysis
            labeled, num_components = measure.label(mask, return_num=True)
            component_volumes = []
            
            if num_components > 0:
                for i in range(1, num_components + 1):
                    comp_mask = (labeled == i)
                    comp_volume = np.sum(comp_mask) * voxel_volume / 1000
                    component_volumes.append(float(comp_volume))
            
            stats["classes"][name] = {
                "label": int(label),
                "voxel_count": int(voxel_count),
                "volume_mm3": float(volume_mm3),
                "volume_ml": float(volume_ml),
                "num_components": int(num_components),
                "component_volumes_ml": component_volumes,
                "largest_component_ml": float(max(component_volumes)) if component_volumes else 0.0,
            }
        
        # Print main statistics
        for name, info in stats["classes"].items():
            if info["volume_ml"] > 0:
                print(f"    {name}: {info['volume_ml']:.1f} mL ({info['num_components']} regions)")
        
        return stats
    
    def _generate_diagnosis_report(self, stats):
        """Generate intelligent diagnostic report"""
        print("  Step 7: Generating diagnostic report...")
        
        report = {
            "analysis_time": datetime.now().isoformat(),
            "summary": {},
            "findings": [],
            "recommendations": [],
            "technical_notes": [],
            "risk_assessment": "Low Risk"
        }
        
        liver_info = stats["classes"].get("Liver", {})
        tumor_info = stats["classes"].get("Hepatic Tumor", {})
        vessel_info = stats["classes"].get("Hepatic Vessel", {})
        
        # Liver analysis
        if liver_info.get("volume_ml", 0) > 0:
            liver_vol = liver_info["volume_ml"]
            report["summary"]["liver_volume"] = f"{liver_vol:.1f} mL"
            
            # Liver volume assessment
            min_normal, max_normal = self.diagnosis_criteria["liver_volume_normal"]
            if liver_vol < min_normal:
                report["findings"].append("Liver volume is small, recommend evaluation for liver atrophy or cirrhosis")
                report["risk_assessment"] = "Moderate Risk"
            elif liver_vol > max_normal:
                report["findings"].append("Liver volume is large, recommend evaluation for hepatomegaly")
                report["risk_assessment"] = "Moderate Risk"
            else:
                report["findings"].append("Liver volume within normal range")
        else:
            report["findings"].append("Failed to successfully segment liver, please check image quality")
            report["risk_assessment"] = "Cannot Assess"
            return report
        
        # Vessel analysis
        if vessel_info.get("volume_ml", 0) > 0:
            vessel_vol = vessel_info["volume_ml"]
            vessel_count = vessel_info["num_components"]
            report["summary"]["vessel_count"] = vessel_count
            report["summary"]["vessel_volume"] = f"{vessel_vol:.1f} mL"
            report["findings"].append("Hepatic vessels appear normal")
        else:
            report["findings"].append("Vascular structures not clearly visualized")
        
        # Tumor analysis
        if tumor_info.get("volume_ml", 0) > 0:
            tumor_vol = tumor_info["volume_ml"]
            tumor_count = tumor_info["num_components"]
            largest_tumor = tumor_info["largest_component_ml"]
            
            report["summary"]["tumor_count"] = tumor_count
            report["summary"]["tumor_volume"] = f"{tumor_vol:.1f} mL"
            
            if tumor_count == 1:
                if largest_tumor < 2.0:
                    report["findings"].append("Single small hepatic lesion detected, recommend further imaging evaluation")
                    report["risk_assessment"] = "Low Risk"
                else:
                    report["findings"].append("Single hepatic lesion detected, recommend multiphase contrast-enhanced CT or MRI evaluation")
                    report["risk_assessment"] = "Moderate Risk"
            else:
                report["findings"].append(f"Multiple hepatic lesions detected ({tumor_count}), recommend evaluation for metastatic disease")
                report["risk_assessment"] = "Moderate Risk"
        else:
            report["findings"].append("No significant hepatic space-occupying lesions detected")
        
        # Technical notes
        report["technical_notes"].extend([
            "Based on deep learning AI algorithm automatic segmentation results",
            "3D models generated using Trimesh library and Marching Cubes algorithm",
            "Results are for clinical reference only and cannot replace professional physician diagnosis", 
            "Recommend comprehensive judgment combined with clinical symptoms, laboratory tests, and other imaging examinations",
            "3D models generated for surgical planning and educational demonstrations"
        ])
        
        return report
    
    def _create_color_visualization(self, fused_mask, meta_dict, output_dir):
        """Create colored visualization images and sequence frames"""
        print("  Step 8: Creating colored visualization...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sequence frames directory
        sequence_dir = output_dir / "visualization_sequence"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        # Select best display slice
        total_slices = fused_mask.shape[2]
        best_slice = total_slices // 2
        
        # Create sequence frame visualization
        saved_count = 0
        for z in range(total_slices):
            current_slice = fused_mask[:, :, z]
            
            # Check if current slice has content
            if np.any(current_slice > 0):
                # Create RGB image
                height, width = current_slice.shape
                overlay_rgb = np.zeros((height, width, 4))  # RGBA
                
                # Add color for each class
                for label in [1, 2, 3]:  # Liver, vessel, tumor
                    if label in self.class_colors and np.any(current_slice == label):
                        mask = (current_slice == label)
                        color = self.class_colors[label]
                        overlay_rgb[mask] = color
                
                # Save image
                plt.figure(figsize=(10, 8))
                plt.imshow(overlay_rgb)
                plt.title(f"Slice {z+1}/{total_slices}")
                plt.axis('off')
                
                frame_path = sequence_dir / f"frame_{z+1:04d}.png"
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                saved_count += 1
        
        print(f"    Sequence frames complete: saved {saved_count}/{total_slices} slices with content")
        
        # Create fused visualization
        current_slice = fused_mask[:, :, best_slice]
        height, width = current_slice.shape
        overlay_rgb = np.zeros((height, width, 4))  # RGBA
        
        # Statistics of current slice content
        slice_content = {}
        for label in [1, 2, 3]:  # Liver, vessel, tumor
            count = np.sum(current_slice == label)
            if count > 0:
                slice_content[label] = count
                # Add color
                mask = (current_slice == label)
                color = self.class_colors[label]
                overlay_rgb[mask] = color
        
        # Save fused visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(overlay_rgb)
        plt.title(f"Liver Structure Fused Visualization - Slice {best_slice+1}/{total_slices}")
        plt.axis('off')
        
        overlay_path = output_dir / "liver_overlay_visualization.png"
        plt.savefig(overlay_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved overlay visualization: {overlay_path}")
    
    def _save_mask_as_nifti(self, mask_array, meta_dict, output_path):
        """Save mask as NIfTI format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img = sitk.GetImageFromArray(mask_array.astype(np.uint8))
        img.SetOrigin(meta_dict.get("origin", [0, 0, 0]))
        img.SetSpacing(meta_dict.get("spacing", [1, 1, 1]))
        img.SetDirection(meta_dict.get("direction", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
        sitk.WriteImage(img, str(output_path))
    
    def _save_readable_report(self, report, output_path):
        """Save human-readable report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Liver AI Intelligent Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Time: {report['analysis_time']}\n")
            f.write(f"Risk Assessment: {report['risk_assessment']}\n\n")
            
            f.write("Main Findings:\n")
            f.write("-" * 40 + "\n")
            for finding in report['findings']:
                f.write(f"• {finding}\n")
            f.write("\n")
            
            if report['summary']:
                f.write("Quantitative Metrics:\n")
                f.write("-" * 40 + "\n")
                for key, value in report['summary'].items():
                    f.write(f"• {key}: {value}\n")
                f.write("\n")
            
            f.write("Recommendations:\n")
            f.write("-" * 40 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
            f.write("\n")
            
            f.write("Technical Notes:\n")
            f.write("-" * 40 + "\n")
            for note in report['technical_notes']:
                f.write(f"• {note}\n")
    
    def analyze_with_3d_models(self, ct_path, output_dir="./results"):
        """Complete analysis pipeline (including 3D model generation, based on Trimesh)"""
        print(f"\n{'='*80}")
        print(f"Starting Liver AI Intelligent Analysis (Trimesh-based 3D reconstruction): {ct_path}")
        print(f"{'='*80}")
        start_time = time.time()
        
        try:
            # 1. Preprocessing
            image_tensor, meta_dict = self._preprocess_ct(ct_path)
            
            # 2. Liver segmentation (must succeed)
            liver_mask = self._predict_liver(image_tensor)
            
            # 3. Get liver ROI
            roi_slice, roi_info = self._get_liver_roi(liver_mask)
            if roi_slice is None:
                raise RuntimeError("No liver detected, please check image quality and contrast")
            
            # 4. Vessel segmentation (must execute)
            vessel_mask = self._predict_vessel(image_tensor, liver_mask, roi_slice)
                
            # 5. Tumor segmentation (optional)
            tumor_mask = self._predict_tumor(image_tensor, liver_mask, roi_slice)
            
            # 6. Fuse results - get separate masks for 3D reconstruction
            fused_mask, individual_masks = self._fuse_masks(liver_mask, vessel_mask, tumor_mask)
            
            # 7. Generate 3D models (important step)
            organ_3d_models = self._generate_3d_models(individual_masks, meta_dict, output_dir)
            
            # 8. Statistical analysis
            spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
            stats = self._calculate_statistics(fused_mask, spacing)
            
            # 9. Generate intelligent diagnostic report
            report = self._generate_diagnosis_report(stats)
            
            # 10. Create colored visualization
            self._create_color_visualization(fused_mask, meta_dict, Path(output_dir))
            
            # 11. Save results
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save segmentation results
            seg_path = output_dir / "liver_segmentation.nii.gz"
            self._save_mask_as_nifti(fused_mask, meta_dict, seg_path)
            
            # Save individual masks for each class
            for label, name in self.class_names.items():
                if label > 0 and name in stats["classes"] and stats["classes"][name]["volume_ml"] > 0:
                    single_mask = (fused_mask == label).astype(np.uint8)
                    single_path = output_dir / f"{name.replace('Hepatic ', '').lower()}_mask.nii.gz"
                    self._save_mask_as_nifti(single_mask, meta_dict, single_path)
            
            # Save statistics
            stats_path = output_dir / "statistics.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            # Save diagnostic report
            report_path = output_dir / "diagnosis_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            # Save human-readable report
            readable_report_path = output_dir / "Intelligent_Diagnosis_Report.txt"
            self._save_readable_report(report, readable_report_path)
            
            # Save 3D model information
            if organ_3d_models:
                models_info = {}
                for organ_name, model in organ_3d_models.items():
                    models_info[organ_name] = {
                        "statistics": model.get_statistics(),
                        "file_paths": model.file_paths,
                        "color": model.color,
                        "visible": model.visible
                    }
                
                models_info_path = output_dir / "3d_models_info.json"
                with open(models_info_path, 'w', encoding='utf-8') as f:
                    json.dump(models_info, f, ensure_ascii=False, indent=2)
            
            analysis_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"Analysis Complete!")
            print(f"Total Time: {analysis_time:.1f} seconds")
            print(f"Risk Assessment: {report['risk_assessment']}")
            print(f"Main Findings: {len(report['findings'])} items")
            print(f"Recommendations: {len(report['recommendations'])} items")
            if organ_3d_models:
                print(f"3D Models: {len(organ_3d_models)} organ models generated")
                for organ_name in organ_3d_models.keys():
                    print(f"   • {organ_name}")
            print(f"Results saved in: {output_dir}")
            print(f"{'='*80}\n")
            
            return {
                "success": True,
                "analysis_time": analysis_time,
                "files": {
                    "segmentation": str(seg_path),
                    "statistics": str(stats_path),
                    "report": str(report_path),
                    "readable_report": str(readable_report_path)
                },
                "summary": report["summary"],
                "risk_assessment": report["risk_assessment"],
                "roi_info": roi_info,
                "findings_count": len(report['findings']),
                "recommendations_count": len(report['recommendations']),
                "organ_3d_models": organ_3d_models,
                "has_3d_models": len(organ_3d_models) > 0
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"Analysis Failed: {e}")
            print(f"Time Before Failure: {error_time:.1f} seconds")
            print(f"{'='*80}\n")
            
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "analysis_time": error_time,
                "organ_3d_models": {},
                "has_3d_models": False
            }

# Usage example and test function
def test_pipeline_with_3d():
    """Test pipeline functionality (using Trimesh)"""
    # Configure model paths
    model_paths = {
        "liver": "./liver_model/best_liver_model.pth",    # Liver model
        "vessel": "./vessel_model/best_vessel_model.pth", # Vessel model  
        "tumor": "./tumor_model/best_tumor_model.pth"     # Tumor model
    }
    
    print("Checking model files...")
    for task, path in model_paths.items():
        exists = os.path.exists(path)
        status = "Exists" if exists else "Not Found"
        print(f"  {task}: {path} - {status}")
    
    print(f"Checking 3D reconstruction dependencies...")
    print(f"  trimesh: {'Installed' if HAS_TRIMESH else 'Not Installed'}")
    
    try:
        # Create analysis pipeline
        pipeline = LiverAnalysisPipelineWith3D(model_paths)
        
        # Test CT files
        test_ct_files = [
            "./test_data/case_001.nii.gz",
            "./test_data/case_004.nii.gz",
        ]
        
        ct_file = None
        for test_file in test_ct_files:
            if os.path.exists(test_file):
                ct_file = test_file
                break
        
        if ct_file:
            print(f"\nFound test file: {ct_file}")
            result = pipeline.analyze_with_3d_models(ct_file, output_dir="./enhanced_results_with_3d_trimesh")
            
            if result["success"]:
                print("Test Successful!")
                print(f"Risk Assessment: {result['risk_assessment']}")
                print(f"Segmentation File: {result['files']['segmentation']}")
                print(f"Statistics File: {result['files']['statistics']}")
                print(f"Diagnosis Report: {result['files']['report']}")
                print(f"Readable Report: {result['files']['readable_report']}")
                
                if result["has_3d_models"]:
                    print(f"3D Models Generated:")
                    for organ_name, model in result["organ_3d_models"].items():
                        stats = model.get_statistics()
                        print(f"   • {organ_name}: {stats.get('vertices', 0)} vertices, {stats.get('faces', 0)} faces")
                else:
                    print("No 3D models generated (possibly due to Trimesh not installed or insufficient data)")
            else:
                print(f"Test Failed: {result['error']}")
        else:
            print("Test CT file not found")
            print("Please place test file at one of the following locations:")
            for test_file in test_ct_files:
                print(f"  {test_file}")
                
    except Exception as e:
        print(f"Initialization Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline_with_3d()
