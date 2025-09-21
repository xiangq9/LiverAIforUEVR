#!/usr/bin/env python3
"""
Liver AI Backend Server for UE Plugin (Fixed Version)
Provides HTTP API interface, reimplements AI analysis functionality
"""

import os
import sys
import json
import base64
import asyncio
import threading
import traceback
import tempfile
from datetime import datetime
from pathlib import Path

# Web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Check AI modules
try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import nibabel as nib
    from scipy import ndimage
    from skimage import measure
    HAS_BASIC_MODULES = True
except ImportError as e:
    print(f"Warning: Basic modules not available: {e}")
    HAS_BASIC_MODULES = False

try:
    import monai
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped
    )
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SegResNet, UNet
    from monai.networks.layers import Norm
    HAS_MONAI = True
except ImportError:
    print("Warning: MONAI not available")
    HAS_MONAI = False

try:
    import trimesh
    from trimesh import Trimesh
    HAS_TRIMESH = True
except ImportError:
    print("Warning: Trimesh not available")
    HAS_TRIMESH = False

# API data models
class AnalysisRequest(BaseModel):
    ct_file_path: str
    liver_model_path: str = ""
    vessel_model_path: str = ""
    tumor_model_path: str = ""
    request_id: str = ""

class AnalysisProgress(BaseModel):
    request_id: str
    status: str
    progress: float
    timestamp: str

class OrganStats(BaseModel):
    organ_name: str
    volume_ml: float
    voxel_count: int
    num_components: int
    largest_component: float = 0.0

class MeshData(BaseModel):
    vertices: list
    faces: list
    organ_name: str
    color: list

class AnalysisResult(BaseModel):
    success: bool
    request_id: str
    error_message: str = ""
    organ_stats: list[OrganStats] = []
    diagnostic_report: str = ""
    image_data_base64: str = ""
    segmentation_data_base64: str = ""
    image_dimensions: list[int] = []
    mesh_data: list[MeshData] = []
    timestamp: str

# AI model definitions (extracted from original code)
if HAS_MONAI:
    class OptimizedLiverSegClassifier(nn.Module):
        """Liver Segmentation Classification Model"""
        def __init__(self, out_channels=2, task_type="liver"):
            super().__init__()
            
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
            
            self.task_type = task_type
            self.inference_mode = False
        
        def forward(self, x):
            seg_output = self.segmentation_model(x)
            
            if hasattr(self, 'inference_mode') and self.inference_mode:
                return seg_output
                
            if self.classification_head is not None:
                cls_output = self.classification_head(seg_output)
                return seg_output, cls_output
            else:
                return seg_output
        
        def inference(self, x):
            self.inference_mode = True
            seg_output = self.segmentation_model(x)
            self.inference_mode = False
            return seg_output

    class FocusedTumorModel(nn.Module):
        """Tumor Segmentation Model"""
        def __init__(self, out_channels=2):
            super().__init__()
            
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
            
            self.tumor_attention = nn.Sequential(
                nn.Conv3d(out_channels, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 1, 1),
                nn.Sigmoid()
            )
            
            self.inference_mode = False
        
        def forward(self, x):
            seg_output = self.segmentation_model(x)
            attention_weights = self.tumor_attention(seg_output)
            enhanced_output = seg_output.clone()
            enhanced_output[:, 1:2] = seg_output[:, 1:2] * (1 + 2.0 * attention_weights)
            return enhanced_output
        
        def inference(self, x):
            self.inference_mode = True
            result = self.forward(x)
            self.inference_mode = False
            return result

# 3D reconstruction class
class TrimeshMesh3DGenerator:
    """3D Reconstruction Generator"""
    def __init__(self):
        self.class_colors = {
            "liver": [0.0, 0.8, 0.0],
            "vessel": [0.8, 0.0, 0.0], 
            "tumor": [0.8, 0.8, 0.0]
        }
    
    def mask_to_mesh_marching_cubes(self, mask, spacing, origin=[0, 0, 0], step_size=1):
        """Generate mesh using Marching Cubes algorithm"""
        if not HAS_TRIMESH or not HAS_BASIC_MODULES:
            return None
            
        try:
            from skimage.measure import marching_cubes
            
            smoothed_mask = ndimage.gaussian_filter(mask.astype(float), sigma=0.5)
            verts, faces, normals, values = marching_cubes(
                smoothed_mask, 
                level=0.5,
                step_size=step_size,
                allow_degenerate=False
            )
            
            verts[:, 0] *= spacing[0]
            verts[:, 1] *= spacing[1]
            verts[:, 2] *= spacing[2]
            verts += np.array(origin)
            
            mesh = Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            
            if not mesh.is_winding_consistent:
                mesh.fix_normals()
            
            return mesh
            
        except Exception as e:
            print(f"Marching Cubes error: {e}")
            return None
    
    def generate_organ_3d_model(self, mask, organ_name, spacing, origin=[0, 0, 0]):
        """Generate organ 3D model"""
        voxel_count = np.sum(mask > 0)
        if voxel_count < 100:
            return None
        
        mesh = self.mask_to_mesh_marching_cubes(mask, spacing, origin, step_size=2)
        if mesh is None or len(mesh.vertices) == 0:
            return None
                
        return {
            'mesh_data': mesh,
            'organ_name': organ_name,
            'color': self.class_colors.get(organ_name.lower(), [0.8, 0.8, 0.8])
        }

# Complete AI inference engine
class LiverAIInferenceEngine:
    """Liver AI Inference Engine"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_spacing = (1.5, 1.5, 2.0)
        self.hu_range = (-200, 250)
        self.roi_size = (96, 96, 96)
        self.class_names = {0: "Background", 1: "Liver", 2: "Vessel", 3: "Tumor"}
        print(f"Inference engine initialized on device: {self.device}")
    
    def load_models(self, model_paths):
        """Load AI models"""
        if not HAS_MONAI:
            raise ValueError("MONAI not available")
            
        models = {}
        
        try:
            if model_paths.get("liver") and os.path.exists(model_paths["liver"]):
                checkpoint = torch.load(model_paths["liver"], map_location=self.device, weights_only=False)
                liver_model = OptimizedLiverSegClassifier(out_channels=2, task_type="liver")
                liver_model.load_state_dict(checkpoint['model_state_dict'])
                liver_model.to(self.device)
                liver_model.eval()
                models["liver"] = liver_model
                print("Liver model loaded successfully")
            
            if model_paths.get("vessel") and os.path.exists(model_paths["vessel"]):
                checkpoint = torch.load(model_paths["vessel"], map_location=self.device, weights_only=False)
                vessel_model = OptimizedLiverSegClassifier(out_channels=2, task_type="vessel")
                vessel_model.load_state_dict(checkpoint['model_state_dict'])
                vessel_model.to(self.device)
                vessel_model.eval()
                models["vessel"] = vessel_model
                print("Vessel model loaded successfully")
            
            if model_paths.get("tumor") and os.path.exists(model_paths["tumor"]):
                checkpoint = torch.load(model_paths["tumor"], map_location=self.device, weights_only=False)
                tumor_model = FocusedTumorModel(out_channels=2)
                tumor_model.load_state_dict(checkpoint['model_state_dict'])
                tumor_model.to(self.device)
                tumor_model.eval()
                models["tumor"] = tumor_model
                print("Tumor model loaded successfully")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
            
        return models
    
    def preprocess_ct(self, ct_path):
        """Preprocess CT image"""
        if not HAS_MONAI:
            raise ValueError("MONAI not available")
            
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
        
        # Get metadata
        try:
            if hasattr(processed_data["image"], 'meta') and processed_data["image"].meta is not None:
                meta_dict = processed_data["image"].meta
            else:
                meta_dict = {
                    "spacing": list(self.target_spacing),
                    "origin": [0, 0, 0],
                    "spatial_shape": list(image_tensor.shape[2:])
                }
        except:
            meta_dict = {
                "spacing": list(self.target_spacing),
                "origin": [0, 0, 0],
                "spatial_shape": list(image_tensor.shape[2:])
            }
        
        return image_tensor, meta_dict
    
    def predict_organ(self, model, image_tensor, mask_constraint=None, roi_slice=None):
        """Generic organ prediction function"""
        if not HAS_MONAI:
            raise ValueError("MONAI not available")
            
        input_tensor = image_tensor
        
        # If ROI is specified, only process the ROI region
        if roi_slice is not None:
            input_tensor = image_tensor[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                roi_shape = input_tensor.shape[2:]
                patch_size = (32, 32, 32) if min(roi_shape) < 64 else self.roi_size
                sw_batch_size = 2 if min(roi_shape) < 64 else 4
                
                if hasattr(model, 'inference'):
                    output = sliding_window_inference(
                        input_tensor, patch_size, sw_batch_size,
                        model.inference, overlap=0.5, mode="gaussian"
                    )
                else:
                    def seg_only_inference(x):
                        result = model(x)
                        return result[0] if isinstance(result, tuple) else result
                    
                    output = sliding_window_inference(
                        input_tensor, patch_size, sw_batch_size,
                        seg_only_inference, overlap=0.5, mode="gaussian"
                    )
        
        mask = torch.argmax(output, dim=1, keepdim=True)
        
        # Apply constraint mask (e.g., liver region constraint)
        if mask_constraint is not None:
            if roi_slice is not None:
                constraint = mask_constraint[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
            else:
                constraint = mask_constraint
            mask = mask * constraint
        
        # If processing ROI, need to map back to full size
        if roi_slice is not None:
            full_mask = torch.zeros_like(image_tensor[:, :1])  # Single channel
            full_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]] = mask
            return full_mask
        
        return mask
    
    def get_liver_roi(self, liver_mask, margin=20):
        """Get liver ROI"""
        liver_np = liver_mask.squeeze().cpu().numpy()
        coords = np.where(liver_np > 0)
        if len(coords[0]) == 0:
            return None, None
            
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
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
        
        return roi_slice, roi_info
    
    def run_analysis(self, ct_path, model_paths, progress_callback=None):
        """Run complete analysis"""
        try:
            # Check dependencies
            if not HAS_MONAI or not HAS_BASIC_MODULES:
                raise ValueError("Required AI modules not available")
            
            def update_progress(msg, pct):
                if progress_callback:
                    progress_callback(msg, pct)
                print(f"Progress: {pct}% - {msg}")
            
            update_progress("Loading models...", 10)
            models = self.load_models(model_paths)
            
            if not models.get("liver"):
                raise ValueError("Liver model is required")
            
            update_progress("Preprocessing CT...", 20)
            image_tensor, meta_dict = self.preprocess_ct(ct_path)
            
            update_progress("Liver segmentation...", 40)
            liver_mask = self.predict_organ(models["liver"], image_tensor)
            
            # Get liver ROI
            roi_slice, roi_info = self.get_liver_roi(liver_mask)
            if roi_slice is None:
                raise ValueError("No liver detected in the image")
            
            # Vessel segmentation
            vessel_mask = torch.zeros_like(liver_mask)
            if models.get("vessel"):
                update_progress("Vessel segmentation...", 60)
                vessel_mask = self.predict_organ(
                    models["vessel"], image_tensor, 
                    mask_constraint=liver_mask, roi_slice=roi_slice
                )
            
            # Tumor segmentation
            tumor_mask = torch.zeros_like(liver_mask)
            if models.get("tumor"):
                update_progress("Tumor segmentation...", 70)
                tumor_mask = self.predict_organ(
                    models["tumor"], image_tensor,
                    mask_constraint=liver_mask, roi_slice=roi_slice
                )
            
            update_progress("Processing results...", 80)
            
            # Convert to numpy and merge
            liver_np = liver_mask.squeeze().cpu().numpy().astype(np.uint8)
            vessel_np = vessel_mask.squeeze().cpu().numpy().astype(np.uint8)
            tumor_np = tumor_mask.squeeze().cpu().numpy().astype(np.uint8)
            
            # Create fused mask
            fused_mask = np.zeros_like(liver_np, dtype=np.uint8)
            fused_mask[liver_np == 1] = 1
            fused_mask[vessel_np == 1] = 2
            fused_mask[tumor_np == 1] = 3
            
            individual_masks = {
                "liver": liver_np,
                "vessel": vessel_np,
                "tumor": tumor_np
            }
            
            update_progress("Generating 3D models...", 90)
            organ_models = self.generate_3d_models(individual_masks, meta_dict)
            
            update_progress("Complete!", 100)
            
            return {
                "success": True,
                "image_data": image_tensor.squeeze().cpu().numpy(),
                "fused_mask": fused_mask,
                "individual_masks": individual_masks,
                "organ_models": organ_models,
                "meta_dict": meta_dict
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            if progress_callback:
                progress_callback(error_msg, 0)
            raise
    
    def generate_3d_models(self, individual_masks, meta_dict):
        """Generate 3D models"""
        if not HAS_TRIMESH:
            print("Warning: Trimesh not available, skipping 3D model generation")
            return {}
        
        spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
        origin = meta_dict.get("origin", [0, 0, 0])
        
        organ_3d_models = {}
        generator = TrimeshMesh3DGenerator()
        
        for organ_key, mask in individual_masks.items():
            if np.sum(mask > 0) < 50:
                continue
                
            organ_model = generator.generate_organ_3d_model(
                mask=mask,
                organ_name=organ_key,
                spacing=spacing,
                origin=origin
            )
            
            if organ_model:
                organ_3d_models[organ_key] = organ_model
        
        return organ_3d_models

# Global state management
analysis_status = {}
analysis_results = {}
inference_engine = LiverAIInferenceEngine()

# FastAPI application
app = FastAPI(
    title="Liver AI Backend",
    description="Backend service for UE Liver AI Plugin",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def status_callback(request_id, status, progress):
    """Status callback function"""
    analysis_status[request_id] = AnalysisProgress(
        request_id=request_id,
        status=status,
        progress=progress,
        timestamp=datetime.now().isoformat()
    )

def convert_to_api_result(request_id, result):
    """Convert analysis result to API format"""
    try:
        # Calculate organ statistics
        organ_stats = []
        individual_masks = result.get("individual_masks", {})
        meta_dict = result.get("meta_dict", {})
        spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
        voxel_volume = np.prod(spacing)
        
        for organ_name, mask in individual_masks.items():
            if np.sum(mask > 0) > 0:
                voxel_count = np.sum(mask > 0)
                volume_ml = voxel_count * voxel_volume / 1000
                
                # Connected component analysis
                labeled, num_components = ndimage.label(mask)
                
                largest_component = 0.0
                if num_components > 0:
                    component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
                    largest_component = float(np.max(component_sizes)) * voxel_volume / 1000
                
                organ_stats.append(OrganStats(
                    organ_name=organ_name,
                    volume_ml=float(volume_ml),
                    voxel_count=int(voxel_count),
                    num_components=int(num_components),
                    largest_component=largest_component
                ))
        
        # Encode image data
        image_data = result["image_data"].astype(np.float32)
        fused_mask = result["fused_mask"].astype(np.uint8)
        
        image_data_bytes = image_data.tobytes()
        segmentation_data_bytes = fused_mask.tobytes()
        
        image_data_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
        segmentation_data_base64 = base64.b64encode(segmentation_data_bytes).decode('utf-8')
        
        # Convert 3D mesh data
        mesh_data = []
        organ_models = result.get("organ_models", {})
        for organ_name, model in organ_models.items():
            if model and model.get('mesh_data'):
                mesh = model['mesh_data']
                mesh_data.append(MeshData(
                    vertices=mesh.vertices.tolist(),
                    faces=mesh.faces.tolist(),
                    organ_name=organ_name,
                    color=model.get('color', [0.8, 0.8, 0.8])
                ))
        
        # Generate diagnostic report
        diagnostic_report = generate_diagnostic_report(organ_stats)
        
        return AnalysisResult(
            success=True,
            request_id=request_id,
            organ_stats=organ_stats,
            diagnostic_report=diagnostic_report,
            image_data_base64=image_data_base64,
            segmentation_data_base64=segmentation_data_base64,
            image_dimensions=list(image_data.shape),
            mesh_data=mesh_data,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return AnalysisResult(
            success=False,
            request_id=request_id,
            error_message=f"Result conversion failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

def generate_diagnostic_report(organ_stats):
    """Generate diagnostic report"""
    lines = []
    lines.append("=== AI LIVER ANALYSIS REPORT ===")
    lines.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    for stats in organ_stats:
        lines.append(f"{stats.organ_name.upper()}:")
        lines.append(f"  Volume: {stats.volume_ml:.2f} mL")
        lines.append(f"  Voxel Count: {stats.voxel_count:,}")
        lines.append(f"  Components: {stats.num_components}")
        if stats.largest_component > 0:
            lines.append(f"  Largest Component: {stats.largest_component:.2f} mL")
        lines.append("")
    
    return "\n".join(lines)

# API endpoints
@app.post("/api/analyze", response_model=dict)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start AI analysis"""
    try:
        if not HAS_MONAI or not HAS_BASIC_MODULES:
            raise HTTPException(status_code=500, detail="AI modules not available")
        
        # Validate file exists
        if not os.path.exists(request.ct_file_path):
            raise HTTPException(status_code=400, detail="CT file not found")
        
        request_id = request.request_id or f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize status
        analysis_status[request_id] = AnalysisProgress(
            request_id=request_id,
            status="Starting analysis...",
            progress=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Start background analysis task
        def run_analysis():
            try:
                def progress_callback(status, progress):
                    status_callback(request_id, status, progress)
                
                result = inference_engine.run_analysis(
                    ct_path=request.ct_file_path,
                    model_paths={
                        "liver": request.liver_model_path,
                        "vessel": request.vessel_model_path,
                        "tumor": request.tumor_model_path
                    },
                    progress_callback=progress_callback
                )
                
                # Convert to API response format
                api_result = convert_to_api_result(request_id, result)
                analysis_results[request_id] = api_result
                
            except Exception as e:
                traceback.print_exc()
                analysis_results[request_id] = AnalysisResult(
                    success=False,
                    request_id=request_id,
                    error_message=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        background_tasks.add_task(run_analysis)
        
        return {"request_id": request_id, "status": "Analysis started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{request_id}", response_model=AnalysisProgress)
async def get_analysis_status(request_id: str):
    """Get analysis status"""
    if request_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return analysis_status[request_id]

@app.get("/api/result/{request_id}", response_model=AnalysisResult)
async def get_analysis_result(request_id: str):
    """Get analysis result"""
    if request_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return analysis_results[request_id]

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "ai_modules": HAS_MONAI and HAS_BASIC_MODULES,
        "basic_modules": HAS_BASIC_MODULES,
        "monai": HAS_MONAI,
        "trimesh": HAS_TRIMESH,
        "device": inference_engine.device,
        "timestamp": datetime.now().isoformat()
    }

# Start server
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Liver AI Backend Server...")
    print(f"Basic Modules (numpy, torch, etc.): {HAS_BASIC_MODULES}")
    print(f"MONAI Available: {HAS_MONAI}")
    print(f"Trimesh Available: {HAS_TRIMESH}")
    print(f"Device: {inference_engine.device if HAS_BASIC_MODULES else 'N/A'}")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8888,
        log_level="info"
    )
