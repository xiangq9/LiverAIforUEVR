#!/usr/bin/env python3
"""
Integrated Liver AI Analysis Interface
Integrates 2D segmentation view, 3D model viewer, and AI inference functionality
Three-window layout: left for original image, center for segmentation results, right for 3D models
"""

import os
import sys
import time
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, 
    QSplitter, QGroupBox, QFormLayout, QTextEdit, QTabWidget,
    QComboBox, QCheckBox, QSlider, QSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# VTK imports for 3D visualization
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("Warning: VTK not found. 3D visualization will be disabled.")

# Trimesh imports for 3D reconstruction
try:
    import trimesh
    from trimesh import Trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: Trimesh not found. 3D reconstruction will be disabled.")

# Medical imaging imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import SimpleITK as sitk
    import monai
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, EnsureTyped
    )
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SegResNet, UNet
    from monai.networks.layers import Norm
    from scipy import ndimage
    from skimage import measure, morphology
    HAS_AI_MODULES = True
except ImportError as e:
    HAS_AI_MODULES = False
    print(f"Warning: AI modules not found: {e}")


class NiftiSliceCanvas(FigureCanvas):
    """Enhanced NIfTI slice display canvas with multi-class overlay and synchronization support"""
    
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#2b2b2b')  # Dark background
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e1e')  # Darker axis background
        super(NiftiSliceCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.img_data = None
        self.overlay_data = None
        self.current_slice = 0
        self.total_slices = 0
        self.view_plane = 'axial'
        
        # Synchronization callback function
        self.sync_callback = None
        
        # Class color mapping
        self.class_colors = {
            1: [0.0, 1.0, 0.0, 0.6],  # Liver: green
            2: [1.0, 0.0, 0.0, 0.8],  # Vessel: red
            3: [1.0, 1.0, 0.0, 0.9]   # Tumor: yellow
        }
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
    
    def wheelEvent(self, event):
        """Mouse wheel slice navigation"""
        if self.img_data is None:
            return
            
        delta = event.angleDelta().y()
        new_slice = self.current_slice
        
        if delta > 0 and self.current_slice > 0:
            new_slice = self.current_slice - 1
        elif delta < 0 and self.current_slice < self.total_slices - 1:
            new_slice = self.current_slice + 1
        else:
            return
            
        # Use synchronization callback
        if self.sync_callback:
            self.sync_callback(new_slice)
        else:
            # Fallback: update this canvas directly
            self.current_slice = new_slice
            if self.overlay_data is not None:
                self.update_overlay_display()
            else:
                self.update_display()
    
    def set_data(self, img_data):
        """Set original image data"""
        self.img_data = img_data
        if img_data is not None:
            self.update_slice_range()
            self.current_slice = self.total_slices // 2
            self.update_display()
    
    def set_overlay(self, img_data, overlay_data):
        """Set image and overlay data"""
        self.img_data = img_data
        self.overlay_data = overlay_data
        
        if img_data is not None:
            self.update_slice_range()
            self.current_slice = self.total_slices // 2
            self.update_overlay_display()
    
    def update_slice_range(self):
        """Update slice count"""
        if self.view_plane == 'axial':
            self.total_slices = self.img_data.shape[2]
        elif self.view_plane == 'coronal':
            self.total_slices = self.img_data.shape[1]
        elif self.view_plane == 'sagittal':
            self.total_slices = self.img_data.shape[0]
    
    def change_view_plane(self, plane):
        """Change viewing plane"""
        if plane in ['axial', 'coronal', 'sagittal']:
            self.view_plane = plane
            if self.img_data is not None:
                self.update_slice_range()
                self.current_slice = min(self.current_slice, self.total_slices - 1)
                if self.overlay_data is not None:
                    self.update_overlay_display()
                else:
                    self.update_display()
    
    def update_display(self):
        """Update display of current slice"""
        if self.img_data is None:
            return
            
        self.axes.clear()
        self.axes.set_facecolor('#1e1e1e')  # Ensure dark background
        
        try:
            if self.view_plane == 'axial':
                slice_data = self.img_data[:, :, self.current_slice]
            elif self.view_plane == 'coronal':
                slice_data = self.img_data[:, self.current_slice, :]
            elif self.view_plane == 'sagittal':
                slice_data = self.img_data[self.current_slice, :, :]
            
            vmin = np.percentile(slice_data, 1)
            vmax = np.percentile(slice_data, 99)
            
            self.axes.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
            self.axes.set_title(f'{self.view_plane.capitalize()} - Slice {self.current_slice + 1}/{self.total_slices}', 
                              color='#ffffff')  # White title text
            self.axes.axis('off')
            self.fig.tight_layout()
            self.draw()
            
        except Exception as e:
            print(f"Error in update_display: {e}")
    
    def update_overlay_display(self):
        """Update display with overlay"""
        if self.img_data is None or self.overlay_data is None:
            return
            
        self.axes.clear()
        self.axes.set_facecolor('#1e1e1e')  # Ensure dark background
        
        try:
            if self.view_plane == 'axial':
                slice_data = self.img_data[:, :, self.current_slice]
                overlay_slice = self.overlay_data[:, :, self.current_slice]
            elif self.view_plane == 'coronal':
                slice_data = self.img_data[:, self.current_slice, :]
                overlay_slice = self.overlay_data[:, self.current_slice, :]
            elif self.view_plane == 'sagittal':
                slice_data = self.img_data[self.current_slice, :, :]
                overlay_slice = self.overlay_data[self.current_slice, :, :]
            
            vmin = np.percentile(slice_data, 1)
            vmax = np.percentile(slice_data, 99)
            
            # Display base image
            self.axes.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Overlay segmentation results
            overlay_rgba = np.zeros((*overlay_slice.shape, 4))
            
            for class_id, color in self.class_colors.items():
                mask = (overlay_slice == class_id)
                if np.any(mask):
                    overlay_rgba[mask] = color
            
            self.axes.imshow(overlay_rgba)
            self.axes.set_title(f'{self.view_plane.capitalize()} - Slice {self.current_slice + 1}/{self.total_slices}',
                              color='#ffffff')  # White title text
            self.axes.axis('off')
            self.fig.tight_layout()
            self.draw()
            
        except Exception as e:
            print(f"Error in update_overlay_display: {e}")
    
    def next_slice(self):
        """Next slice"""
        if self.img_data is not None and self.current_slice < self.total_slices - 1:
            new_slice = self.current_slice + 1
            if self.sync_callback:
                self.sync_callback(new_slice)
            else:
                self.current_slice = new_slice
                if self.overlay_data is not None:
                    self.update_overlay_display()
                else:
                    self.update_display()
    
    def prev_slice(self):
        """Previous slice"""
        if self.img_data is not None and self.current_slice > 0:
            new_slice = self.current_slice - 1
            if self.sync_callback:
                self.sync_callback(new_slice)
            else:
                self.current_slice = new_slice
                if self.overlay_data is not None:
                    self.update_overlay_display()
                else:
                    self.update_display()


class Model3DViewer(QWidget):
    """3D Model Viewer"""
    
    def __init__(self, parent=None):
        super(Model3DViewer, self).__init__(parent)
        
        if not HAS_VTK:
            # If VTK is not available, show placeholder
            layout = QVBoxLayout(self)
            label = QLabel("3D Visualization Unavailable\n(VTK not installed)")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color: gray; font-size: 16px;")
            layout.addWidget(label)
            return
        
        # VTK components
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.2)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        # Set interaction style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.vtkWidget)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Currently displayed actors
        self.organ_actors = {}
        self.organ_visibility = {}
        
        # Initialize
        self.interactor.Initialize()
    
    def create_control_panel(self):
        """Create 3D control panel"""
        panel = QGroupBox("3D Controls")
        layout = QHBoxLayout(panel)
        
        # Organ visibility controls
        self.liver_checkbox = QCheckBox("Liver")
        self.liver_checkbox.setChecked(True)
        self.liver_checkbox.toggled.connect(lambda: self.toggle_organ_visibility('liver'))
        
        self.vessel_checkbox = QCheckBox("Vessel")
        self.vessel_checkbox.setChecked(True)
        self.vessel_checkbox.toggled.connect(lambda: self.toggle_organ_visibility('vessel'))
        
        self.tumor_checkbox = QCheckBox("Tumor")
        self.tumor_checkbox.setChecked(True)
        self.tumor_checkbox.toggled.connect(lambda: self.toggle_organ_visibility('tumor'))
        
        # Opacity control
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        
        layout.addWidget(QLabel("Show:"))
        layout.addWidget(self.liver_checkbox)
        layout.addWidget(self.vessel_checkbox)
        layout.addWidget(self.tumor_checkbox)
        layout.addWidget(QLabel("Opacity:"))
        layout.addWidget(self.opacity_slider)
        
        return panel
    
    def clear_scene(self):
        """Clear scene"""
        if not HAS_VTK:
            return
            
        for actor in self.organ_actors.values():
            self.renderer.RemoveActor(actor)
        
        self.organ_actors.clear()
        self.organ_visibility.clear()
        self.vtkWidget.GetRenderWindow().Render()
    
    def add_organ_mesh(self, organ_name, mesh_data, color=None):
        """Add organ mesh model"""
        if not HAS_VTK or not HAS_TRIMESH:
            return
            
        try:
            # Convert from Trimesh to VTK
            vertices = mesh_data.vertices
            faces = mesh_data.faces
            
            # Create VTK points
            points = vtk.vtkPoints()
            for vertex in vertices:
                points.InsertNextPoint(vertex[0], vertex[1], vertex[2])
            
            # Create VTK cells
            polys = vtk.vtkCellArray()
            for face in faces:
                poly = vtk.vtkPolygon()
                poly.GetPointIds().SetNumberOfIds(3)
                for i in range(3):
                    poly.GetPointIds().SetId(i, face[i])
                polys.InsertNextCell(poly)
            
            # Create polydata
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            
            # Compute normals
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set color
            if color is None:
                if organ_name == 'liver':
                    color = [0.0, 0.8, 0.0]  # Green
                elif organ_name == 'vessel':
                    color = [0.8, 0.0, 0.0]  # Red
                elif organ_name == 'tumor':
                    color = [0.8, 0.8, 0.0]  # Yellow
                else:
                    color = [0.7, 0.7, 0.7]  # Gray
            
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(0.8)
            
            # Add to renderer
            self.renderer.AddActor(actor)
            self.organ_actors[organ_name] = actor
            self.organ_visibility[organ_name] = True
            
            print(f"Added {organ_name} mesh: {len(vertices)} vertices, {len(faces)} faces")
            
        except Exception as e:
            print(f"Error adding {organ_name} mesh: {e}")
    
    def toggle_organ_visibility(self, organ_name):
        """Toggle organ visibility"""
        if organ_name in self.organ_actors:
            checkbox_map = {
                'liver': self.liver_checkbox,
                'vessel': self.vessel_checkbox,
                'tumor': self.tumor_checkbox
            }
            
            checkbox = checkbox_map.get(organ_name)
            if checkbox:
                visible = checkbox.isChecked()
                self.organ_actors[organ_name].SetVisibility(visible)
                self.organ_visibility[organ_name] = visible
                self.vtkWidget.GetRenderWindow().Render()
    
    def update_opacity(self, value):
        """Update opacity"""
        opacity = value / 100.0
        for actor in self.organ_actors.values():
            actor.GetProperty().SetOpacity(opacity)
        self.vtkWidget.GetRenderWindow().Render()
    
    def reset_camera(self):
        """Reset camera"""
        if HAS_VTK:
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
    
    def update_display(self, organ_models):
        """Update 3D display"""
        if not HAS_VTK or not HAS_TRIMESH:
            return
            
        self.clear_scene()
        
        for organ_name, model in organ_models.items():
            if hasattr(model, 'mesh_data') and model.mesh_data:
                self.add_organ_mesh(organ_name, model.mesh_data, model.color[:3] if model.color else None)
        
        self.reset_camera()


# AI model classes (simplified version, extracted from previous code)
if HAS_AI_MODULES:
    class OptimizedLiverSegClassifier(nn.Module):
        """Liver segmentation classification model"""
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
        """Tumor segmentation model"""
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

    # 3D reconstruction related classes
    class Organ3DModel:
        """Organ 3D model"""
        def __init__(self, organ_name, mesh_data=None, color=None):
            self.organ_name = organ_name
            self.mesh_data = mesh_data
            self.color = color or [0.8, 0.8, 0.8]
            self.visible = True
            self.file_paths = {}
            
        def set_mesh(self, mesh):
            self.mesh_data = mesh
            
        def set_color(self, color):
            self.color = color
    
    class TrimeshMask3DGenerator:
        """3D reconstruction generator"""
        def __init__(self):
            self.class_colors = {
                "liver": [0.0, 0.8, 0.0],
                "vessel": [0.8, 0.0, 0.0], 
                "tumor": [0.8, 0.8, 0.0]
            }
        
        def mask_to_mesh_marching_cubes(self, mask, spacing, origin=[0, 0, 0], step_size=1):
            """Generate mesh using Marching Cubes algorithm"""
            if not HAS_TRIMESH:
                return None
                
            try:
                from skimage.measure import marching_cubes
                from scipy import ndimage
                
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
                
            organ_model = Organ3DModel(
                organ_name=organ_name,
                mesh_data=mesh,
                color=self.class_colors.get(organ_name.lower(), [0.8, 0.8, 0.8])
            )
            
            return organ_model


class InferenceThread(QThread):
    """Inference thread"""
    progress_update = Signal(str)
    progress_value = Signal(int)
    inference_complete = Signal(bool, object)
    
    def __init__(self, parent=None):
        super(InferenceThread, self).__init__(parent)
        self.ct_path = None
        self.model_paths = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # AI related parameters
        self.target_spacing = (1.5, 1.5, 2.0)
        self.hu_range = (-200, 250)
        self.roi_size = (96, 96, 96)
        
        self.class_names = {0: "Background", 1: "Liver", 2: "Vessel", 3: "Tumor"}
    
    def set_parameters(self, ct_path, model_paths):
        """Set inference parameters"""
        self.ct_path = ct_path
        self.model_paths = model_paths
    
    def run(self):
        """Execute inference"""
        if not HAS_AI_MODULES:
            self.inference_complete.emit(False, "AI modules not available")
            return
            
        try:
            self.progress_update.emit("Loading models...")
            self.progress_value.emit(10)
            
            # Load models
            models = self.load_models()
            if not models:
                self.inference_complete.emit(False, "Failed to load models")
                return
            
            self.progress_update.emit("Preprocessing CT...")
            self.progress_value.emit(20)
            
            # Preprocessing
            image_tensor, meta_dict = self.preprocess_ct()
            
            self.progress_update.emit("Liver segmentation...")
            self.progress_value.emit(40)
            
            # Liver segmentation
            liver_mask = self.predict_liver(models["liver"], image_tensor)
            
            # Get ROI
            roi_slice, roi_info = self.get_liver_roi(liver_mask)
            if roi_slice is None:
                self.inference_complete.emit(False, "No liver detected")
                return
            
            self.progress_update.emit("Vessel segmentation...")
            self.progress_value.emit(60)
            
            # Vessel segmentation
            vessel_mask = self.predict_vessel(models["vessel"], image_tensor, liver_mask, roi_slice)
            
            self.progress_update.emit("Tumor segmentation...")
            self.progress_value.emit(70)
            
            # Tumor segmentation
            tumor_mask = self.predict_tumor(models.get("tumor"), image_tensor, liver_mask, roi_slice)
            
            self.progress_update.emit("Generating 3D models...")
            self.progress_value.emit(80)
            
            # Fuse masks
            fused_mask, individual_masks = self.fuse_masks(liver_mask, vessel_mask, tumor_mask)
            
            # Generate 3D models
            organ_models = self.generate_3d_models(individual_masks, meta_dict)
            
            self.progress_update.emit("Complete!")
            self.progress_value.emit(100)
            
            # Return results
            result = {
                "image_data": image_tensor.squeeze().cpu().numpy(),
                "fused_mask": fused_mask,
                "individual_masks": individual_masks,
                "organ_models": organ_models,
                "meta_dict": meta_dict,
                "success": True
            }
            
            self.inference_complete.emit(True, result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.progress_update.emit(f"Error: {str(e)}")
            self.inference_complete.emit(False, str(e))
    
    def load_models(self):
        """Load AI models"""
        models = {}
        
        try:
            if "liver" in self.model_paths and os.path.exists(self.model_paths["liver"]):
                checkpoint = torch.load(self.model_paths["liver"], map_location=self.device, weights_only=False)
                liver_model = OptimizedLiverSegClassifier(out_channels=2, task_type="liver")
                liver_model.load_state_dict(checkpoint['model_state_dict'])
                liver_model.to(self.device)
                liver_model.eval()
                models["liver"] = liver_model
            
            if "vessel" in self.model_paths and os.path.exists(self.model_paths["vessel"]):
                checkpoint = torch.load(self.model_paths["vessel"], map_location=self.device, weights_only=False)
                vessel_model = OptimizedLiverSegClassifier(out_channels=2, task_type="vessel")
                vessel_model.load_state_dict(checkpoint['model_state_dict'])
                vessel_model.to(self.device)
                vessel_model.eval()
                models["vessel"] = vessel_model
            
            if "tumor" in self.model_paths and os.path.exists(self.model_paths["tumor"]):
                checkpoint = torch.load(self.model_paths["tumor"], map_location=self.device, weights_only=False)
                tumor_model = FocusedTumorModel(out_channels=2)
                tumor_model.load_state_dict(checkpoint['model_state_dict'])
                tumor_model.to(self.device)
                tumor_model.eval()
                models["tumor"] = tumor_model
                
        except Exception as e:
            print(f"Error loading models: {e}")
            
        return models
    
    def preprocess_ct(self):
        """Preprocess CT image"""
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
        
        data = {"image": str(self.ct_path)}
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
    
    def predict_liver(self, model, image_tensor):
        """Liver segmentation prediction"""
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                if hasattr(model, 'inference'):
                    liver_output = sliding_window_inference(
                        image_tensor, self.roi_size, 4, 
                        model.inference, overlap=0.5, mode="gaussian"
                    )
                else:
                    def seg_only_inference(x):
                        result = model(x)
                        return result[0] if isinstance(result, tuple) else result
                    
                    liver_output = sliding_window_inference(
                        image_tensor, self.roi_size, 4, 
                        seg_only_inference, overlap=0.5, mode="gaussian"
                    )
        
        return torch.argmax(liver_output, dim=1, keepdim=True)
    
    def predict_vessel(self, model, image_tensor, liver_mask, roi_slice):
        """Vessel segmentation prediction"""
        image_roi = image_tensor[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        liver_roi = liver_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                roi_shape = image_roi.shape[2:]
                patch_size = (32, 32, 32) if min(roi_shape) < 64 else (64, 64, 64)
                sw_batch_size = 2 if min(roi_shape) < 64 else 4
                
                roi_output = sliding_window_inference(
                    image_roi, patch_size, sw_batch_size,
                    model, overlap=0.6, mode="gaussian"
                )
        
        roi_mask = torch.argmax(roi_output, dim=1, keepdim=True)
        roi_mask = roi_mask * liver_roi
        
        full_mask = torch.zeros_like(liver_mask)
        full_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]] = roi_mask
        
        return full_mask
    
    def predict_tumor(self, model, image_tensor, liver_mask, roi_slice):
        """Tumor segmentation prediction"""
        if model is None:
            return torch.zeros_like(liver_mask)
        
        image_roi = image_tensor[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        liver_roi = liver_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                roi_shape = image_roi.shape[2:]
                patch_size = (32, 32, 32) if min(roi_shape) < 64 else (64, 64, 64)
                sw_batch_size = 2 if min(roi_shape) < 64 else 4
                
                roi_output = sliding_window_inference(
                    image_roi, patch_size, sw_batch_size,
                    model, overlap=0.6, mode="gaussian"
                )
        
        roi_mask = torch.argmax(roi_output, dim=1, keepdim=True)
        roi_mask = roi_mask * liver_roi
        
        full_mask = torch.zeros_like(liver_mask)
        full_mask[:, :, roi_slice[0], roi_slice[1], roi_slice[2]] = roi_mask
        
        return full_mask
    
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
    
    def fuse_masks(self, liver_mask, vessel_mask, tumor_mask):
        """Fuse segmentation results"""
        liver_np = liver_mask.squeeze().cpu().numpy().astype(np.uint8)
        vessel_np = vessel_mask.squeeze().cpu().numpy().astype(np.uint8)
        tumor_np = tumor_mask.squeeze().cpu().numpy().astype(np.uint8) if tumor_mask is not None else np.zeros_like(liver_np)
        
        vessel_np = vessel_np * liver_np
        tumor_np = tumor_np * liver_np
        
        fused = np.zeros_like(liver_np, dtype=np.uint8)
        fused[liver_np == 1] = 1
        fused[vessel_np == 1] = 2
        fused[tumor_np == 1] = 3
        
        masks = {
            "liver": liver_np,
            "vessel": vessel_np,
            "tumor": tumor_np
        }
        
        return fused, masks
    
    def generate_3d_models(self, individual_masks, meta_dict):
        """Generate 3D models"""
        if not HAS_TRIMESH:
            return {}
        
        spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
        origin = meta_dict.get("origin", [0, 0, 0])
        
        organ_3d_models = {}
        generator = TrimeshMask3DGenerator()
        
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


class LiverAnalysisMainWindow(QMainWindow):
    """Main window - three-window layout"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liver AI Analysis - Integrated 2D/3D Viewer")
        self.setGeometry(100, 100, 2000, 1200)  # Increased height to 1200
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Data
        self.input_file = ""
        self.model_paths = {
            "liver": "",
            "vessel": "", 
            "tumor": ""
        }
        
        self.current_results = None
        
        self.init_ui()
    
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        dark_stylesheet = """
        /* Main window and basic components */
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9pt;
        }
        
        /* Group boxes */
        QGroupBox {
            background-color: #3c3c3c;
            border: 2px solid #555555;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            font-size: 10pt;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
            color: #00d4aa;
        }
        
        /* Button styles */
        QPushButton {
            background-color: #4a4a4a;
            color: #ffffff;
            border: 2px solid #666666;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: bold;
            min-height: 20px;
        }
        
        QPushButton:hover {
            background-color: #5a5a5a;
            border-color: #00d4aa;
        }
        
        QPushButton:pressed {
            background-color: #3a3a3a;
        }
        
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666666;
            border-color: #444444;
        }
        
        /* Special button style - Run Analysis */
        QPushButton[objectName="run_button"] {
            background-color: #00aa88;
            border-color: #00d4aa;
            color: #ffffff;
        }
        
        QPushButton[objectName="run_button"]:hover {
            background-color: #00cc99;
        }
        
        QPushButton[objectName="run_button"]:pressed {
            background-color: #008866;
        }
        
        /* Labels */
        QLabel {
            color: #ffffff;
            background-color: transparent;
        }
        
        /* Input boxes and selectors */
        QLineEdit, QComboBox {
            background-color: #404040;
            color: #ffffff;
            border: 2px solid #666666;
            border-radius: 4px;
            padding: 6px;
        }
        
        QLineEdit:focus, QComboBox:focus {
            border-color: #00d4aa;
        }
        
        /* Progress bar */
        QProgressBar {
            background-color: #404040;
            border: 2px solid #666666;
            border-radius: 4px;
            text-align: center;
            color: #ffffff;
        }
        
        QProgressBar::chunk {
            background-color: #00d4aa;
            border-radius: 2px;
        }
        
        /* Slider */
        QSlider::groove:horizontal {
            background-color: #404040;
            height: 8px;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #00d4aa;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        
        QSlider::sub-page:horizontal {
            background-color: #00aa88;
            border-radius: 4px;
        }
        
        /* Checkboxes */
        QCheckBox {
            color: #ffffff;
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #666666;
            border-radius: 3px;
            background-color: #404040;
        }
        
        QCheckBox::indicator:checked {
            background-color: #00d4aa;
            border-color: #00d4aa;
        }
        
        QCheckBox::indicator:checked:pressed {
            background-color: #008866;
        }
        
        /* Text editor - diagnostic results display */
        QTextEdit {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 2px solid #555555;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10pt;
            line-height: 1.4;
        }
        
        QTextEdit:focus {
            border-color: #00d4aa;
        }
        
        /* Splitter */
        QSplitter::handle {
            background-color: #555555;
            width: 3px;
            height: 3px;
        }
        
        QSplitter::handle:hover {
            background-color: #00d4aa;
        }
        
        /* Scrollbar */
        QScrollBar:vertical {
            background-color: #404040;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #666666;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #00d4aa;
        }
        """
        
        self.setStyleSheet(dark_stylesheet)
        
    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Main display area - three-window layout
        display_splitter = QSplitter(Qt.Horizontal)
        
        # 2D display area (dual canvas)
        display_2d = self.create_2d_display()
        display_splitter.addWidget(display_2d)
        
        # 3D display area  
        display_3d = self.create_3d_display()
        display_splitter.addWidget(display_3d)
        
        display_splitter.setSizes([1400, 600])  # 2D area wider, 3D area moderate
        main_layout.addWidget(display_splitter)
        
        # Diagnostic results display area
        results_group = QGroupBox("AI Diagnostic Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)  # Limit height
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText("No analysis results yet. Load a CT file and run AI analysis to see diagnostic results here.")
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Status bar
        self.progress_bar = QProgressBar()
        self.progress_text = QLabel("Ready")
        main_layout.addWidget(self.progress_text)
        main_layout.addWidget(self.progress_bar)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Inference thread
        self.inference_thread = InferenceThread()
        self.inference_thread.progress_update.connect(self.update_progress_text)
        self.inference_thread.progress_value.connect(self.update_progress_value)
        self.inference_thread.inference_complete.connect(self.inference_finished)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QGroupBox("Control Panel")
        layout = QFormLayout()
        
        # File selection
        self.input_path_label = QLabel("No file selected")
        self.input_button = QPushButton("Browse CT File...")
        self.input_button.clicked.connect(self.browse_input_file)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_path_label, 1)
        input_layout.addWidget(self.input_button)
        layout.addRow("CT File:", input_layout)
        
        # Model paths
        self.liver_model_label = QLabel("No model selected")
        self.liver_model_button = QPushButton("Browse...")
        self.liver_model_button.clicked.connect(lambda: self.browse_model_file("liver"))
        
        liver_layout = QHBoxLayout()
        liver_layout.addWidget(self.liver_model_label, 1)
        liver_layout.addWidget(self.liver_model_button)
        layout.addRow("Liver Model:", liver_layout)
        
        self.vessel_model_label = QLabel("No model selected")
        self.vessel_model_button = QPushButton("Browse...")
        self.vessel_model_button.clicked.connect(lambda: self.browse_model_file("vessel"))
        
        vessel_layout = QHBoxLayout()
        vessel_layout.addWidget(self.vessel_model_label, 1)
        vessel_layout.addWidget(self.vessel_model_button)
        layout.addRow("Vessel Model:", vessel_layout)
        
        self.tumor_model_label = QLabel("No model selected (optional)")
        self.tumor_model_button = QPushButton("Browse...")
        self.tumor_model_button.clicked.connect(lambda: self.browse_model_file("tumor"))
        
        tumor_layout = QHBoxLayout()
        tumor_layout.addWidget(self.tumor_model_label, 1)
        tumor_layout.addWidget(self.tumor_model_button)
        layout.addRow("Tumor Model:", tumor_layout)
        
        # Execute button
        self.run_button = QPushButton("Run AI Analysis")
        self.run_button.setObjectName("run_button")  # For special styling
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.setEnabled(False)
        layout.addRow(self.run_button)
        
        panel.setLayout(layout)
        return panel
    
    def create_2d_display(self):
        """Create dual 2D display area"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # View controls
        view_controls = QHBoxLayout()
        
        self.axial_button = QPushButton("Axial")
        self.coronal_button = QPushButton("Coronal") 
        self.sagittal_button = QPushButton("Sagittal")
        
        self.axial_button.clicked.connect(lambda: self.change_view_plane('axial'))
        self.coronal_button.clicked.connect(lambda: self.change_view_plane('coronal'))
        self.sagittal_button.clicked.connect(lambda: self.change_view_plane('sagittal'))
        
        view_controls.addWidget(QLabel("View:"))
        view_controls.addWidget(self.axial_button)
        view_controls.addWidget(self.coronal_button) 
        view_controls.addWidget(self.sagittal_button)
        view_controls.addStretch()
        
        layout.addLayout(view_controls)
        
        # Dual 2D canvas layout
        canvas_layout = QHBoxLayout()
        
        # Original image canvas
        orig_widget = QGroupBox("Original Image")
        orig_layout = QVBoxLayout()
        self.orig_canvas = NiftiSliceCanvas(self, width=6, height=6)
        self.orig_canvas.sync_callback = self.sync_both_slices
        orig_layout.addWidget(self.orig_canvas)
        orig_widget.setLayout(orig_layout)
        
        # Segmentation result canvas
        seg_widget = QGroupBox("Segmentation Result")  
        seg_layout = QVBoxLayout()
        self.seg_canvas = NiftiSliceCanvas(self, width=6, height=6)
        self.seg_canvas.sync_callback = self.sync_both_slices
        seg_layout.addWidget(self.seg_canvas)
        seg_widget.setLayout(seg_layout)
        
        canvas_layout.addWidget(orig_widget)
        canvas_layout.addWidget(seg_widget)
        layout.addLayout(canvas_layout)
        
        # Slice controls
        slice_controls = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(self.prev_slice)
        self.next_button.clicked.connect(self.next_slice)
        
        slice_controls.addWidget(self.prev_button)
        slice_controls.addWidget(self.next_button)
        layout.addLayout(slice_controls)
        
        widget.setLayout(layout)
        return widget
    
    def create_3d_display(self):
        """Create 3D display area"""
        widget = QGroupBox("3D Model View")
        layout = QVBoxLayout()
        
        # 3D viewer
        self.viewer_3d = Model3DViewer(self)
        layout.addWidget(self.viewer_3d)
        
        widget.setLayout(layout)
        return widget
    
    def sync_both_slices(self, slice_num):
        """Synchronize both 2D canvases to the same slice"""
        if not hasattr(self, 'orig_canvas') or not hasattr(self, 'seg_canvas'):
            return
            
        # Get maximum valid slices for both canvases
        max_slice_orig = self.orig_canvas.total_slices - 1 if self.orig_canvas.img_data is not None else 0
        max_slice_seg = self.seg_canvas.total_slices - 1 if self.seg_canvas.img_data is not None else max_slice_orig
        
        # Use smaller range to ensure both canvases can display the slice
        max_valid_slice = min(max_slice_orig, max_slice_seg) if max_slice_seg > 0 else max_slice_orig
        
        # Ensure slice is within valid range
        valid_slice = max(0, min(slice_num, max_valid_slice))
        
        print(f"Syncing to slice {valid_slice + 1}/{max_valid_slice + 1}")
        
        # Update original image canvas
        if self.orig_canvas.img_data is not None:
            self.orig_canvas.current_slice = valid_slice
            self.orig_canvas.update_display()
        
        # Update segmentation result canvas
        if self.seg_canvas.img_data is not None:
            self.seg_canvas.current_slice = valid_slice
            if self.seg_canvas.overlay_data is not None:
                self.seg_canvas.update_overlay_display()
            else:
                self.seg_canvas.update_display()
    
    def change_view_plane(self, plane):
        """Change viewing plane"""
        self.orig_canvas.change_view_plane(plane)
        self.seg_canvas.change_view_plane(plane)
        # Synchronize slice position
        self.sync_both_slices(self.orig_canvas.current_slice)
    
    def prev_slice(self):
        """Previous slice"""
        if hasattr(self, 'orig_canvas') and self.orig_canvas.img_data is not None:
            current_slice = self.orig_canvas.current_slice
            if current_slice > 0:
                self.sync_both_slices(current_slice - 1)
    
    def next_slice(self):
        """Next slice"""
        if hasattr(self, 'orig_canvas') and self.orig_canvas.img_data is not None:
            current_slice = self.orig_canvas.current_slice
            if current_slice < self.orig_canvas.total_slices - 1:
                self.sync_both_slices(current_slice + 1)
    
    def browse_input_file(self):
        """Select input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CT File", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        
        if file_path:
            self.input_file = file_path
            self.input_path_label.setText(os.path.basename(file_path))
            
            # Load and display original image
            try:
                nifti_img = nib.load(file_path)
                img_data = nifti_img.get_fdata()
                
                print(f"Loaded image shape: {img_data.shape}")
                print(f"Image data type: {img_data.dtype}")
                print(f"Image value range: {img_data.min()} - {img_data.max()}")
                
                # Ensure data is valid
                if img_data is not None and img_data.size > 0:
                    # Display original image in both canvases
                    self.orig_canvas.set_data(img_data)
                    self.seg_canvas.set_data(img_data)
                    
                    # Initialize synchronization
                    self.sync_both_slices(self.orig_canvas.current_slice)
                    
                    self.update_ui_state()
                    print("Image loaded successfully!")
                else:
                    raise ValueError("Empty or invalid image data")
                
            except Exception as e:
                print(f"Error loading image: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Cannot load CT file: {str(e)}")
    
    def browse_model_file(self, model_type):
        """Select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {model_type.title()} Model", "", "PyTorch Models (*.pth)"
        )
        
        if file_path:
            self.model_paths[model_type] = file_path
            
            if model_type == "liver":
                self.liver_model_label.setText(os.path.basename(file_path))
            elif model_type == "vessel":
                self.vessel_model_label.setText(os.path.basename(file_path))
            elif model_type == "tumor":
                self.tumor_model_label.setText(os.path.basename(file_path))
            
            self.update_ui_state()
    
    def update_ui_state(self):
        """Update UI state"""
        has_input = bool(self.input_file and os.path.exists(self.input_file))
        has_liver_model = bool(self.model_paths["liver"] and os.path.exists(self.model_paths["liver"]))
        has_vessel_model = bool(self.model_paths["vessel"] and os.path.exists(self.model_paths["vessel"]))
        
        can_run = has_input and has_liver_model and has_vessel_model and HAS_AI_MODULES
        self.run_button.setEnabled(can_run)
        
        if not HAS_AI_MODULES:
            self.run_button.setText("Run AI Analysis (AI modules not available)")
        else:
            self.run_button.setText("Run AI Analysis")
    
    def run_analysis(self):
        """Run analysis"""
        if not HAS_AI_MODULES:
            QMessageBox.warning(self, "Error", "AI modules are not available")
            return
            
        # Disable UI
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Start inference thread
        self.inference_thread.set_parameters(self.input_file, self.model_paths)
        self.inference_thread.start()
    
    def update_progress_text(self, text):
        """Update progress text"""
        self.progress_text.setText(text)
    
    def update_progress_value(self, value):
        """Update progress value"""
        self.progress_bar.setValue(value)
    
    def format_diagnostic_results(self, result):
        """Format diagnostic results as structured text"""
        if not result or not result.get("success", False):
            return "Analysis failed or no results available."
        
        # Get statistics
        individual_masks = result.get("individual_masks", {})
        meta_dict = result.get("meta_dict", {})
        spacing = meta_dict.get("spacing", [1.5, 1.5, 2.0])
        voxel_volume = np.prod(spacing)
        
        # Calculate volume statistics
        stats = {}
        for organ_name, mask in individual_masks.items():
            if np.sum(mask > 0) > 0:
                voxel_count = np.sum(mask > 0)
                volume_mm3 = voxel_count * voxel_volume
                volume_ml = volume_mm3 / 1000
                
                # Connected component analysis
                labeled, num_components = measure.label(mask, return_num=True)
                component_volumes = []
                if num_components > 0:
                    for i in range(1, num_components + 1):
                        comp_mask = (labeled == i)
                        comp_volume = np.sum(comp_mask) * voxel_volume / 1000
                        component_volumes.append(comp_volume)
                
                stats[organ_name] = {
                    "voxel_count": int(voxel_count),
                    "volume_ml": float(volume_ml),
                    "num_components": int(num_components),
                    "component_volumes": component_volumes,
                    "largest_component": float(max(component_volumes)) if component_volumes else 0.0
                }
        
        # Generate structured report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AI LIVER ANALYSIS - DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Image Spacing: {spacing[0]:.1f} x {spacing[1]:.1f} x {spacing[2]:.1f} mm")
        report_lines.append(f"Voxel Volume: {voxel_volume:.3f} mm³")
        report_lines.append("")
        
        # Organ analysis results
        report_lines.append("ORGAN SEGMENTATION RESULTS:")
        report_lines.append("-" * 40)
        
        organ_names = {"liver": "Liver", "vessel": "Hepatic Vessel", "tumor": "Hepatic Tumor"}
        
        for organ_key, organ_english in organ_names.items():
            if organ_key in stats:
                organ_stats = stats[organ_key]
                report_lines.append(f"")
                report_lines.append(f"* {organ_english.upper()} ({organ_key.upper()})")
                report_lines.append(f"   Volume: {organ_stats['volume_ml']:.2f} mL")
                report_lines.append(f"   Voxel Count: {organ_stats['voxel_count']:,}")
                report_lines.append(f"   Components: {organ_stats['num_components']}")
                if organ_stats['num_components'] > 1:
                    report_lines.append(f"   Largest Component: {organ_stats['largest_component']:.2f} mL")
            else:
                report_lines.append(f"")
                report_lines.append(f"* {organ_english.upper()} ({organ_key.upper()})")
                report_lines.append(f"   Status: Not detected")
        
        report_lines.append("")
        
        # Clinical assessment
        report_lines.append("CLINICAL ASSESSMENT:")
        report_lines.append("-" * 40)
        
        # Liver assessment
        if "liver" in stats:
            liver_vol = stats["liver"]["volume_ml"]
            if liver_vol < 800:
                report_lines.append("WARNING: Liver volume appears small - consider hepatic atrophy")
            elif liver_vol > 2200:
                report_lines.append("WARNING: Liver volume appears enlarged - consider hepatomegaly")
            else:
                report_lines.append("OK: Liver volume within normal range")
        else:
            report_lines.append("ERROR: Liver segmentation failed - check image quality")
        
        # Vessel assessment
        if "vessel" in stats:
            vessel_count = stats["vessel"]["num_components"]
            if vessel_count > 0:
                report_lines.append(f"OK: Hepatic vessels detected ({vessel_count} components)")
            else:
                report_lines.append("WARNING: Limited vessel visualization")
        else:
            report_lines.append("WARNING: Vessel structures not clearly identified")
        
        # Tumor assessment
        if "tumor" in stats:
            tumor_vol = stats["tumor"]["volume_ml"]
            tumor_count = stats["tumor"]["num_components"]
            largest_tumor = stats["tumor"]["largest_component"]
            
            if tumor_count == 1:
                if largest_tumor < 2.0:
                    report_lines.append(f"WARNING: Small focal lesion detected ({largest_tumor:.1f} mL)")
                    report_lines.append("   -> Recommend follow-up imaging")
                else:
                    report_lines.append(f"ALERT: Significant focal lesion detected ({largest_tumor:.1f} mL)")
                    report_lines.append("   -> Recommend immediate clinical correlation")
            elif tumor_count > 1:
                report_lines.append(f"ALERT: Multiple focal lesions detected ({tumor_count} lesions)")
                report_lines.append(f"   -> Total volume: {tumor_vol:.1f} mL")
                report_lines.append("   -> Consider metastatic disease workup")
        else:
            report_lines.append("OK: No focal hepatic lesions detected")
        
        report_lines.append("")
        
        # 3D model information
        organ_models = result.get("organ_models", {})
        if organ_models:
            report_lines.append("3D RECONSTRUCTION:")
            report_lines.append("-" * 40)
            for organ_name, model in organ_models.items():
                if hasattr(model, 'mesh_data') and model.mesh_data:
                    vertices = len(model.mesh_data.vertices)
                    faces = len(model.mesh_data.faces)
                    report_lines.append(f"OK: {organ_name.capitalize()}: {vertices:,} vertices, {faces:,} faces")
            report_lines.append("")
        
        # Technical notes
        report_lines.append("TECHNICAL NOTES:")
        report_lines.append("-" * 40)
        report_lines.append("• Analysis performed using deep learning AI models")
        report_lines.append("• 3D reconstruction via Marching Cubes algorithm")
        report_lines.append("• Results are for research/educational purposes")
        report_lines.append("• Clinical correlation and expert review recommended")
        report_lines.append("• Not intended as a substitute for professional diagnosis")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def inference_finished(self, success, result):
        """Inference completion callback"""
        self.run_button.setEnabled(True)
        
        if success and isinstance(result, dict):
            self.current_results = result
            
            # Update 2D display
            img_data = result["image_data"]
            fused_mask = result["fused_mask"]
            
            # Original image canvas displays original image
            self.orig_canvas.set_data(img_data)
            
            # Segmentation result canvas displays overlay
            self.seg_canvas.set_overlay(img_data, fused_mask)
            
            # Synchronize slice position
            self.sync_both_slices(self.orig_canvas.current_slice)
            
            # Update 3D display
            if result["organ_models"]:
                self.viewer_3d.update_display(result["organ_models"])
            
            # Display diagnostic results
            diagnostic_text = self.format_diagnostic_results(result)
            self.results_text.setPlainText(diagnostic_text)
            
            self.progress_text.setText("Analysis complete!")
            QMessageBox.information(self, "Success", "AI analysis completed successfully!")
            
        else:
            error_msg = result if isinstance(result, str) else "Analysis failed"
            self.progress_text.setText(f"Error: {error_msg}")
            self.results_text.setPlainText(f"Analysis Failed: {error_msg}\n\nPlease check your input file and model paths, then try again.")
            QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")


def main():
    app = QApplication(sys.argv)
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"VTK: {'Available' if HAS_VTK else 'Not Available'}")
    print(f"Trimesh: {'Available' if HAS_TRIMESH else 'Not Available'}")
    print(f"AI modules: {'Available' if HAS_AI_MODULES else 'Not Available'}")
    
    window = LiverAnalysisMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
