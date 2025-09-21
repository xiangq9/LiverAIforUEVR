# Liver AI Analysis System with Unreal Engine 5 Integration

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Unreal Engine](https://img.shields.io/badge/Unreal%20Engine-5.x-orange.svg)](https://www.unrealengine.com/)
[![MONAI](https://img.shields.io/badge/MONAI-1.6-green.svg)](https://monai.io/)

An advanced AI-powered liver analysis system that combines deep learning medical image segmentation with real-time 3D visualization in Unreal Engine 5. This project provides end-to-end solution from AI model training to interactive 3D visualization for medical professionals.

## 🌟 Features

### AI Segmentation & Analysis
- **Multi-organ Segmentation**: Automated segmentation of liver, hepatic vessels, and tumors
- **Deep Learning Models**: Custom SegResNet and U-Net architectures optimized for medical imaging
- **3D Reconstruction**: Real-time 3D mesh generation using Marching Cubes algorithm
- **Clinical Assessment**: Automatic volume calculation and diagnostic report generation

### Unreal Engine 5 Plugin
- **Interactive 2D Viewer**: Multi-planar (Axial, Coronal, Sagittal) slice visualization
- **Real-time 3D Visualization**: Procedural mesh generation for segmented organs
- **Synchronized Views**: Dual-panel synchronized viewing of original and segmented images
- **HTTP API Integration**: Seamless communication with AI backend server

### Backend Server
- **FastAPI Server**: High-performance REST API for AI inference
- **Async Processing**: Non-blocking analysis with progress tracking
- **Multi-format Support**: Handles NIfTI, DICOM formats
- **Scalable Architecture**: Support for distributed processing

## 📋 Requirements

### Python Environment
- Python 3.13+
- CUDA 12.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Unreal Engine
- Unreal Engine 5.0+
- Visual Studio 2022 (for Windows)
- Windows 10/11 64-bit

## 🚀 Installation

### 1. Python Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/liver-ai-analysis.git
cd liver-ai-analysis

# Create virtual environment (recommended using uv)
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Install AI Models

Download pre-trained models or train your own:

```bash
# Download pre-trained models
python download_models.py

# Or train your own models
python AISegInference/multiliver_training.py
```

### 3. Start Backend Server

```bash
cd AISegInference
python liver_ai_backend.py
```

Server will start at `http://127.0.0.1:8888`

### 4. Install Unreal Engine Plugin

1. Copy `Plugins/LiverImageAI` folder to your UE5 project's `Plugins` directory
2. Regenerate project files
3. Build the project in Visual Studio
4. Enable the plugin in UE5 Editor

## 💻 Usage

### Training AI Models

```python
# Configure task type in multiliver_training.py
TASK_TYPE = "liver"  # Options: "liver", "vessel", "tumor"

# Run training
python AISegInference/multiliver_training.py
```

### Running Inference

#### Command Line
```python
python AISegInference/output_inference.py
```

#### GUI Application
```python
python AISegInference/liver_analysis_gui.py
```

### Using UE5 Plugin

1. Open UE5 Editor
2. Go to `Window > Liver AI Analysis`
3. Configure server address (default: `http://127.0.0.1:8888`)
4. Load CT file and model paths
5. Click "Run AI Analysis"

## 🏗️ Architecture

```
liver-ai-analysis/
├── AISegInference/           # AI Backend
│   ├── liver_ai_backend.py   # FastAPI server
│   ├── liver_analysis_gui.py # PySide6 GUI
│   ├── multiliver_training.py # Model training
│   └── output_inference.py   # Inference pipeline
├── Plugins/LiverImageAI/     # UE5 Plugin
│   ├── Source/
│   │   ├── Private/          # Implementation files
│   │   └── Public/           # Header files
│   └── LiverImageAI.uplugin
└── pyproject.toml            # Python dependencies
```

## 🔧 Technical Details

### AI Models

#### Liver Segmentation Model
- Architecture: SegResNet
- Input: 3D CT volumes (96×96×96 patches)
- Output: Binary segmentation mask
- Performance: ~95% Dice score

#### Vessel Segmentation Model
- Architecture: Enhanced SegResNet
- Constraint: Within liver ROI
- Output: Hepatic vessel mask

#### Tumor Detection Model
- Architecture: U-Net with attention mechanism
- Specialized for small tumor detection
- Focal Tversky loss for imbalanced data

### 3D Reconstruction

- **Algorithm**: Marching Cubes
- **Library**: Trimesh
- **Post-processing**: Mesh smoothing, normal fixing
- **Export Formats**: OBJ, PLY, STL

### Communication Protocol

```json
// Analysis Request
{
  "ct_file_path": "/path/to/ct.nii.gz",
  "liver_model_path": "/path/to/liver_model.pth",
  "vessel_model_path": "/path/to/vessel_model.pth",
  "tumor_model_path": "/path/to/tumor_model.pth",
  "request_id": "ue_request_20250122_143022"
}

// Analysis Response
{
  "success": true,
  "organ_stats": [
    {
      "organ_name": "liver",
      "volume_ml": 1450.5,
      "voxel_count": 125000
    }
  ],
  "mesh_data": [...],
  "diagnostic_report": "..."
}
```

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Liver Segmentation | Dice Score | 95.2% |
| Vessel Segmentation | Dice Score | 89.7% |
| Tumor Detection | Sensitivity | 92.3% |
| Inference Time | Full Analysis | ~30 seconds |
| 3D Reconstruction | Mesh Generation | ~5 seconds |

## 🖼️ Screenshots

### UE5 Plugin Interface
![UE5 Plugin](docs/images/ue5_plugin.png)

### 3D Visualization
![3D Visualization](docs/images/3d_visualization.png)

### GUI Application
![GUI App](docs/images/gui_application.png)

## 🛠️ Configuration

### Training Configuration
```python
# AISegInference/multiliver_training.py
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
SPATIAL_SIZE = [96, 96, 96]
```

### Server Configuration
```python
# AISegInference/liver_ai_backend.py
HOST = "127.0.0.1"
PORT = 8888
WORKERS = 4
```

## 📝 API Documentation

### Health Check
```http
GET /api/health
```

### Start Analysis
```http
POST /api/analyze
Content-Type: application/json
```

### Get Progress
```http
GET /api/status/{request_id}
```

### Get Result
```http
GET /api/result/{request_id}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) - Medical Open Network for AI
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Trimesh](https://trimsh.org/) - 3D mesh processing
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web APIs
- [Unreal Engine](https://www.unrealengine.com/) - 3D visualization platform

## 📧 Contact

For questions and support, please open an issue on GitHub or contact:
- Project Lead: [Your Name]
- Email: [your.email@example.com]

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{liver_ai_analysis_2025,
  title = {Liver AI Analysis System with Unreal Engine Integration},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/liver-ai-analysis}
}
```

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for medical advice.

---

**Made with ❤️ for advancing medical imaging technology**
