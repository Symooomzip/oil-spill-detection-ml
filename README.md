# 🛢️ Oil Spill Detection System

A comprehensive deep learning system for detecting and segmenting oil spills in satellite/aerial imagery using state-of-the-art computer vision models.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Dataset](#dataset)
- [Training](#training)
- [Web Interface](#web-interface)
- [Results](#results)

## 🌊 Overview

This project implements multiple state-of-the-art deep learning models for oil spill detection with two main tasks:

1. **Classification**: Determine if an image contains an oil spill
2. **Segmentation**: Precisely locate and segment oil spill regions

The system includes a user-friendly Gradio web interface for easy inference and visualization.

## ✨ Features

- 🎯 **Multiple Classification Models**: InternImage, MobileNetV4, RepLKNet, ConvNeXtV2, EfficientNetV3
- 🎨 **Advanced Segmentation Models**: VMUNet, UNet++, TransUNet, U-Mamba, MultiCBAM
- 🖥️ **Interactive Web Interface**: Gradio-based UI for easy model testing
- 📊 **Comprehensive Training Pipeline**: Automated training with performance tracking
- 📈 **Visualization Tools**: Performance plots and segmentation overlays
- 💾 **Model Persistence**: Save and load trained models
- 🔄 **Data Augmentation**: Built-in augmentation for robust training

## 📊 Model Performance

### Classification Models

| Model          | Validation Accuracy |
| -------------- | ------------------- |
| InternImage    | 100.0%              |
| MobileNetV4    | 100.0%              |
| RepLKNet       | 100.0%              |
| ConvNeXtV2     | 100.0%              |
| EfficientNetV3 | 100.0%              |

### Segmentation Models

| Model     | Validation IoU |
| --------- | -------------- |
| VMUNet    | 0.8769         |
| UNet++    | 0.8662         |
| TransUNet | 0.8454         |
| U-Mamba   | 0.8444         |
| MultiCBAM | 0.8439         |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/oil-spill-detection.git
cd oil-spill-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv oil_spill_env

# Activate virtual environment
# On Windows:
oil_spill_env\\Scripts\\activate
# On Linux/Mac:
source oil_spill_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gradio opencv-python numpy matplotlib pandas pillow
pip install timm segmentation-models-pytorch albumentations
pip install roboflow scikit-learn tqdm
```

## 💻 Usage

### Quick Start with Web Interface

Run the Gradio web application:

```bash
python oil_spill_gradio_app.py
```

The interface will open in your browser at `http://localhost:7860`

### Using the Main Script

For training and evaluation:

```bash
python main.py
```

### Training Custom Models

```bash
python run_training.py
```

## 📁 Project Structure

```
oil-spill-detection/
├── main.py                      # Main training and evaluation script
├── oil_spill_gradio_app.py     # Gradio web interface
├── run_training.py             # Training pipeline
├── helper_functions.py         # Utility functions
├── REPORT.md                   # Performance report
├── OilSpillProject/            # Trained models directory
│   ├── *_best.pth             # Best model checkpoints
│   └── *_final.pth            # Final model checkpoints
├── classification_results.png  # Classification performance plots
├── segmentation_results.png    # Segmentation performance plots
└── README.md                   # This file
```

## 🤖 Models

### Classification Models

1. **InternImage**: Transformer-based model with large kernel attention
2. **MobileNetV4**: Efficient mobile-optimized architecture
3. **RepLKNet**: Large kernel convolutional network
4. **ConvNeXtV2**: Modern ConvNet architecture
5. **EfficientNetV3**: Compound scaling efficient network

### Segmentation Models

1. **VMUNet**: Vision Mamba U-Net for medical/satellite segmentation
2. **UNet++**: Nested U-Net with dense skip connections
3. **TransUNet**: Transformer-CNN hybrid architecture
4. **U-Mamba**: U-Net with Mamba blocks
5. **MultiCBAM**: Multi-scale CBAM attention mechanism

## 📦 Dataset

The project uses the **Oil Spill Detection** dataset from Roboflow:

- **Format**: COCO
- **Task**: Instance segmentation
- **Classes**: Oil spill regions
- **Splits**: Train, Validation, Test

Dataset is automatically downloaded when running the scripts.

## 🏋️ Training

### Train All Models

```bash
python run_training.py
```

### Train Specific Model Type

Edit `run_training.py` to select classification or segmentation:

```python
# For classification only
cls_results = train_all_classification_models(train_loader, val_loader, num_epochs=30)

# For segmentation only
seg_results = train_all_segmentation_models(train_loader, val_loader, num_epochs=30)
```

### Training Configuration

- **Batch Size**: 16 (adjustable based on GPU memory)
- **Learning Rate**: 0.001
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing / StepLR
- **Loss Functions**:
  - Classification: CrossEntropyLoss
  - Segmentation: Dice Loss
- **Metrics**:
  - Classification: Accuracy
  - Segmentation: IoU (Intersection over Union)

## 🌐 Web Interface

The Gradio interface provides two main tabs:

### 1. Classification Tab

- Upload an image
- Select a classification model
- Get prediction with confidence scores
- View probability distribution

### 2. Segmentation Tab

- Upload an image
- Select a segmentation model
- View segmentation mask
- See overlay visualization
- Get oil spill area statistics

## 📈 Results

Performance visualizations are automatically generated:

- `classification_results.png`: Accuracy and loss curves for all classification models
- `segmentation_results.png`: IoU and loss curves for all segmentation models

See `REPORT.md` for detailed performance analysis.

## 🛠️ Advanced Usage

### Custom Dataset

To use your own dataset:

1. Prepare data in COCO format
2. Update dataset path in `main.py`:
   ```python
   DATASET_PATH = "path/to/your/dataset"
   ```

### Model Inference

```python
from helper_functions import load_model, predict

# Load model
model = load_model("InternImage_best.pth")

# Predict
result = predict(model, "path/to/image.jpg")
```

### Export Models

Models are saved in PyTorch format (`.pth`):

- `*_best.pth`: Best validation performance
- `*_final.pth`: Final epoch weights

## 🐛 Troubleshooting

### CUDA Out of Memory

- Reduce batch size in training scripts
- Use smaller model variants
- Enable gradient checkpointing

### Model Loading Errors

- Ensure model files are in `OilSpillProject/` directory
- Check file naming convention: `ModelName_best.pth`
- Verify PyTorch version compatibility

### Gradio Interface Issues

- Check port 7860 is not in use
- Try different port: `demo.launch(server_port=7861)`
- Ensure models are properly loaded

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{oil-spill-detection-2024,
  title={Oil Spill Detection System using Deep Learning},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/oil-spill-detection}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Roboflow](https://roboflow.com/) for dataset hosting
- [Timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) for segmentation architectures
- [Gradio](https://gradio.app/) for the web interface framework

## 👥 Authors

**Fakir mohammed**

- GitHub: [@Symooomzip](https://github.com/Symooomzip)

## 🔗 Links

- [Project Repository](https://github.com/Symooomzip/oil-spill-detection)
- [Issue Tracker](https://github.com/Symooomzip/oil-spill-detection/issues)
- [Dataset](https://universe.roboflow.com/mon-ehjb6/oil-spill-detection-gajxc)

---

**Made with ❤️ for environmental protection**
