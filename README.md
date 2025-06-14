# RCC Project

This project implements a 3D medical image segmentation pipeline using a modified nnUNet architecture, with support for both patch-based and full volume training approaches.

Disclaimer: the project has been abandoned. It currently does not achieve its goal.

## Installation on RunPod.io

1. Install the KiTS23 package and download the dataset:
```bash
cd /workspace
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
kits23_download_data
```

2. Set up the Python environment with required dependencies (order is important!):
```bash
cd /workspace/RCC_project

# First uninstall any existing numpy to avoid conflicts
pip uninstall -y numpy

# Install numpy 1.x specifically (required for PyTorch compatibility)
pip install 'numpy>=1.24.0,<2.0.0'

# Install PyTorch
pip install torch torchvision

# Install remaining requirements
pip install -r requirements.txt
```

## Training Approaches

### Patch-based Training
Initially, the model is trained on patches (smaller 3D segments) of the medical volumes. This approach:
- Enables training on machines with limited memory
- Provides more training samples through patch extraction
- Allows for effective data augmentation

### Full Volume Training
After successful patch training, you can fine-tune the model on full volumes. This:
- Improves global context understanding
- Enhances boundary consistency
- Better matches the inference scenario

## Dependencies

The project requires the following major dependencies (all included in requirements.txt):
- PyTorch
- CUDA (for GPU support)
- nibabel (for medical image processing)
- numpy (version 1.x for compatibility)
- tqdm (for progress bars)
- tensorboard (for logging)
- matplotlib (for visualization)

Note: We specifically use numpy 1.x as numpy 2.x is currently incompatible with some PyTorch operations.

## Project Status
This project has been archived as of June 8, 2025. The final state includes:
- Complete patch-based training pipeline
- Full-volume segmentation with kidney masking
- Tumor-rich window warmup functionality

For any questions, please open an issue on GitHub.