# RCC Project

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
cd /workspace/RCC_Project

# First uninstall any existing numpy to avoid conflicts
pip uninstall -y numpy

# Install numpy 1.x specifically (required for PyTorch compatibility)
pip install 'numpy>=1.24.0,<2.0.0'

# Install PyTorch
pip install torch torchvision

# Install remaining requirements
pip install -r requirements.txt
```

## Running the Training

To run the training with custom batch size and number of epochs:
```bash
python trainer.py --batch_size 2 --epochs 1
```

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