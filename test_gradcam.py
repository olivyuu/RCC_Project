import torch
from pathlib import Path
import matplotlib.pyplot as plt
from dataset import KiTS23Dataset
from model import nnUNetv2
from visualization import generate_gradcam, visualize_gradcam
from config import nnUNetConfig
import random

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    model = nnUNetv2()
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.cuda() if torch.cuda.is_available() else model

def test_gradcam(checkpoint_path='checkpoints/best_model_dice_0.8334.pth', num_samples=5):
    """Test Grad-CAM on validation cases"""
    # Load configuration
    config = nnUNetConfig()
    
    # Initialize dataset in validation mode
    dataset = KiTS23Dataset(
        root_dir=config.data_dir,
        config=config,
        train=False,
        preprocess=False  # Use existing preprocessed files
    )
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Set the target layer for Grad-CAM (using the bottleneck layer)
    model.set_target_layer('bottleneck')
    
    # Create output directory for visualizations
    output_dir = Path('gradcam_results')
    output_dir.mkdir(exist_ok=True)
    
    # Randomly select samples
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    for idx in indices:
        print(f"\nProcessing sample {idx}...")
        
        # Get the sample
        image, mask = dataset[idx]
        image = image.unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            image = image.cuda()
        
        # Generate Grad-CAM
        cam, prediction = generate_gradcam(model, image)
        
        # Get the original image and model outputs
        original_image = image.squeeze().detach().cpu().numpy()
        
        # Create visualization
        print(f"Generating visualization for sample {idx}...")
        save_path = output_dir / f'gradcam_sample_{idx}.png'
        visualize_gradcam(
            image=original_image,
            cam=cam,
            prediction=prediction,  # Pass model prediction for tumor probabilities
            slice_indices=None,  # Will automatically find slices with tumors
            save_path=save_path
        )
        
        print(f"Saved visualization to {save_path}")

if __name__ == '__main__':
    test_gradcam()