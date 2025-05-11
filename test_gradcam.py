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
    config = nnUNetConfig()
    model = nnUNetv2(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features
    )
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.cuda() if torch.cuda.is_available() else model

def test_gradcam(checkpoint_path='checkpoints/latest.pth', num_samples=5):
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
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path)
    model.eval()
    
    # Set the target layer for Grad-CAM (using the bottleneck layer)
    model.set_target_layer('bottleneck')
    
    # Create output directory for visualizations
    output_dir = Path('gradcam_results')
    output_dir.mkdir(exist_ok=True)
    
    # Randomly select samples
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"\nProcessing {len(indices)} samples...")
    for idx in indices:
        print(f"\nAnalyzing sample {idx}...")
        
        # Get the sample
        image, mask = dataset[idx]
        image = image.unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            image = image.cuda()
        
        try:
            # Generate Grad-CAM
            print("Generating Grad-CAM visualization...")
            cam, prediction = generate_gradcam(model, image)
            
            # Get the original image
            original_image = image.squeeze().cpu().numpy()
            
            # Create visualization paths
            base_path = output_dir / f'sample_{idx}'
            full_scan_path = base_path.with_suffix('.png')
            
            print("Creating visualizations...")
            # This will create both full scan and detailed region visualizations
            visualize_gradcam(
                image=original_image,
                cam=cam,
                prediction=prediction,
                save_path=full_scan_path
            )
            
            print(f"Saved visualizations for sample {idx}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for model predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/latest.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=5,
                      help='Number of samples to process')
    
    args = parser.parse_args()
    print("\nStarting Grad-CAM visualization generation...")
    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Processing {args.samples} samples")
    
    test_gradcam(args.checkpoint, args.samples)
    print("\nVisualization generation complete!")