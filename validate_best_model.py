import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from model import nnUNetv2
from dataset_volume import KiTS23VolumeDataset
from config import nnUNetConfig

def calculate_dice(output, target):
    """Calculate Dice score for a single case."""
    with torch.no_grad():
        # Ensure outputs and targets have same size
        if output.shape != target.shape:
            target = torch.nn.functional.interpolate(
                target,
                size=output.shape[-3:],
                mode='nearest'
            )

        pred = torch.argmax(output, dim=1)
        if len(target.shape) == len(pred.shape) + 1:
            target = target.squeeze(1)
            
        intersection = (pred * target).sum().item()
        union = pred.sum().item() + target.sum().item()
        
        return (2. * intersection + 1e-5) / (union + 1e-5)

def main():
    # Set multiprocessing start method to 'spawn'
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to best model checkpoint')
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = nnUNetConfig()
    
    # Load model
    model = nnUNetv2(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create validation dataset
    dataset = KiTS23VolumeDataset(
        config.data_dir,
        config,
        train=True,  # We'll split from full dataset
        preprocess=True
    )
    
    # Split into train/val using same seed as training
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    # Use fewer workers to avoid CUDA issues
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one volume at a time
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=KiTS23VolumeDataset.collate_fn
    )
    
    # Evaluate on validation set
    dice_scores = []
    
    print(f"\nEvaluating on {len(val_loader)} validation cases...")
    for i, (images, targets) in enumerate(tqdm(val_loader)):
        try:
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.no_grad(), autocast(enabled=config.use_amp):
                outputs = model(images)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                
                dice = calculate_dice(outputs, targets)
                dice_scores.append(dice)
                
            # Clear GPU memory after each case
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing case {i}: {str(e)}")
            continue
            
    # Calculate statistics
    dice_array = np.array(dice_scores)
    mean_dice = np.mean(dice_array)
    std_dice = np.std(dice_array)
    median_dice = np.median(dice_array)
    
    print("\nValidation Results:")
    print(f"Mean Dice ± std: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Median Dice: {median_dice:.4f}")
    print(f"Range: [{np.min(dice_array):.4f}, {np.max(dice_array):.4f}]")
    print(f"\nPublication format:")
    print(f"The model achieved a best validation Dice score of {np.max(dice_array):.4f}, "
          f"with a mean Dice of {mean_dice:.4f} ± {std_dice:.4f} across the validation cases.")

if __name__ == '__main__':
    main()