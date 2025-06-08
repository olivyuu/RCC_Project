import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm
import random
from models.segmentation.model import SegmentationModel
from models.segmentation.volume.dataset_volume import KiTS23VolumeDataset

def find_topk_slices(volume: torch.Tensor, k: int = 3):
    """
    Find top-k slices with most tumor in each anatomical plane
    Args:
        volume: Tensor of shape [D,H,W]
        k: Number of top slices to return per plane
    Returns:
        Dict with top slice indices for each plane
    """
    # Axial: sum over H,W → length-D array
    axial_counts = volume.sum(dim=(1,2)).cpu().numpy()      # shape [D]
    # Coronal: sum over D,W → length-H 
    coronal_counts = volume.sum(dim=(0,2)).cpu().numpy()    # shape [H]
    # Sagittal: sum over D,H → length-W
    sagittal_counts = volume.sum(dim=(0,1)).cpu().numpy()   # shape [W]

    top_axial = axial_counts.argsort()[-k:][::-1].tolist()
    top_coronal = coronal_counts.argsort()[-k:][::-1].tolist()
    top_sagittal = sagittal_counts.argsort()[-k:][::-1].tolist()

    return {
        "axial": top_axial,
        "coronal": top_coronal,
        "sagittal": top_sagittal
    }

def visualize_predictions(ct_vol: torch.Tensor,
                         kidney_vol: torch.Tensor,
                         tumor_vol: torch.Tensor,
                         pred_vol: torch.Tensor,
                         case_id: str,
                         slice_info: dict,
                         output_dir: Path):
    """Create visualizations for each plane's top slices"""
    
    for plane, slice_idxs in slice_info.items():
        for z in slice_idxs:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Extract appropriate slices based on plane
            if plane == "axial":
                ct_slice = ct_vol[z]
                kidney_slice = kidney_vol[z]
                tumor_slice = tumor_vol[z]
                pred_slice = pred_vol[z]
                view_name = f"Axial z={z}"
            elif plane == "coronal":
                ct_slice = ct_vol[:,z,:]
                kidney_slice = kidney_vol[:,z,:]
                tumor_slice = tumor_vol[:,z,:]
                pred_slice = pred_vol[:,z,:]
                view_name = f"Coronal y={z}"
            else:  # sagittal
                ct_slice = ct_vol[:,:,z]
                kidney_slice = kidney_vol[:,:,z]
                tumor_slice = tumor_vol[:,:,z]
                pred_slice = pred_vol[:,:,z]
                view_name = f"Sagittal x={z}"
            
            # Plot all modalities
            axes[0,0].imshow(ct_slice.cpu(), cmap='gray')
            axes[0,0].set_title('CT')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(kidney_slice.cpu(), cmap='gray')
            axes[0,1].set_title('Kidney Mask')
            axes[0,1].axis('off')
            
            axes[1,0].imshow(tumor_slice.cpu(), cmap='gray')
            axes[1,0].set_title('Ground Truth Tumor')
            axes[1,0].axis('off')
            
            # Plot predicted tumor with colorbar
            im = axes[1,1].imshow(pred_slice.cpu(), cmap='gray')
            axes[1,1].set_title('Predicted Tumor')
            axes[1,1].axis('off')
            plt.colorbar(im, ax=axes[1,1])
            
            # Add case/slice information
            plt.suptitle(f'Case {case_id} - {view_name}')
            
            # Save figure
            output_path = output_dir / f'case_{case_id}_{plane}_slice_{z}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test volume segmentation model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory (parent of preprocessed_volumes)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--features', type=int, default=32,
                       help='Number of base features')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Probability threshold for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for volume selection')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'test_results.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SegmentationModel(
        in_channels=2,  # CT + kidney mask
        out_channels=1, # Tumor
        features=args.features
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
    logger.info(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
    
    # Create dataset
    dataset = KiTS23VolumeDataset(
        data_dir=args.data_dir,  # Parent directory of preprocessed_volumes
        use_kidney_mask=True,
        training=False,
        debug=False
    )
    
    # Randomly select 3 test volumes
    random.seed(args.seed)
    all_indices = list(range(len(dataset)))
    test_indices = random.sample(all_indices, k=3)
    
    # Test on selected volumes
    model.eval()
    all_dice_scores = []
    soft_dice_scores = []  # Track both thresholded and soft Dice
    
    with torch.no_grad():
        for idx in test_indices:
            data = dataset[idx]
            inputs = data['input'].unsqueeze(0).to(device)  # Add batch dimension
            targets = data['target'].to(device)
            case_id = data['case_id']
            
            # Get predictions
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)[0,0]  # Remove batch/channel dims
            pred_mask = (probs > args.threshold).float()
            
            # Calculate metrics
            target_tumor = targets[0,0]  # Remove batch/channel dims
            
            # Hard Dice (thresholded)
            intersection = (pred_mask * target_tumor).sum().item()
            union = pred_mask.sum().item() + target_tumor.sum().item()
            dice_score = (2 * intersection + 1e-5) / (union + 1e-5)
            all_dice_scores.append(dice_score)
            
            # Soft Dice (on probabilities)
            soft_intersection = (probs * target_tumor).sum().item()
            soft_union = probs.sum().item() + target_tumor.sum().item()
            soft_dice = (2 * soft_intersection + 1e-5) / (soft_union + 1e-5)
            soft_dice_scores.append(soft_dice)
            
            # Find top slices in each plane
            slice_info = find_topk_slices(target_tumor)
            
            # Log volume-level results
            logger.info(f"\nCase {case_id}:")
            logger.info(f"Ground truth tumor voxels: {int(target_tumor.sum().item())}")
            logger.info(f"Predicted tumor voxels: {int(pred_mask.sum().item())}")
            logger.info(f"Hard Dice (thresh={args.threshold:.2f}): {dice_score:.4f}")
            logger.info(f"Soft Dice: {soft_dice:.4f}")
            logger.info("Top slices selected:")
            for plane, slices in slice_info.items():
                logger.info(f"  {plane}: {slices}")
            
            # Generate visualizations
            visualize_predictions(
                ct_vol=inputs[0,0],        # First image, CT channel
                kidney_vol=inputs[0,1],    # First image, kidney channel
                tumor_vol=target_tumor,    # Ground truth tumor
                pred_vol=probs,            # Raw probabilities
                case_id=case_id,
                slice_info=slice_info,
                output_dir=output_dir
            )
            
            # Save raw scores for this case
            np.save(
                output_dir / f'case_{case_id}_probs.npy',
                probs.cpu().numpy()
            )
            
            # Clear GPU memory
            torch.cuda.empty_cache()
    
    # Log overall results
    logger.info("\nOverall Results:")
    logger.info(f"Mean Hard Dice: {np.mean(all_dice_scores):.4f}")
    logger.info(f"Mean Soft Dice: {np.mean(soft_dice_scores):.4f}")
    logger.info(f"Test cases: {test_indices}")
    
    # Save all scores
    np.savez(
        output_dir / 'dice_scores.npz',
        hard_dice=np.array(all_dice_scores),
        soft_dice=np.array(soft_dice_scores),
        case_indices=np.array(test_indices)
    )

if __name__ == '__main__':
    main()