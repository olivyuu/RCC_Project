import argparse
from pathlib import Path
import logging
import time
from models.segmentation.volume.volume_preprocessor import VolumePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeConfig:
    """Simple config class for volume preprocessing"""
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.window_range = (-1024, 1024)
        self.volume_max_dim = (128, 256, 256)  # Target shape for downsampling

def main():
    parser = argparse.ArgumentParser(description='Precompute volumes for Phase 3/4 training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing case folders')
    parser.add_argument('--output_dir', type=str, 
                      default='/workspace/RCC_Project',  # preprocessor will append /preprocessed_volumes
                      help='Output directory for preprocessed files')
    args = parser.parse_args()
    
    print(f"Starting preprocessing pipeline:")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target shape: (128, 256, 256)")
    print(f"Data type: float32 (4 bytes/voxel)")
    print("=" * 80)
    
    # Create config
    config = VolumeConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Initialize preprocessor
    preprocessor = VolumePreprocessor(config)
    
    # Process dataset
    start_time = time.time()
    preprocessor.preprocess_dataset(Path(args.data_dir))
    
    # Final stats
    total_time = time.time() - start_time
    print(f"Preprocessing complete!")
    print(f"Total processing time: {total_time/3600:.1f} hours")
    print(f"Preprocessed volumes saved to {Path(args.output_dir)/'preprocessed_volumes'}")

if __name__ == "__main__":
    main()