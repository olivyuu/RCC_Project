import argparse
from pathlib import Path
import logging
from models.segmentation.volume.volume_preprocessor import VolumePreprocessor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolumeConfig:
    """Simple config class for volume preprocessing"""
    def __init__(self, data_dir: str, output_dir: str, window_range: tuple):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.window_range = window_range

def main():
    parser = argparse.ArgumentParser(description='Precompute volumes for Phase 3/4 training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing case folders')
    parser.add_argument('--output_dir', type=str, 
                      default='/workspace/RCC_Project',  # preprocessor will append /preprocessed_volumes
                      help='Output directory for preprocessed files')
    parser.add_argument('--window_min', type=int, default=-1024,
                      help='Minimum HU value for windowing')
    parser.add_argument('--window_max', type=int, default=1024,
                      help='Maximum HU value for windowing')
    args = parser.parse_args()
    
    # Create simple config
    config = VolumeConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_range=(args.window_min, args.window_max)
    )
    
    # Initialize preprocessor
    preprocessor = VolumePreprocessor(config)
    
    # Process dataset
    start_time = time.time()
    preprocessor.preprocess_dataset(Path(args.data_dir))
    
    # Final stats
    total_time = time.time() - start_time
    logger.info(f"Preprocessing complete!")
    logger.info(f"Total processing time: {total_time/3600:.1f} hours")
    logger.info(f"Preprocessed volumes saved to {Path(args.output_dir)/'preprocessed_volumes'}")

if __name__ == "__main__":
    main()