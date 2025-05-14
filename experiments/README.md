# Experiment Tracking

This directory contains all experiment configurations, results, and model checkpoints to ensure reproducibility.

## Directory Structure

```
experiments/
├── patch_training/
│   ├── configs/               # Training configurations
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   └── splits/               # Dataset splits
│
├── volume_training/
│   ├── configs/              # Full volume training configs
│   ├── checkpoints/          # Full volume model checkpoints
│   ├── logs/                 # Training logs
│   └── splits/               # Dataset splits
│
└── transitions/              # Documentation of patch→volume transitions
    └── README.md            # Transition notes and results
```

## Reproducibility Guidelines

### 1. Configuration Management
- Every training run must have its configuration saved
- Use timestamp-based naming: `config_YYYYMMDD_HHMMSS.yaml`
- Include all hyperparameters, even if using defaults

### 2. Dataset Splits
- Save train/val/test splits for each experiment
- Include case IDs and preprocessing parameters
- Document any data filtering/selection criteria

### 3. Model Checkpoints
- Save with descriptive names: `model_type_dice_score_epoch.pth`
- Include validation metrics in filename
- Keep both best and final models

### 4. Logging
- Use tensorboard for training metrics
- Save console outputs to log files
- Document any manual interventions

### 5. Transition Documentation
When transitioning from patch to full volume:

1. Record patch model details:
```yaml
patch_model:
  checkpoint: "path/to/checkpoint.pth"
  train_config: "path/to/config.yaml"
  validation_dice: X.XXX
  training_date: "YYYY-MM-DD"
```

2. Document transition parameters:
```yaml
transition:
  freeze_layers: true/false
  freeze_epochs: N
  learning_rate: X.XXX
  gradient_accumulation: N
```

3. Track results:
```yaml
results:
  initial_dice: X.XXX  # First epoch on full volumes
  best_dice: X.XXX
  epochs_to_best: N
  total_epochs: N
```

## Usage Example

1. Start new experiment:
```bash
# Create experiment directory
python tools/create_experiment.py --name "volume_ft_001"

# Save initial configuration
python tools/save_config.py --exp "volume_ft_001" --config config.yaml

# Start training with logging
python train_volume.py \
    --experiment "volume_ft_001" \
    --patch_weights "experiments/patch_training/checkpoints/best_model.pth" \
    --config "experiments/volume_ft_001/config.yaml"
```

2. Monitor training:
```bash
tensorboard --logdir experiments/volume_ft_001/logs
```

3. Document transition:
```bash
python tools/document_transition.py \
    --patch_exp "patch_001" \
    --volume_exp "volume_ft_001"
```

## Reviewing Experiments

Each experiment directory contains:
1. Full configuration used
2. Dataset splits for reproduction
3. Training logs and metrics
4. Model checkpoints
5. Transition documentation (for full volume experiments)

This structure ensures that any experiment can be reproduced by:
1. Using the same configuration
2. Following the same dataset splits
3. Using the same random seeds
4. Following the documented transition process