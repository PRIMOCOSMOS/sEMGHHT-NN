# Training Script Usage Guide

This guide explains how to use the improved training script for sEMG-HHT CNN-SVM classifier.

## Features

1. **Checkpoint Saving and Resuming**: Models are saved at regular intervals and training can be resumed
2. **6-Class Multi-Label Classification**: 
   - Gender: Male (M), Female (F)
   - Movement Quality: Full, Half, Invalid
   - Total: 6 classes (M_full, M_half, M_invalid, F_full, F_half, F_invalid)
3. **Automatic Test File Handling**: Files with 'test' in the name are automatically used for inference after training
4. **Accuracy Testing**: Validation accuracy is computed and reported

## Data Format

### File Naming Convention

Training files should follow this naming pattern:
```
MUSCLENAME_movement_GENDER_###.npz
```

Examples:
- `BICEPS_fatiguetest_M_006.npz` → Male, Full movement
- `TRICEPS_full_F_012.npz` → Female, Full movement  
- `FOREARM_half_M_003.npz` → Male, Half movement
- `DELTOID_invalid_F_008.npz` → Female, Invalid movement
- `BICEPS_wrong_M_002.npz` → Male, Invalid movement

Test files (for inference only) should contain 'test' in the filename:
- `BICEPS_test_001.npz`
- `TRICEPS_test_M_002.npz`

### File Content

Each `.npz` file should contain a 256×256 HHT matrix. The matrix should be stored with key `'hht'` or as the first array in the file.

```python
import numpy as np

# Save HHT matrix
np.savez('BICEPS_fatiguetest_M_001.npz', hht=hht_matrix)

# Or simply
np.savez('BICEPS_fatiguetest_M_001.npz', hht_matrix)
```

## Quick Start

### 1. Generate Sample Data (for testing)

```bash
python generate_sample_data.py --output_dir ./data --n_samples 20 --n_test 10
```

This creates:
- 120 training samples (20 per class × 6 classes)
- 10 test samples (unlabeled)

### 2. Train the Model

```bash
python train.py --data_dir ./data --checkpoint_dir ./checkpoints
```

### 3. Resume Training (if interrupted)

```bash
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume
```

## Command Line Options

```bash
python train.py --help
```

Available options:
- `--data_dir`: Directory containing .npz files (required)
- `--checkpoint_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--test_size`: Validation set size, 0-1 (default: 0.2)
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--resume`: Resume from latest checkpoint
- `--cpu`: Force CPU usage (otherwise uses GPU if available)

## Training Output

The training script will:

1. Load and parse all `.npz` files from the data directory
2. Separate training files from test files (files with 'test' in name)
3. Split training data into train/validation sets
4. Train the CNN-SVM classifier
5. Evaluate on both training and validation sets
6. Save checkpoints with model weights and metadata
7. Run inference on test files
8. Save predictions to JSON file

### Checkpoint Files

For each checkpoint, the following files are saved:
- `checkpoint_encoder.pt`: CNN encoder weights
- `checkpoint_scaler.pkl`: Feature scaler
- `checkpoint_svm.pkl`: Trained SVM classifier
- `checkpoint_metadata.pkl`: Training metadata (epoch, history, etc.)

### Test Predictions

Predictions for test files are saved to `checkpoints/test_predictions.json`:

```json
{
  "BICEPS_test_001.npz": {
    "prediction": "M_full",
    "confidence": 0.8523,
    "probabilities": {
      "M_full": 0.8523,
      "M_half": 0.0234,
      "M_invalid": 0.0145,
      "F_full": 0.0567,
      "F_half": 0.0321,
      "F_invalid": 0.0210
    }
  }
}
```

## Example Training Session

```bash
# Generate sample data
python generate_sample_data.py --output_dir ./data --n_samples 30

# Train model
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# Output will show:
# - Data loading statistics
# - Class distribution
# - Training progress
# - Validation accuracy
# - Test file predictions
```

## Using with Real Data

1. Prepare your HHT matrices as 256×256 numpy arrays
2. Save them as `.npz` files with appropriate filenames
3. Place all files in a single directory
4. Run the training script pointing to that directory

```python
# Example: Convert your sEMG data to HHT and save
import numpy as np

# Your sEMG processing code here...
hht_matrix = compute_hht_transform(semg_signal)  # Should be 256×256

# Save with appropriate filename
filename = f"{muscle}_{movement}_{gender}_{sample_id}.npz"
np.savez(f'data/{filename}', hht=hht_matrix)
```

## Class Mapping

The 6 classes are mapped as follows:

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | M_full | Male, Full movement |
| 1 | M_half | Male, Half movement |
| 2 | M_invalid | Male, Invalid movement |
| 3 | F_full | Female, Full movement |
| 4 | F_half | Female, Half movement |
| 5 | F_invalid | Female, Invalid movement |

## Model Architecture

The CNN architecture remains unchanged:
- 3 Convolutional blocks (Conv2D + InstanceNorm + LeakyReLU)
- Channel progression: 1 → 64 → 128 → 256
- Spatial reduction: 256×256 → 128×128 → 64×64 → 32×32
- Global Average Pooling → 256-dim feature vector
- SVM classifier (RBF kernel, C=10.0)

## Troubleshooting

**Issue**: "No .npz files found"
- Check that your data directory contains `.npz` files
- Verify the path is correct

**Issue**: "Warning: file has wrong shape"
- Ensure all HHT matrices are exactly 256×256

**Issue**: "No valid training files found"
- Check filename format includes gender (M or F) and movement type
- Ensure filenames don't all contain 'test' (those are excluded from training)

**Issue**: CUDA out of memory
- Reduce `--batch_size` parameter
- Use `--cpu` flag to train on CPU

## Performance Tips

1. **GPU Acceleration**: Training is much faster with CUDA-enabled GPU
2. **Batch Size**: Larger batch sizes (32-64) work well for feature extraction
3. **Data Balance**: Ensure roughly equal samples per class for best results
4. **Validation Size**: 20-30% validation split is recommended

## Using Trained Models for Inference

After training, you can use the `inference.py` script to make predictions on new data:

### Single File Inference

```bash
python inference.py \
  --checkpoint ./checkpoints/final \
  --input ./new_data/BICEPS_unknown_001.npz \
  --output predictions.json
```

### Batch Inference on Directory

```bash
python inference.py \
  --checkpoint ./checkpoints/final \
  --input ./new_data/ \
  --output batch_predictions.json
```

### Command Line Options

- `--checkpoint`: Path to checkpoint (without extension)
- `--input`: Input file (.npz) or directory containing .npz files
- `--output`: Output JSON file for predictions (optional)
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--cpu`: Force CPU usage

### Example Output

```json
{
  "BICEPS_unknown_001.npz": {
    "prediction": "M_full",
    "confidence": 0.8523,
    "probabilities": {
      "M_full": 0.8523,
      "M_half": 0.0234,
      "M_invalid": 0.0145,
      "F_full": 0.0567,
      "F_half": 0.0321,
      "F_invalid": 0.0210
    }
  }
}
```

## Next Steps

After training:
1. Check validation accuracy and classification report
2. Review test file predictions in `test_predictions.json`
3. Use `inference.py` with the saved checkpoint for inference on new data
4. Fine-tune hyperparameters if needed (SVM C, kernel, etc.)
