# Production Training Scripts

This directory contains production-ready scripts for training and using the sEMG-HHT CNN-SVM classifier with real data.

## 🚀 Quick Start

```bash
# 1. Generate sample data (for testing)
python generate_sample_data.py --output_dir ./data --n_samples 30

# 2. Train the model
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# 3. Run inference on new data
python inference.py --checkpoint ./checkpoints/final --input ./new_data/
```

Or use the complete workflow script:

```bash
./example_workflow.sh
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `train.py` | Main training script with checkpoint support |
| `inference.py` | Inference script for using trained models |
| `generate_sample_data.py` | Generate synthetic data for testing |
| `example_workflow.sh` | Complete workflow example script |
| `TRAINING_GUIDE.md` | Detailed training documentation |

## 🎯 Key Features

### Training (`train.py`)

- **6-Class Multi-Label Classification**
  - Gender: Male (M), Female (F)
  - Movement Quality: Full, Half, Invalid
  
- **Checkpoint Management**
  - Automatic checkpoint saving
  - Resume training from last checkpoint
  - Saves model, scaler, SVM, and metadata

- **Automatic Test File Handling**
  - Files with 'test' in name are excluded from training
  - Used for inference after training complete
  - Predictions saved to JSON

- **Validation & Metrics**
  - Train/validation split
  - Accuracy and classification reports
  - Confusion matrix

### Inference (`inference.py`)

- Single file or batch inference
- Confidence scores and class probabilities
- JSON output format
- GPU/CPU support

### Data Generation (`generate_sample_data.py`)

- Creates synthetic 256×256 HHT matrices
- Proper filename formatting
- Configurable samples per class
- Test file generation

## 📊 Data Format

### File Naming Convention

Training files:
```
MUSCLENAME_movement_GENDER_###.npz
```

Examples:
- `BICEPS_fatiguetest_M_006.npz` → Male, Full movement
- `TRICEPS_half_F_012.npz` → Female, Half movement
- `FOREARM_invalid_M_003.npz` → Male, Invalid movement

Test files (unlabeled):
- `BICEPS_test_001.npz`
- `DELTOID_test_042.npz`

### File Content

Each `.npz` file contains a 256×256 HHT matrix:

```python
import numpy as np

# Save HHT matrix
hht_matrix = your_hht_computation(signal)  # Shape: (256, 256)
np.savez('BICEPS_fatiguetest_M_001.npz', hht=hht_matrix)
```

## 🔧 Usage Examples

### Example 1: Basic Training

```bash
python train.py \
    --data_dir ./my_data \
    --checkpoint_dir ./checkpoints \
    --batch_size 32 \
    --test_size 0.2
```

### Example 2: Resume Training

```bash
python train.py \
    --data_dir ./my_data \
    --checkpoint_dir ./checkpoints \
    --resume
```

### Example 3: Inference on Directory

```bash
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./new_data/ \
    --output predictions.json \
    --batch_size 32
```

### Example 4: Single File Inference

```bash
python inference.py \
    --checkpoint ./checkpoints/final \
    --input ./new_data/BICEPS_unknown_001.npz
```

## 📈 Expected Output

### Training Output

```
Using device: cuda
Data directory: ./data
Checkpoint directory: ./checkpoints

Loading data...
Loaded 120 training samples
Found 8 test files (will be used for inference after training)

Class distribution:
  F_full: 20 samples
  F_half: 20 samples
  F_invalid: 20 samples
  M_full: 20 samples
  M_half: 20 samples
  M_invalid: 20 samples

Training set: 96 samples
Validation set: 24 samples

============================================================
Training SVM Classifier
============================================================
Training Accuracy: 1.0000
Validation Accuracy: 0.8333

Test predictions saved to: ./checkpoints/test_predictions.json
```

### Inference Output

```
Prediction: M_full
Confidence: 0.8523

Class probabilities:
  M_full: 0.8523
  M_half: 0.0234
  M_invalid: 0.0145
  F_full: 0.0567
  F_half: 0.0321
  F_invalid: 0.0210
```

## 🏗️ Model Architecture

The CNN encoder (unchanged):
- **Layer 1**: Conv2D(1→64) + InstanceNorm + LeakyReLU + Stride(2)
- **Layer 2**: Conv2D(64→128) + InstanceNorm + LeakyReLU + Stride(2)
- **Layer 3**: Conv2D(128→256) + InstanceNorm + LeakyReLU + Stride(2)
- **Global Average Pooling**: 32×32×256 → 256
- **SVM Classifier**: RBF kernel, C=10.0, 6 classes

Spatial reduction: 256×256 → 128×128 → 64×64 → 32×32 → 256-dim vector

## 💾 Checkpoint Files

For each checkpoint, these files are saved:

| File | Content |
|------|---------|
| `*_encoder.pt` | CNN encoder weights (PyTorch) |
| `*_scaler.pkl` | Feature scaler (scikit-learn) |
| `*_svm.pkl` | Trained SVM classifier |
| `*_metadata.pkl` | Training metadata (epoch, history, etc.) |

Both `final_*` and `latest_*` checkpoints are saved.

## 🐛 Troubleshooting

**Problem**: "No .npz files found"
- Ensure data directory contains .npz files
- Check path is correct

**Problem**: "No valid training files found"
- Verify filename format: `MUSCLE_movement_GENDER_###.npz`
- Ensure not all files contain 'test' in name

**Problem**: CUDA out of memory
- Reduce `--batch_size` (try 16 or 8)
- Use `--cpu` flag

**Problem**: Low accuracy
- Check data quality and labels
- Ensure balanced class distribution
- Try adjusting SVM hyperparameters (C, kernel)

## 🔬 Class Mapping

| Class ID | Label | Gender | Movement |
|----------|-------|--------|----------|
| 0 | M_full | Male | Full |
| 1 | M_half | Male | Half |
| 2 | M_invalid | Male | Invalid |
| 3 | F_full | Female | Full |
| 4 | F_half | Female | Half |
| 5 | F_invalid | Female | Invalid |

## 📚 Additional Resources

- **Detailed Training Guide**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Jupyter Notebook**: See [semg_hht_cnn_classifier.ipynb](semg_hht_cnn_classifier.ipynb) for exploration
- **Main README**: See [README.md](README.md) for project overview

## 🤝 Contributing

When adding new features:
1. Maintain the CNN architecture (no structural changes)
2. Keep checkpoint compatibility
3. Update documentation
4. Test with sample data

## 📄 License

MIT License - See main [README.md](README.md) for details
