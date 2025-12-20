# sEMG-HHT CNN Classifier

A Convolutional Neural Network (CNN) based classifier for surface electromyography (sEMG) signals using Hilbert-Huang Transform (HHT) representation. This project is designed for multi-class classification tasks such as movement quality assessment and gender classification.

## 🎯 Overview

This project implements a deep learning pipeline that:
1. Takes 256×256 HHT matrices as input (derived from sEMG signals)
2. Extracts features using a 3-layer CNN encoder
3. Performs multi-class classification using SVM or end-to-end neural network

## 🏗️ Architecture

### CNN Encoder
- **3 Convolutional Layers**, each containing:
  - Conv2D (kernel=3, stride=2, padding=1)
  - Instance Normalization (IN)
  - LeakyReLU activation (slope=0.2)
- **Global Average Pooling** at the end
- Output: 256-dimensional feature vector

### Classifier Options
1. **SVM Classifier**: Extract features with CNN, classify with SVM (supports multi-class)
2. **End-to-End Model**: Fully trainable neural network with FC layers

## 📊 Classification Tasks

The model supports multi-dimensional classification:
- **Gender**: Male (M) vs Female (F)
- **Movement Quality**: Full movement vs Half movement vs Invalid movement
- **Combined 6-class classification** (Gender × Movement Quality)

## 🚀 Quick Start

### Option 0: Command-Line Training Script (Recommended for Real Data)

For training with real sEMG HHT data files:

```bash
# Generate sample data (for testing)
python generate_sample_data.py --output_dir ./data --n_samples 30

# Train the model
python train.py --data_dir ./data --checkpoint_dir ./checkpoints

# Resume training from checkpoint
python train.py --data_dir ./data --checkpoint_dir ./checkpoints --resume
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed usage instructions.

### Option 1: Run on Kaggle

1. Create a new Kaggle notebook
2. Upload `semg_hht_cnn_classifier.ipynb`
3. Enable GPU accelerator (Settings → Accelerator → GPU)
4. Run all cells

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/PRIMOCOSMOS/sEMGHHT-NN.git
cd sEMGHHT-NN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook semg_hht_cnn_classifier.ipynb
```

### Option 3: Run with Docker (GPU)

```bash
# Build and run with GPU support
docker-compose up --build

# Access Jupyter at http://localhost:8888
```

### Option 4: Run with Docker (CPU only)

```bash
# Build CPU-only image
docker build -f Dockerfile.cpu -t semg-hht-classifier-cpu .

# Run container
docker run -p 8888:8888 -v $(pwd)/data:/app/data semg-hht-classifier-cpu

# Access Jupyter at http://localhost:8888
```

## 📁 Project Structure

```
sEMGHHT-NN/
├── semg_hht_cnn_classifier.ipynb  # Main Jupyter notebook (demo/exploration)
├── train.py                        # Production training script with checkpoints
├── generate_sample_data.py         # Generate synthetic data for testing
├── TRAINING_GUIDE.md               # Detailed training guide
├── requirements.txt                # Python dependencies
├── Dockerfile                      # GPU-enabled Docker image
├── Dockerfile.cpu                  # CPU-only Docker image
├── docker-compose.yml              # Docker Compose configuration
├── data/                           # Data directory (create as needed)
├── checkpoints/                    # Model checkpoints (auto-created)
└── models/                         # Saved models (create as needed)
```

## 🔧 Usage with Real Data

### Training Script Features

The `train.py` script provides:
- **Checkpoint saving**: Automatically saves model state at regular intervals
- **Resume training**: Continue from the last checkpoint if interrupted
- **6-class classification**: Gender (M/F) × Movement quality (Full/Half/Invalid)
- **Automatic test file detection**: Files with 'test' in name are used for inference
- **Validation metrics**: Accuracy and classification reports after training

### Data File Format

```python
# File naming convention:
# MUSCLENAME_movement_GENDER_###.npz

# Examples:
# BICEPS_fatiguetest_M_006.npz  -> Male, Full movement
# TRICEPS_half_F_012.npz        -> Female, Half movement
# FOREARM_invalid_M_003.npz     -> Male, Invalid movement
# DELTOID_test_001.npz          -> Unlabeled test file

# Each .npz file should contain a 256×256 HHT matrix
import numpy as np
hht_matrix = compute_your_hht(signal)  # 256×256 array
np.savez('BICEPS_fatiguetest_M_001.npz', hht=hht_matrix)
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete documentation.

### 1. Prepare HHT Matrices (Jupyter Notebook)

If using the Jupyter notebook for exploration:

```python
from semg_hht_cnn_classifier import compute_hht_matrix

# Load your sEMG signal
signal = load_your_semg_data()  # 1D numpy array
fs = 1000  # Sampling frequency in Hz

# Compute HHT matrix
hht_matrix = compute_hht_matrix(signal, fs, matrix_size=256)
```

### 2. Train the Classifier

```python
# Prepare data
X = np.stack([compute_hht_matrix(s, fs) for s in signals])  # (n_samples, 256, 256)
y = np.array(labels)  # (n_samples,)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
classifier = sEMGHHTClassifier(device=device)
classifier.fit(X_train, y_train)

# Evaluate
results = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn 0.24+
- NumPy 1.19+
- Matplotlib 3.4+
- (Optional) PyEMD for real HHT computation

## 🐳 Docker Deployment

### GPU Version (NVIDIA GPU required)
```bash
# Requires nvidia-docker2
docker-compose up --build
```

### CPU Version
```bash
docker build -f Dockerfile.cpu -t semg-classifier-cpu .
docker run -p 8888:8888 semg-classifier-cpu
```

## 📈 Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_channels` | 64 | Number of channels in first conv layer |
| `num_layers` | 3 | Number of convolutional blocks |
| `svm_kernel` | 'rbf' | SVM kernel type |
| `svm_C` | 10.0 | SVM regularization parameter |
| `dropout_rate` | 0.5 | Dropout rate for end-to-end model |

## 🔬 Methodology

### HHT (Hilbert-Huang Transform)
The HHT is a time-frequency analysis method that:
1. Decomposes signal into Intrinsic Mode Functions (IMFs) using EMD
2. Applies Hilbert transform to each IMF
3. Creates time-frequency representation (Hilbert spectrum)

### CNN Feature Extraction
The 3-layer CNN progressively:
- Reduces spatial dimensions: 256 → 128 → 64 → 32
- Increases feature channels: 1 → 64 → 128 → 256
- Global Average Pooling produces fixed 256-dim vector

## 📖 References

- Huang, N. E., et al. (1998). "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis."
- sEMG signal processing for gesture recognition

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
