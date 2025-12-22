"""
sEMG-HHT Dual Classifier Training Script

This script trains TWO separate classifiers on sEMG Hilbert spectra:
1. SVM for gender classification: Male (M), Female (F) - 2 classes
2. Deep Learning CNN for action quality: Full, Half, Invalid - 3 classes

Features:
- Checkpoint saving and resuming
- Automatic test file detection and post-training inference
- Accuracy testing on training and validation sets
- Improved CNN architecture with better convergence
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
import glob
import re
from typing import Tuple, Optional, Dict, List
import warnings
import json

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class ConvBlock(nn.Module):
    """Convolutional block with Conv2D, BatchNorm, and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 2, padding: int = 1,
                 leaky_slope: float = 0.2, use_batchnorm: bool = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding,
                              bias=not use_batchnorm)
        # Use BatchNorm instead of InstanceNorm for better training stability
        self.norm = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class sEMGHHTEncoder(nn.Module):
    """
    Improved CNN Encoder for sEMG-HHT matrix classification.
    
    Architecture:
    - Input: 1×256×256 (single-channel HHT matrix)
    - Configurable number of ConvBlocks with increasing channels
    - Global Average Pooling
    - Output: Feature vector for classification
    
    Improvements:
    - Configurable depth for better feature extraction
    - BatchNormalization for training stability
    - Proper weight initialization
    - Input validation to ensure proper dimensions
    """
    
    def __init__(self, in_channels: int = 1, 
                 base_channels: int = 64,
                 num_layers: int = 5,
                 leaky_slope: float = 0.2,
                 use_batchnorm: bool = True):
        super(sEMGHHTEncoder, self).__init__()
        
        # Validate input parameters
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if num_layers > 8:
            raise ValueError(f"num_layers must be at most 8 for 256x256 input (each layer halves spatial dims), got {num_layers}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Build convolutional layers with increasing channels
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(ConvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                leaky_slope=leaky_slope,
                use_batchnorm=use_batchnorm
            ))
            current_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate output feature dimension
        self.feature_dim = base_channels * (2 ** (num_layers - 1))
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


class ActionQualityCNNClassifier(nn.Module):
    """
    Deep Learning CNN Classifier for Action Quality (Full/Half/Invalid).
    
    This is an end-to-end trainable neural network with:
    - Configurable CNN encoder (flexible number of layers)
    - Dropout for regularization
    - Fully connected classification head (adaptive to encoder size)
    - 3 output classes: Full, Half, Invalid
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 base_channels: int = 64,
                 num_encoder_layers: int = 5,
                 dropout_rate: float = 0.5,
                 num_classes: int = 3):
        super(ActionQualityCNNClassifier, self).__init__()
        
        # Validate parameters
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        # Encoder
        self.encoder = sEMGHHTEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_encoder_layers,
            use_batchnorm=True
        )
        
        feature_dim = self.encoder.get_feature_dim()
        
        # Adaptive classification head: scale intermediate layers based on feature_dim
        # For small feature_dim (shallow networks), use smaller intermediate layers
        # For large feature_dim (deep networks), use larger intermediate layers
        hidden_dim_1 = min(max(256, feature_dim // 4), 1024)
        hidden_dim_2 = min(max(128, feature_dim // 8), 512)
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim_2, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier_weights()
    
    def _initialize_classifier_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.encoder(x)


class GenderSVMClassifier:
    """
    SVM Classifier for Gender Classification (Male/Female).
    
    Uses CNN features with SVM for robust gender classification.
    2 output classes: M, F
    """
    
    def __init__(self,
                 encoder: Optional[sEMGHHTEncoder] = None,
                 svm_kernel: str = 'rbf',
                 svm_C: float = 10.0,
                 svm_gamma: str = 'scale',
                 device: torch.device = torch.device('cpu')):
        # Validate device parameter
        if isinstance(device, str):
            device = torch.device(device)
        elif not isinstance(device, torch.device):
            raise TypeError(f"device must be torch.device or str, got {type(device)}")
        
        self.device = device
        
        if encoder is None:
            self.encoder = sEMGHHTEncoder(
                in_channels=1,
                base_channels=64,
                num_layers=4,
                use_batchnorm=True
            )
        else:
            self.encoder = encoder
        
        self.encoder.to(self.device)
        self.encoder.eval()  # Freeze encoder for gender classification
        
        self.scaler = StandardScaler()
        self.svm = SVC(
            kernel=svm_kernel,
            C=svm_C,
            gamma=svm_gamma,
            decision_function_shape='ovr',
            probability=True
        )
        
        self._is_fitted = False
    
    def extract_features(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Extract features from HHT matrices using the CNN encoder."""
        self.encoder.eval()
        
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        
        features_list = []
        n_samples = X.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                batch_features = self.encoder(batch)
                features_list.append(batch_features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """Fit the classifier."""
        print("Extracting features for gender classification...")
        features = self.extract_features(X, batch_size)
        
        print("Normalizing features...")
        features_scaled = self.scaler.fit_transform(features)
        
        print("Training SVM for gender classification...")
        self.svm.fit(features_scaled, y)
        
        self._is_fitted = True
        print("Gender classifier training complete!")
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predict gender labels."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        features = self.extract_features(X, batch_size)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict(features_scaled)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predict gender probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        features = self.extract_features(X, batch_size)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict_proba(features_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> dict:
        """Evaluate the classifier on test data."""
        y_pred = self.predict(X, batch_size)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }


class sEMGHHTClassifier:
    """
    Complete classification pipeline combining CNN encoder and SVM classifier.
    """
    
    def __init__(self, 
                 encoder: Optional[sEMGHHTEncoder] = None,
                 svm_kernel: str = 'rbf',
                 svm_C: float = 10.0,
                 svm_gamma: str = 'scale',
                 device: torch.device = torch.device('cpu')):
        self.device = device
        
        if encoder is None:
            self.encoder = sEMGHHTEncoder(
                in_channels=1, 
                base_channels=64, 
                num_layers=3
            )
        else:
            self.encoder = encoder
        
        self.encoder.to(self.device)
        
        self.scaler = StandardScaler()
        self.svm = SVC(
            kernel=svm_kernel,
            C=svm_C,
            gamma=svm_gamma,
            decision_function_shape='ovr',
            probability=True
        )
        
        self._is_fitted = False
    
    def extract_features(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Extract features from HHT matrices using the CNN encoder."""
        self.encoder.eval()
        
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        
        features_list = []
        n_samples = X.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                batch_features = self.encoder(batch)
                features_list.append(batch_features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """Fit the classifier."""
        print("Extracting features from training data...")
        features = self.extract_features(X, batch_size)
        
        print("Normalizing features...")
        features_scaled = self.scaler.fit_transform(features)
        
        print("Training SVM classifier...")
        self.svm.fit(features_scaled, y)
        
        self._is_fitted = True
        print("Training complete!")
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        features = self.extract_features(X, batch_size)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict(features_scaled)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting")
        
        features = self.extract_features(X, batch_size)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict_proba(features_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                 batch_size: int = 32) -> dict:
        """Evaluate the classifier on test data."""
        y_pred = self.predict(X, batch_size)
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }


def normalize_hht_matrix(hht_matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize HHT matrix to [0, 1] range.
    
    Args:
        hht_matrix: Input HHT matrix
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized matrix in [0, 1] range
    """
    min_val = hht_matrix.min()
    max_val = hht_matrix.max()
    
    # Avoid division by zero with epsilon
    if max_val - min_val < epsilon:
        # If matrix is nearly constant, return zeros
        return np.zeros_like(hht_matrix, dtype=np.float32)
    
    return ((hht_matrix - min_val) / (max_val - min_val + epsilon)).astype(np.float32)


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract labels.
    
    Format: MUSCLENAME_movement_GENDER_###.npz
    Examples:
    - BICEPS_fatiguetest_M_006.npz -> {gender: M, movement: full}
    - TRICEPS_half_F_012.npz -> {gender: F, movement: half}
    - BICEPS_invalid_M_003.npz -> {gender: M, movement: invalid}
    
    Returns None if filename starts with 'Test' (unlabeled test data)
    Note: 'fatiguetest' is a valid movement type, not a test file
    """
    basename = os.path.basename(filename)
    
    # Skip files that start with 'Test' (case-insensitive)
    # Examples: Test1_1_015.npz, test_sample.npz
    if basename.lower().startswith('test'):
        return None
    
    # Extract gender (M or F)
    gender_match = re.search(r'[_-]([MF])[_-]', basename)
    if not gender_match:
        return None
    gender = gender_match.group(1)
    
    # Extract movement quality
    basename_lower = basename.lower()
    if 'fatiguetest' in basename_lower or 'full' in basename_lower:
        movement = 'full'
    elif 'half' in basename_lower:
        movement = 'half'
    elif 'invalid' in basename_lower or 'wrong' in basename_lower:
        movement = 'invalid'
    else:
        return None
    
    return {'gender': gender, 'movement': movement}


def load_data_from_directory(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], LabelEncoder, LabelEncoder]:
    """
    Load HHT matrices from npz files in a directory.
    
    Returns:
        X: Training data
        y_gender: Gender labels (0=M, 1=F)
        y_action: Action quality labels (0=full, 1=half, 2=invalid)
        filenames: List of training filenames
        test_files: List of test filenames (with 'test' in name)
        gender_encoder: LabelEncoder for gender
        action_encoder: LabelEncoder for action quality
    """
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    X_list = []
    y_gender_list = []
    y_action_list = []
    filenames = []
    test_files = []
    
    # Create label encoders
    gender_classes = ['M', 'F']
    action_classes = ['full', 'half', 'invalid']
    
    gender_encoder = LabelEncoder()
    gender_encoder.fit(gender_classes)
    
    action_encoder = LabelEncoder()
    action_encoder.fit(action_classes)
    
    for npz_file in npz_files:
        labels = parse_filename(npz_file)
        
        # Skip files with 'test' in name (for post-training inference)
        if labels is None:
            test_files.append(npz_file)
            continue
        
        # Load data
        try:
            data = np.load(npz_file)
            # Assume the HHT matrix is stored with key 'hht' or is the first array
            if 'hht' in data:
                hht_matrix = data['hht']
            else:
                # Use first array in the file
                hht_matrix = data[list(data.keys())[0]]
            
            # Ensure correct size (256x256)
            if hht_matrix.shape != (256, 256):
                print(f"Warning: {npz_file} has shape {hht_matrix.shape}, expected (256, 256). Skipping.")
                continue
            
            # Normalize data to [0, 1] range for better training
            hht_matrix = normalize_hht_matrix(hht_matrix)
            
            # Create separate labels
            gender_label = gender_encoder.transform([labels['gender']])[0]
            action_label = action_encoder.transform([labels['movement']])[0]
            
            X_list.append(hht_matrix)
            y_gender_list.append(gender_label)
            y_action_list.append(action_label)
            filenames.append(npz_file)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No valid training files found!")
    
    X = np.array(X_list, dtype=np.float32)
    y_gender = np.array(y_gender_list)
    y_action = np.array(y_action_list)
    
    print(f"\nLoaded {len(X)} training samples")
    print(f"Found {len(test_files)} test files (will be used for inference after training)")
    
    print(f"\nGender distribution:")
    for i, class_name in enumerate(gender_encoder.classes_):
        count = np.sum(y_gender == i)
        print(f"  {class_name}: {count} samples")
    
    print(f"\nAction quality distribution:")
    for i, class_name in enumerate(action_encoder.classes_):
        count = np.sum(y_action == i)
        print(f"  {class_name}: {count} samples")
    
    return X, y_gender, y_action, filenames, test_files, gender_encoder, action_encoder


def save_action_quality_checkpoint(model: ActionQualityCNNClassifier,
                                  optimizer: torch.optim.Optimizer,
                                  checkpoint_path: str,
                                  epoch: int = 0,
                                  history: dict = None,
                                  label_encoder: LabelEncoder = None,
                                  best_val_acc: float = 0.0):
    """
    Save action quality model checkpoint with atomic write and disk space check.
    
    Uses temporary file and rename to ensure atomic write operation.
    Checks available disk space before saving.
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    final_path = f"{checkpoint_path}_action_quality.pt"
    temp_path = f"{final_path}.tmp"
    
    # Check available disk space (estimate needed: ~100MB per checkpoint)
    try:
        import shutil
        stat = shutil.disk_usage(checkpoint_dir if checkpoint_dir else '.')
        available_gb = stat.free / (1024 ** 3)
        if available_gb < 0.5:  # Less than 500MB free
            print(f"Warning: Low disk space ({available_gb:.2f} GB free). Skipping checkpoint save.")
            return
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history or {},
        'label_encoder': label_encoder,
        'best_val_acc': best_val_acc
    }
    
    try:
        # Save to temporary file first
        torch.save(checkpoint, temp_path)
        # Atomic rename (on most filesystems)
        os.replace(temp_path, final_path)
        print(f"Action quality checkpoint saved at epoch {epoch}: {final_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise


def save_gender_checkpoint(classifier: GenderSVMClassifier,
                           checkpoint_path: str,
                           label_encoder: LabelEncoder = None):
    """Save gender classifier checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save encoder
    torch.save(classifier.encoder.state_dict(), f"{checkpoint_path}_gender_encoder.pt")
    
    # Save scaler and SVM
    with open(f"{checkpoint_path}_gender_scaler.pkl", 'wb') as f:
        pickle.dump(classifier.scaler, f)
    
    with open(f"{checkpoint_path}_gender_svm.pkl", 'wb') as f:
        pickle.dump(classifier.svm, f)
    
    # Save metadata
    metadata = {
        'is_fitted': classifier._is_fitted,
        'label_encoder': label_encoder
    }
    with open(f"{checkpoint_path}_gender_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Gender classifier checkpoint saved: {checkpoint_path}_gender_*")


def save_checkpoint(classifier: sEMGHHTClassifier, 
                   checkpoint_path: str,
                   epoch: int = 0,
                   history: dict = None,
                   label_encoder: LabelEncoder = None):
    """Save model checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save encoder
    torch.save(classifier.encoder.state_dict(), f"{checkpoint_path}_encoder.pt")
    
    # Save scaler and SVM
    with open(f"{checkpoint_path}_scaler.pkl", 'wb') as f:
        pickle.dump(classifier.scaler, f)
    
    with open(f"{checkpoint_path}_svm.pkl", 'wb') as f:
        pickle.dump(classifier.svm, f)
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'history': history or {},
        'is_fitted': classifier._is_fitted,
        'label_encoder': label_encoder
    }
    with open(f"{checkpoint_path}_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}_*")


def load_checkpoint(checkpoint_path: str, 
                   device: torch.device = torch.device('cpu')) -> Tuple[sEMGHHTClassifier, dict]:
    """Load model checkpoint."""
    classifier = sEMGHHTClassifier(device=device)
    
    # Load encoder
    classifier.encoder.load_state_dict(
        torch.load(f"{checkpoint_path}_encoder.pt", map_location=device)
    )
    
    # Load scaler and SVM
    with open(f"{checkpoint_path}_scaler.pkl", 'rb') as f:
        classifier.scaler = pickle.load(f)
    
    with open(f"{checkpoint_path}_svm.pkl", 'rb') as f:
        classifier.svm = pickle.load(f)
    
    # Load metadata
    with open(f"{checkpoint_path}_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    classifier._is_fitted = metadata.get('is_fitted', True)
    
    print(f"Checkpoint loaded from epoch {metadata.get('epoch', 0)}: {checkpoint_path}_*")
    return classifier, metadata


def train_with_checkpoints(data_dir: str,
                          checkpoint_dir: str = './checkpoints',
                          test_size: float = 0.2,
                          batch_size: int = 32,
                          epochs: int = 100,
                          learning_rate: float = 0.001,
                          num_encoder_layers: int = 5,
                          base_channels: int = 64,
                          device: torch.device = None,
                          resume: bool = False):
    """
    Train dual classifiers with checkpoint saving and validation.
    
    Trains TWO separate classifiers:
    1. Deep Learning CNN for Action Quality (Full/Half/Invalid)
    2. SVM for Gender Classification (M/F)
    
    Args:
        data_dir: Directory containing training data (.npz files)
        checkpoint_dir: Directory to save checkpoints
        test_size: Proportion of data to use for validation (0-1)
        batch_size: Batch size for training
        epochs: Number of training epochs for action quality CNN
        learning_rate: Learning rate for CNN training
        num_encoder_layers: Number of convolutional layers in encoder (1-8)
        base_channels: Base number of channels in first conv layer (default: 64)
        device: Device to use (cuda/cpu), auto-detected if None
        resume: If True, resume from latest checkpoint if available
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    X, y_gender, y_action, filenames, test_files, gender_encoder, action_encoder = load_data_from_directory(data_dir)
    
    # Split data (use same split for both tasks)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_gender_train, y_gender_val, y_action_train, y_action_val = train_test_split(
        X, y_gender, y_action, test_size=test_size, random_state=SEED, stratify=y_action
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # ========================================================================
    # PART 1: Train Gender Classifier (SVM)
    # ========================================================================
    print("\n" + "="*60)
    print("PART 1: Training Gender Classifier (SVM)")
    print("="*60)
    
    gender_classifier = GenderSVMClassifier(
        encoder=None,
        svm_kernel='rbf',
        svm_C=10.0,
        svm_gamma='scale',
        device=device
    )
    
    gender_classifier.fit(X_train, y_gender_train, batch_size=batch_size)
    
    # Evaluate gender classifier
    print("\nEvaluating gender classifier on validation set...")
    gender_val_results = gender_classifier.evaluate(X_val, y_gender_val, batch_size)
    print(f"Gender Validation Accuracy: {gender_val_results['accuracy']:.4f}")
    print("\nGender Classification Report:")
    print(gender_val_results['classification_report'])
    
    # Save gender classifier
    gender_checkpoint_path = os.path.join(checkpoint_dir, 'final')
    save_gender_checkpoint(gender_classifier, gender_checkpoint_path, gender_encoder)
    
    # ========================================================================
    # PART 2: Train Action Quality Classifier (Deep Learning CNN)
    # ========================================================================
    print("\n" + "="*60)
    print("PART 2: Training Action Quality Classifier (Deep Learning CNN)")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  - Encoder Layers: {num_encoder_layers}")
    print(f"  - Base Channels: {base_channels}")
    print(f"  - Feature Dimension: {base_channels * (2 ** (num_encoder_layers - 1))}")
    
    # Initialize action quality model
    action_model = ActionQualityCNNClassifier(
        in_channels=1,
        base_channels=base_channels,
        num_encoder_layers=num_encoder_layers,
        dropout_rate=0.5,
        num_classes=3
    ).to(device)
    
    # Create data loaders
    if X_train.ndim == 3:
        X_train_expanded = X_train[:, np.newaxis, :, :]
        X_val_expanded = X_val[:, np.newaxis, :, :]
    else:
        X_train_expanded = X_train
        X_val_expanded = X_val
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_expanded, dtype=torch.float32),
        torch.tensor(y_action_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val_expanded, dtype=torch.float32),
        torch.tensor(y_action_val, dtype=torch.long)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(action_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print("This may take a while. Training progress will be displayed.\n")
    
    for epoch in range(epochs):
        # Training phase
        action_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = action_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        action_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = action_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Print progress every 5 epochs or when achieving best
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            action_checkpoint_path = os.path.join(checkpoint_dir, 'best')
            save_action_quality_checkpoint(
                action_model, optimizer, action_checkpoint_path,
                epoch, history, action_encoder, best_val_acc
            )
            print(f"  ⭐ New best model! Val Acc: {val_acc:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            action_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}')
            save_action_quality_checkpoint(
                action_model, optimizer, action_checkpoint_path,
                epoch, history, action_encoder, best_val_acc
            )
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final')
    save_action_quality_checkpoint(
        action_model, optimizer, final_checkpoint_path,
        epochs-1, history, action_encoder, best_val_acc
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Action Quality Val Acc: {best_val_acc:.4f} (epoch {best_epoch+1})")
    print(f"Gender Classifier Val Acc: {gender_val_results['accuracy']:.4f}")
    
    # ========================================================================
    # PART 3: Run Inference on Test Files
    # ========================================================================
    if len(test_files) > 0:
        print("\n" + "="*60)
        print("Running Inference on Test Files")
        print("="*60)
        
        X_test_list = []
        valid_test_files = []
        
        for test_file in test_files:
            try:
                data = np.load(test_file)
                if 'hht' in data:
                    hht_matrix = data['hht']
                else:
                    hht_matrix = data[list(data.keys())[0]]
                
                if hht_matrix.shape == (256, 256):
                    # Normalize using the same function as training data
                    hht_matrix = normalize_hht_matrix(hht_matrix)
                    X_test_list.append(hht_matrix)
                    valid_test_files.append(test_file)
            except Exception as e:
                print(f"Error loading {test_file}: {e}")
        
        if len(X_test_list) > 0:
            X_test = np.array(X_test_list, dtype=np.float32)
            
            # Predict gender
            y_gender_pred = gender_classifier.predict(X_test, batch_size)
            y_gender_proba = gender_classifier.predict_proba(X_test, batch_size)
            
            # Predict action quality
            X_test_tensor = torch.tensor(X_test[:, np.newaxis, :, :], dtype=torch.float32).to(device)
            action_model.eval()
            with torch.no_grad():
                action_outputs = action_model(X_test_tensor)
                action_proba = torch.softmax(action_outputs, dim=1).cpu().numpy()
                y_action_pred = action_outputs.argmax(dim=1).cpu().numpy()
            
            print(f"\nPredictions for {len(valid_test_files)} test files:")
            predictions = {}
            for i, filename in enumerate(valid_test_files):
                gender_name = gender_encoder.classes_[y_gender_pred[i]]
                action_name = action_encoder.classes_[y_action_pred[i]]
                
                gender_conf = y_gender_proba[i][y_gender_pred[i]]
                action_conf = action_proba[i][y_action_pred[i]]
                
                print(f"{os.path.basename(filename)}:")
                print(f"  Gender: {gender_name} (confidence: {gender_conf:.4f})")
                print(f"  Action: {action_name} (confidence: {action_conf:.4f})")
                
                predictions[os.path.basename(filename)] = {
                    'gender': {
                        'prediction': gender_name,
                        'confidence': float(gender_conf),
                        'probabilities': {
                            gender_encoder.classes_[j]: float(y_gender_proba[i][j])
                            for j in range(len(gender_encoder.classes_))
                        }
                    },
                    'action_quality': {
                        'prediction': action_name,
                        'confidence': float(action_conf),
                        'probabilities': {
                            action_encoder.classes_[j]: float(action_proba[i][j])
                            for j in range(len(action_encoder.classes_))
                        }
                    }
                }
            
            # Save predictions
            predictions_file = os.path.join(checkpoint_dir, 'test_predictions.json')
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"\nPredictions saved to: {predictions_file}")
    
    return action_model, gender_classifier, history


def train_with_checkpoints_old(data_dir: str,
                          checkpoint_dir: str = './checkpoints',
                          test_size: float = 0.2,
                          batch_size: int = 32,
                          device: torch.device = None,
                          resume: bool = False):
    """
    Train classifier with checkpoint saving and validation.
    
    Note: SVM doesn't have epochs like neural networks. We train it once
    and save the trained model as a checkpoint.
    
    Args:
        data_dir: Directory containing training data (.npz files)
        checkpoint_dir: Directory to save checkpoints
        test_size: Proportion of data to use for validation (0-1)
        batch_size: Batch size for feature extraction
        device: Device to use (cuda/cpu), auto-detected if None
        resume: If True, resume from latest checkpoint if available
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    X, y, filenames, test_files = load_data_from_directory(data_dir)
    
    # Create label encoder for class names
    all_classes = ['M_full', 'M_half', 'M_invalid', 'F_full', 'F_half', 'F_invalid']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Check for existing checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest')
    if resume and os.path.exists(f"{latest_checkpoint}_encoder.pt"):
        print("\nResuming from checkpoint...")
        classifier, metadata = load_checkpoint(latest_checkpoint, device)
        history = metadata.get('history', {})
    else:
        print("\nStarting new training...")
        classifier = sEMGHHTClassifier(
            encoder=None,
            svm_kernel='rbf',
            svm_C=10.0,
            svm_gamma='scale',
            device=device
        )
        history = {'train_acc': [], 'val_acc': []}
    
    # Train the classifier (SVM trains in one shot)
    if not classifier._is_fitted:
        print("\n" + "="*60)
        print("Training SVM Classifier")
        print("="*60)
        classifier.fit(X_train, y_train, batch_size=batch_size)
        
        # Evaluate on training set
        print("\nEvaluating on training set...")
        train_results = classifier.evaluate(X_train, y_train, batch_size)
        print(f"Training Accuracy: {train_results['accuracy']:.4f}")
        history['train_acc'].append(train_results['accuracy'])
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_results = classifier.evaluate(X_val, y_val, batch_size)
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        history['val_acc'].append(val_results['accuracy'])
        
        print("\nClassification Report (Validation):")
        print(val_results['classification_report'])
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'final')
        save_checkpoint(classifier, checkpoint_path, epoch=1, 
                       history=history, label_encoder=label_encoder)
        
        # Also save as latest
        latest_path = os.path.join(checkpoint_dir, 'latest')
        save_checkpoint(classifier, latest_path, epoch=1, 
                       history=history, label_encoder=label_encoder)
    
    # Run inference on test files
    if len(test_files) > 0:
        print("\n" + "="*60)
        print("Running Inference on Test Files")
        print("="*60)
        
        X_test_list = []
        valid_test_files = []
        
        for test_file in test_files:
            try:
                data = np.load(test_file)
                if 'hht' in data:
                    hht_matrix = data['hht']
                else:
                    hht_matrix = data[list(data.keys())[0]]
                
                if hht_matrix.shape == (256, 256):
                    X_test_list.append(hht_matrix)
                    valid_test_files.append(test_file)
            except Exception as e:
                print(f"Error loading {test_file}: {e}")
        
        if len(X_test_list) > 0:
            X_test = np.array(X_test_list, dtype=np.float32)
            y_test_pred = classifier.predict(X_test, batch_size)
            y_test_proba = classifier.predict_proba(X_test, batch_size)
            
            # Get SVM's class labels (these are the actual class indices it was trained on)
            svm_classes = classifier.svm.classes_
            
            print(f"\nPredictions for {len(valid_test_files)} test files:")
            for i, (filename, pred, proba) in enumerate(zip(valid_test_files, y_test_pred, y_test_proba)):
                # Find which position in proba corresponds to the predicted class
                pred_idx = np.where(svm_classes == pred)[0][0]
                class_name = label_encoder.classes_[pred]
                confidence = proba[pred_idx]
                print(f"{os.path.basename(filename)}: {class_name} (confidence: {confidence:.4f})")
            
            # Save predictions
            predictions_file = os.path.join(checkpoint_dir, 'test_predictions.json')
            predictions = {}
            for f, p, proba in zip(valid_test_files, y_test_pred, y_test_proba):
                pred_idx = np.where(svm_classes == p)[0][0]
                # Create probability dict with actual class names
                prob_dict = {label_encoder.classes_[cls]: float(proba[i]) 
                           for i, cls in enumerate(svm_classes)}
                predictions[os.path.basename(f)] = {
                    'prediction': label_encoder.classes_[p],
                    'confidence': float(proba[pred_idx]),
                    'probabilities': prob_dict
                }
            
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"\nPredictions saved to: {predictions_file}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    
    return classifier, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sEMG-HHT Dual Classifier (Gender SVM + Action Quality CNN)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing .npz files with HHT matrices')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation set size (0-1)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16, optimized for 256x256 images with limited memory)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for action quality CNN')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for action quality CNN training')
    parser.add_argument('--num_encoder_layers', type=int, default=5,
                       help='Number of convolutional layers in CNN encoder (1-8, default: 5)')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in first conv layer (default: 64)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    train_with_checkpoints(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_encoder_layers=args.num_encoder_layers,
        base_channels=args.base_channels,
        device=device,
        resume=args.resume
    )
