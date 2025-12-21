"""
sEMG-HHT CNN Classifier Training Script

This script trains a CNN-SVM classifier on sEMG Hilbert spectra for multi-label classification:
- Gender: Male (M), Female (F)
- Movement Quality: Full, Half, Invalid

Features:
- Checkpoint saving and resuming
- Automatic test file detection and post-training inference
- Accuracy testing on training and validation sets
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
    """Convolutional block with Conv2D, InstanceNorm, and LeakyReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 2, padding: int = 1,
                 leaky_slope: float = 0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x


class sEMGHHTEncoder(nn.Module):
    """
    CNN Encoder for sEMG-HHT matrix classification.
    
    Architecture:
    - Input: 1×256×256 (single-channel HHT matrix)
    - 3 ConvBlocks with increasing channels
    - Global Average Pooling
    - Output: Feature vector for SVM classification
    """
    
    def __init__(self, in_channels: int = 1, 
                 base_channels: int = 64,
                 num_layers: int = 3,
                 leaky_slope: float = 0.2):
        super(sEMGHHTEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Build convolutional layers
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
                leaky_slope=leaky_slope
            ))
            current_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate output feature dimension
        self.feature_dim = base_channels * (2 ** (num_layers - 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


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


def create_combined_label(gender: str, movement: str, label_encoder: LabelEncoder = None) -> int:
    """
    Create combined label from gender and movement.
    
    6 classes total:
    0: M_full, 1: M_half, 2: M_invalid
    3: F_full, 4: F_half, 5: F_invalid
    """
    combined = f"{gender}_{movement}"
    
    if label_encoder is None:
        # Define class mapping
        class_mapping = {
            'M_full': 0, 'M_half': 1, 'M_invalid': 2,
            'F_full': 3, 'F_half': 4, 'F_invalid': 5
        }
        return class_mapping.get(combined, -1)
    else:
        return label_encoder.transform([combined])[0]


def load_data_from_directory(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load HHT matrices from npz files in a directory.
    
    Returns:
        X: Training data
        y: Training labels
        filenames: List of training filenames
        test_files: List of test filenames (with 'test' in name)
    """
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    X_list = []
    y_list = []
    filenames = []
    test_files = []
    
    # Create label encoder for consistent label mapping
    all_classes = ['M_full', 'M_half', 'M_invalid', 'F_full', 'F_half', 'F_invalid']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_classes)
    
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
            
            # Create combined label
            combined = f"{labels['gender']}_{labels['movement']}"
            label = label_encoder.transform([combined])[0]
            
            X_list.append(hht_matrix)
            y_list.append(label)
            filenames.append(npz_file)
            
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue
    
    if len(X_list) == 0:
        raise ValueError("No valid training files found!")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    print(f"\nLoaded {len(X)} training samples")
    print(f"Found {len(test_files)} test files (will be used for inference after training)")
    print(f"\nClass distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y == i)
        print(f"  {class_name}: {count} samples")
    
    return X, y, filenames, test_files


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
    
    parser = argparse.ArgumentParser(description='Train sEMG-HHT CNN-SVM Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing .npz files with HHT matrices')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation set size (0-1)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
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
        device=device,
        resume=args.resume
    )
