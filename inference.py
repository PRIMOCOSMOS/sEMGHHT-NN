"""
Inference script for using a trained sEMG-HHT classifier.

This script loads a saved checkpoint and performs inference on new data.
"""

import torch
import numpy as np
import pickle
import argparse
import json
import os
import glob
from train import sEMGHHTClassifier, load_checkpoint
from sklearn.preprocessing import LabelEncoder


def load_model_for_inference(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to checkpoint (without extension)
        device: Device to use (default: auto-detect)
    
    Returns:
        classifier: Loaded classifier
        label_encoder: Label encoder for class names
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classifier, metadata = load_checkpoint(checkpoint_path, device)
    label_encoder = metadata.get('label_encoder')
    
    if label_encoder is None:
        # Create default label encoder
        all_classes = ['M_full', 'M_half', 'M_invalid', 'F_full', 'F_half', 'F_invalid']
        label_encoder = LabelEncoder()
        label_encoder.fit(all_classes)
    
    return classifier, label_encoder


def predict_single_file(classifier, label_encoder, npz_file: str, batch_size: int = 32):
    """
    Predict class for a single NPZ file.
    
    Args:
        classifier: Trained classifier
        label_encoder: Label encoder for class names
        npz_file: Path to .npz file
        batch_size: Batch size for feature extraction
    
    Returns:
        dict with prediction, confidence, and probabilities
    """
    # Load data
    data = np.load(npz_file)
    if 'hht' in data:
        hht_matrix = data['hht']
    else:
        hht_matrix = data[list(data.keys())[0]]
    
    if hht_matrix.shape != (256, 256):
        raise ValueError(f"Expected shape (256, 256), got {hht_matrix.shape}")
    
    # Add batch dimension
    X = hht_matrix[np.newaxis, :, :]
    
    # Predict
    y_pred = classifier.predict(X, batch_size)[0]
    y_proba = classifier.predict_proba(X, batch_size)[0]
    
    # Get SVM's class labels
    svm_classes = classifier.svm.classes_
    pred_idx = np.where(svm_classes == y_pred)[0][0]
    
    # Create result
    result = {
        'prediction': label_encoder.classes_[y_pred],
        'confidence': float(y_proba[pred_idx]),
        'probabilities': {
            label_encoder.classes_[cls]: float(y_proba[i])
            for i, cls in enumerate(svm_classes)
        }
    }
    
    return result


def predict_directory(classifier, label_encoder, data_dir: str, 
                      output_file: str = None, batch_size: int = 32):
    """
    Predict classes for all .npz files in a directory.
    
    Args:
        classifier: Trained classifier
        label_encoder: Label encoder for class names
        data_dir: Directory containing .npz files
        output_file: Optional JSON file to save predictions
        batch_size: Batch size for feature extraction
    
    Returns:
        dict mapping filenames to predictions
    """
    npz_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    predictions = {}
    
    print(f"Processing {len(npz_files)} files...")
    for npz_file in npz_files:
        try:
            result = predict_single_file(classifier, label_encoder, npz_file, batch_size)
            filename = os.path.basename(npz_file)
            predictions[filename] = result
            
            print(f"{filename}: {result['prediction']} (confidence: {result['confidence']:.4f})")
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            continue
    
    # Save predictions if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to: {output_file}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained sEMG-HHT classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint (without extension, e.g., checkpoints/final)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file (.npz) or directory containing .npz files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for predictions (optional)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cpu' if args.cpu else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    classifier, label_encoder = load_model_for_inference(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Run inference
    if os.path.isfile(args.input):
        # Single file
        print(f"\nProcessing file: {args.input}")
        result = predict_single_file(classifier, label_encoder, args.input, args.batch_size)
        
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nClass probabilities:")
        for class_name, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {prob:.4f}")
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({os.path.basename(args.input): result}, f, indent=2)
            print(f"\nPrediction saved to: {args.output}")
    
    elif os.path.isdir(args.input):
        # Directory
        print(f"\nProcessing directory: {args.input}")
        predictions = predict_directory(
            classifier, label_encoder, args.input, args.output, args.batch_size
        )
        print(f"\nProcessed {len(predictions)} files")
    
    else:
        raise ValueError(f"Input path does not exist: {args.input}")


if __name__ == '__main__':
    main()
