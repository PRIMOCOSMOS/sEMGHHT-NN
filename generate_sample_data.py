"""
Generate sample sEMG HHT data for testing the training script.

Creates synthetic 256x256 HHT matrices with appropriate filenames.
"""

import numpy as np
import os
from pathlib import Path


def generate_hht_matrix(matrix_size=256, gender='M', movement='full', seed=None):
    """
    Generate a synthetic HHT matrix with different patterns for different classes.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create time and frequency axes
    t = np.linspace(0, 1, matrix_size)
    f = np.linspace(0, 100, matrix_size)
    T, F = np.meshgrid(t, f)
    
    # Different patterns for different genders and movements
    if gender == 'M':
        base_amplitude = 1.0
        freq_offset = 0
    else:  # F
        base_amplitude = 0.8
        freq_offset = 10
    
    if movement == 'full':
        noise_level = 0.1
        freq_spread = 0.3
        n_components = 3
    elif movement == 'half':
        noise_level = 0.2
        freq_spread = 0.5
        n_components = 2
    else:  # invalid
        noise_level = 0.4
        freq_spread = 0.8
        n_components = 1
    
    # Generate base HHT pattern
    center_freq = 30 + freq_offset + np.random.uniform(-5, 5)
    center_time = 0.5 + np.random.uniform(-0.1, 0.1)
    
    hht_matrix = base_amplitude * np.exp(
        -((F - center_freq) ** 2) / (2 * (10 * freq_spread) ** 2)
        -((T - center_time) ** 2) / (2 * 0.2 ** 2)
    )
    
    # Add secondary components
    for _ in range(n_components - 1):
        sec_freq = np.random.uniform(10, 80)
        sec_time = np.random.uniform(0.2, 0.8)
        sec_amp = base_amplitude * np.random.uniform(0.2, 0.4)
        
        hht_matrix += sec_amp * np.exp(
            -((F - sec_freq) ** 2) / (2 * (8 * freq_spread) ** 2)
            -((T - sec_time) ** 2) / (2 * 0.15 ** 2)
        )
    
    # Add noise
    noise = noise_level * np.random.randn(matrix_size, matrix_size)
    hht_matrix += noise
    
    # Normalize to [0, 1]
    hht_matrix = (hht_matrix - hht_matrix.min()) / (hht_matrix.max() - hht_matrix.min() + 1e-8)
    
    return hht_matrix.astype(np.float32)


def generate_dataset(output_dir, n_samples_per_class=20, n_test_samples=10):
    """
    Generate a complete dataset with all 6 classes plus test samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    muscles = ['BICEPS', 'TRICEPS', 'FOREARM', 'DELTOID']
    genders = ['M', 'F']
    movements = ['fatiguetest', 'half', 'invalid']
    
    sample_count = 0
    
    print(f"Generating dataset in {output_dir}...")
    print(f"Samples per class: {n_samples_per_class}")
    print(f"Test samples: {n_test_samples}")
    
    # Generate training samples
    for gender in genders:
        for movement in movements:
            for i in range(n_samples_per_class):
                muscle = np.random.choice(muscles)
                sample_id = f"{sample_count:03d}"
                
                filename = f"{muscle}_{movement}_{gender}_{sample_id}.npz"
                filepath = os.path.join(output_dir, filename)
                
                hht_matrix = generate_hht_matrix(
                    matrix_size=256,
                    gender=gender,
                    movement=movement if movement != 'fatiguetest' else 'full',
                    seed=sample_count
                )
                
                np.savez(filepath, hht=hht_matrix)
                sample_count += 1
    
    print(f"Generated {sample_count} training samples")
    
    # Generate test samples (unlabeled, filename starts with 'Test')
    test_count = 0
    for i in range(n_test_samples):
        sample_id = f"{i + 1}_{test_count + 1:03d}"
        
        filename = f"Test{sample_id}.npz"
        filepath = os.path.join(output_dir, filename)
        
        # Randomly pick a class for generation (but label not in filename)
        gender = np.random.choice(genders)
        movement = np.random.choice(['full', 'half', 'invalid'])
        
        hht_matrix = generate_hht_matrix(
            matrix_size=256,
            gender=gender,
            movement=movement,
            seed=sample_count + i + 1000
        )
        
        np.savez(filepath, hht=hht_matrix)
        test_count += 1
    
    print(f"Generated {test_count} test samples")
    print(f"\nTotal files created: {sample_count + test_count}")
    print(f"Dataset ready at: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample sEMG HHT data')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for generated data')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of samples per class')
    parser.add_argument('--n_test', type=int, default=10,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.n_samples, args.n_test)
