#!/bin/bash
# Complete workflow example for sEMG-HHT CNN-SVM Classifier

echo "=========================================="
echo "sEMG-HHT CNN Classifier - Complete Workflow"
echo "=========================================="

# Step 1: Generate sample data (for testing)
echo -e "\n[Step 1/4] Generating sample data..."
python generate_sample_data.py \
    --output_dir ./example_data \
    --n_samples 25 \
    --n_test 5

# Step 2: Train the model
echo -e "\n[Step 2/4] Training the model..."
python train.py \
    --data_dir ./example_data \
    --checkpoint_dir ./example_checkpoints \
    --batch_size 32 \
    --test_size 0.2

# Step 3: Run inference on test files (already done during training)
echo -e "\n[Step 3/4] Test predictions saved in example_checkpoints/test_predictions.json"

# Step 4: Test inference on new data
echo -e "\n[Step 4/4] Testing inference on a single file..."
# Get first test file
TEST_FILE=$(ls example_data/*test*.npz | head -1)
if [ -n "$TEST_FILE" ]; then
    python inference.py \
        --checkpoint ./example_checkpoints/final \
        --input "$TEST_FILE" \
        --cpu
else
    echo "No test files found for inference demo"
fi

echo -e "\n=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo "Results:"
echo "  - Model checkpoints: example_checkpoints/"
echo "  - Test predictions: example_checkpoints/test_predictions.json"
echo ""
echo "To clean up:"
echo "  rm -rf example_data example_checkpoints"
