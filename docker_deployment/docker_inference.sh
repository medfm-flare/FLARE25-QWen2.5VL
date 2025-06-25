#!/bin/bash

# QwenVL Docker Inference Script
# FLARE 2025 Medical Multimodal VQA Challenge

set -e  # Exit on any error

echo "========================================="
echo "FLARE 2025 QwenVL Inference Container"
echo "========================================="

# Environment setup
export PYTHONPATH=/app:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Default values
DATASET_PATH=${DATASET_PATH:-"/app/input/organized_dataset"}
OUTPUT_PATH=${OUTPUT_PATH:-"/app/output"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-VL-7B-Instruct"}
LORA_WEIGHTS=${LORA_WEIGHTS:-"leoyinn/flare25-qwen2.5vl"}
MAX_TOKENS=${MAX_TOKENS:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-"auto"}
VERBOSE=${VERBOSE:-"false"}

echo "Configuration:"
echo "  Dataset Path: $DATASET_PATH"
echo "  Output Path: $OUTPUT_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  LoRA Weights: $LORA_WEIGHTS"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo "  Verbose: $VERBOSE"
echo "========================================="

# Check if GPU is available
echo "Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

# Check input paths
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path not found: $DATASET_PATH"
    echo "Please mount your dataset to /app/input/organized_dataset"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Build command arguments
CMD_ARGS=(
    "--dataset_path" "$DATASET_PATH"
    "--output_file" "$OUTPUT_PATH/predictions.json"
    "--max_tokens" "$MAX_TOKENS"
    "--batch_size" "$BATCH_SIZE"
    "--device" "$DEVICE"
)

# Add model name if not using LoRA
if [ -z "$LORA_WEIGHTS" ]; then
    CMD_ARGS+=("--model_name" "$MODEL_NAME")
else
    echo "Using LoRA weights: $LORA_WEIGHTS"
    # Check if it's a local directory or HuggingFace model ID
    if [[ "$LORA_WEIGHTS" == *"/"* ]] && [[ ! "$LORA_WEIGHTS" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$ ]]; then
        # Looks like a local path, check if it exists
        if [ ! -d "$LORA_WEIGHTS" ]; then
            echo "ERROR: LoRA weights path not found: $LORA_WEIGHTS"
            echo "Please mount your LoRA weights directory"
            exit 1
        fi
    else
        # HuggingFace model ID or simple path - let the Python script handle it
        echo "Using HuggingFace model ID: $LORA_WEIGHTS"
    fi
    CMD_ARGS+=("--lora_weights" "$LORA_WEIGHTS")
fi

# Add verbose flag if enabled
if [ "$VERBOSE" == "true" ]; then
    CMD_ARGS+=("--verbose")
fi

echo "Running inference..."
echo "Command: python inference.py ${CMD_ARGS[*]}"
echo "========================================="

# Run inference
python inference.py "${CMD_ARGS[@]}"

# Check if output was generated
if [ -f "$OUTPUT_PATH/predictions.json" ]; then
    echo "========================================="
    echo "Inference completed successfully!"
    echo "Output saved to: $OUTPUT_PATH/predictions.json"
    
    # Show basic stats
    python -c "
import json
try:
    with open('$OUTPUT_PATH/predictions.json', 'r') as f:
        predictions = json.load(f)
    print(f'Total predictions: {len(predictions)}')
    
    # Count by task type
    task_counts = {}
    for pred in predictions:
        task_type = pred.get('TaskType', 'Unknown')
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print('Predictions by task type:')
    for task_type, count in sorted(task_counts.items()):
        print(f'  {task_type}: {count}')
        
except Exception as e:
    print(f'Error reading predictions: {e}')
"
    echo "========================================="
else
    echo "ERROR: No output file generated!"
    exit 1
fi

echo "Container execution completed." 