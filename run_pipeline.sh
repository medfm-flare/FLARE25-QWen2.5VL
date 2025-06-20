#!/bin/bash
# FLARE 2025 QWen2.5-VL Baseline Pipeline Runner

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Default values
BASE_DIR="organized_dataset"
OUTPUT_DIR="processed_data_qwenvl"
MODEL_OUTPUT="finetuned_qwenvl"
EVAL_OUTPUT="evaluation_results"
USE_SUBSET=false
MAX_SAMPLES=""
SKIP_VALIDATION=false
SKIP_INSTALL_TEST=false
SKIP_DATASET_VALIDATION=false
SKIP_PREPARE=false
SKIP_TRAIN=false
SKIP_EVAL=false
NUM_EPOCHS=3

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use-subset)
            USE_SUBSET=true
            shift
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-install-test)
            SKIP_INSTALL_TEST=true
            shift
            ;;
        --skip-dataset-validation)
            SKIP_DATASET_VALIDATION=true
            shift
            ;;
        --skip-prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --base-dir DIR             Base directory for datasets (default: organized_dataset)"
            echo "  --output-dir DIR           Output directory for processed data (default: processed_data_qwenvl)"
            echo "  --use-subset               Use only 3 datasets for testing"
            echo "  --max-samples N            Maximum samples per dataset"
            echo "  --skip-validation          Skip all validation steps (install test + dataset validation)"
            echo "  --skip-install-test        Skip installation testing step"
            echo "  --skip-dataset-validation  Skip dataset validation step"
            echo "  --skip-prepare             Skip data preparation step"
            echo "  --skip-train               Skip training step"
            echo "  --skip-eval                Skip evaluation step"
            echo "  --num-epochs N             Number of training epochs (default: 3)"
            echo "  --help                     Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Python installation
print_status "Checking Python installation..."
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Check CUDA availability
print_status "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p $OUTPUT_DIR
mkdir -p $MODEL_OUTPUT
mkdir -p $EVAL_OUTPUT

# Step 0: Validation Steps
# Handle --skip-validation flag
if [ "$SKIP_VALIDATION" = true ]; then
    SKIP_INSTALL_TEST=true
    SKIP_DATASET_VALIDATION=true
fi

# Step 0a: Installation Testing
if [ "$SKIP_INSTALL_TEST" = false ]; then
    print_status "Step 0a: Testing installation..."
    
    TEST_INSTALL_CMD="python test_installation.py"
    
    print_status "Running: $TEST_INSTALL_CMD"
    if $TEST_INSTALL_CMD; then
        print_status "Installation test completed successfully!"
    else
        print_error "Installation test failed!"
        exit 1
    fi
else
    print_warning "Skipping installation test step"
fi

# Step 0b: Dataset Validation
if [ "$SKIP_DATASET_VALIDATION" = false ]; then
    print_status "Step 0b: Validating dataset..."
    
    VALIDATE_CMD="python validate_dataset.py --base_dir $BASE_DIR --check_images"
    
    print_status "Running: $VALIDATE_CMD"
    if $VALIDATE_CMD; then
        print_status "Dataset validation completed successfully!"
    else
        print_error "Dataset validation failed!"
        exit 1
    fi
else
    print_warning "Skipping dataset validation step"
fi

# Step 1: Data Preparation
if [ "$SKIP_PREPARE" = false ]; then
    print_status "Step 1: Preparing datasets..."
    
    PREPARE_CMD="python prepare_data.py --base_dir $BASE_DIR --output_dir $OUTPUT_DIR"
    
    if [ "$USE_SUBSET" = true ]; then
        PREPARE_CMD="$PREPARE_CMD --use_subset"
    fi
    
    if [ ! -z "$MAX_SAMPLES" ]; then
        PREPARE_CMD="$PREPARE_CMD --max_samples $MAX_SAMPLES"
    fi
    
    print_status "Running: $PREPARE_CMD"
    if $PREPARE_CMD; then
        print_status "Data preparation completed successfully!"
    else
        print_error "Data preparation failed!"
        exit 1
    fi
else
    print_warning "Skipping data preparation step"
fi

# Step 2: Fine-tuning
if [ "$SKIP_TRAIN" = false ]; then
    print_status "Step 2: Fine-tuning model..."
    
    TRAIN_CMD="python finetune_qwenvl.py --processed_data_dir $OUTPUT_DIR --output_dir $MODEL_OUTPUT --num_epochs $NUM_EPOCHS"
    
    print_status "Running: $TRAIN_CMD"
    if $TRAIN_CMD; then
        print_status "Training completed successfully!"
    else
        print_error "Training failed!"
        exit 1
    fi
else
    print_warning "Skipping training step"
fi

# Step 3: Evaluation
if [ "$SKIP_EVAL" = false ]; then
    print_status "Step 3: Evaluating model..."
    
    EVAL_CMD="python evaluate_model.py --processed_data_dir $OUTPUT_DIR --lora_weights $MODEL_OUTPUT/final --output_dir $EVAL_OUTPUT"
    
    print_status "Running: $EVAL_CMD"
    if $EVAL_CMD; then
        print_status "Evaluation completed successfully!"
        
        # Display summary
        print_status "Pipeline completed! Results summary:"
        echo "- Processed data: $OUTPUT_DIR"
        echo "- Trained model: $MODEL_OUTPUT/final"
        echo "- Evaluation results: $EVAL_OUTPUT"
        
        # Show key metrics if available
        if [ -f "$EVAL_OUTPUT/comprehensive_model_comparison.json" ]; then
            print_status "Key metrics from evaluation:"
            python -c "
import json
with open('$EVAL_OUTPUT/comprehensive_model_comparison.json', 'r') as f:
    data = json.load(f)
    if 'overall_summary' in data:
        summary = data['overall_summary']
        print(f'- Tasks improved: {summary[\"improved_tasks\"]}/{summary[\"total_tasks\"]}')
        print(f'- Improvement rate: {summary.get(\"improvement_rate_percent\", 0):.1f}%')
"
        fi
    else
        print_error "Evaluation failed!"
        exit 1
    fi
else
    print_warning "Skipping evaluation step"
fi

print_status "Pipeline execution completed!" 