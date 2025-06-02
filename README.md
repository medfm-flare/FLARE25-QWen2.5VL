# FLARE 2025 QWen2.5-VL Baseline

This repository provides a baseline implementation for fine-tuning QWen2.5-VL-7B on the FLARE 2025 2D multimodal medical image challenge datasets.

## Overview

The pipeline supports all 19 datasets across 7 medical imaging modalities:
- **Retinography**: retino, fundus
- **Ultrasound**: BUSI-det, BUS-UCLM-det, BUSI, BUS-UCLM, iugc
- **X-ray**: boneresorption, dental, periapical, IU_XRay, chestdr
- **Clinical**: neojaundice
- **Microscopy**: chromosome, neurips22cell, bone_marrow
- **Endoscopy**: endo
- **Dermatology**: bcn20000
- **Mammography**: CMMD

### Supported Task Types:
- Classification (Balanced Accuracy)
- Multi-label Classification (F1 Score)
- Detection (F1 Score @ IoU 0.5)
- Instance Detection (F1 Score @ IoU 0.3/0.5)
- Cell Counting (Mean Absolute Error)
- Regression (Mean Absolute Error)
- Report Generation (GREEN Score)

## Requirements

- Python 3.8+
- CUDA 11.8+ with GPU (minimum 24GB VRAM recommended)
- 100GB+ free disk space for datasets and models

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd finetune_qwenvl_script

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The pipeline expects the FLARE 2025 dataset to be organized in the following structure:

```
organized_dataset/
├── training/
│   ├── Retinography/
│   │   ├── retino/
│   │   │   ├── imagesTr/
│   │   │   └── retino_questions_train.json
│   │   └── fundus/
│   │       └── ...
│   └── ...
├── validation-hidden/
│   └── ...
└── validation-public/
    └── ...
```

## Usage

### 1. Prepare Data

```bash
# Process all 19 datasets
python prepare_data.py \
    --base_dir organized_dataset \
    --output_dir processed_data_qwenvl

# For testing with subset (3 datasets only)
python prepare_data.py \
    --base_dir organized_dataset \
    --output_dir processed_data_qwenvl_subset \
    --use_subset \
    --max_samples 100
```

### 2. Fine-tune Model

```bash
# Full training
python finetune_qwenvl.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --processed_data_dir processed_data_qwenvl \
    --output_dir finetuned_qwenvl \
    --num_epochs 3

# Resume from checkpoint
python finetune_qwenvl.py \
    --resume_from_checkpoint finetuned_qwenvl/checkpoint-1000
```

### 3. Evaluate Model

```bash
# Evaluate and compare with base model
python evaluate_model.py \
    --processed_data_dir processed_data_qwenvl \
    --lora_weights finetuned_qwenvl/final \
    --output_dir evaluation_results

# Evaluate only fine-tuned model
python evaluate_model.py \
    --processed_data_dir processed_data_qwenvl \
    --lora_weights finetuned_qwenvl/final \
    --skip_base_model
```

## Key Features

### Memory Optimization
- 4-bit quantization with QLoRA
- Dynamic batch sizing based on available GPU memory
- Image caching with LRU eviction
- Gradient checkpointing

### Training Strategy
- Only trains on assistant responses (proper instruction masking)
- Task-specific prompt engineering
- Early stopping with patience
- Cosine learning rate schedule with warmup

### Evaluation Metrics
- Comprehensive metrics per task type
- Multiple IoU thresholds for detection tasks (0.3-0.7)
- Per-chromosome breakdown for instance detection
- Detailed comparison reports
- Comprehensive GREEN Score for report generation with:
  - Clinical entity matching with severity assessment
  - Anatomical location grounding with laterality
  - Temporal information consistency
  - Size/measurement accuracy
  - Uncertainty and negation handling
  - Clinical significance weighting
  - Report structure evaluation

## Configuration Options

### prepare_data.py
- `--max_samples`: Limit samples per dataset for prototyping
- `--use_subset`: Use only 3 datasets for testing

### finetune_qwenvl.py
- `--batch_size`: Per-device batch size (auto-determined by default)
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--num_epochs`: Number of training epochs (default: 3)
- `--lora_rank`: LoRA rank (default: 16)
- `--early_stopping_patience`: Early stopping patience (default: 5)

### evaluate_model.py
- `--max_eval_samples`: Limit evaluation samples per task
- `--save_predictions`: Save individual predictions
- `--evaluate_base_model`: Compare with base model (default: True)

## Performance Notes

- Training requires ~20GB GPU memory with default settings
- Full dataset training takes ~24-48 hours on A100
- Evaluation takes ~2-4 hours depending on dataset size

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size or increase gradient accumulation
- Ensure no other processes are using GPU
- Check GPU memory with `nvidia-smi`

### Token/Feature Mismatch
- Fixed by locking image resolution to 448x448
- Ensure processor configuration is not modified

### Slow Data Loading
- Increase image cache size if RAM permits
- Use SSD for dataset storage
- Check image file integrity

## Citation

If you use this baseline in your research, please cite:

```bibtex
@misc{flare2025_qwenvl_baseline,
  title={QWen2.5-VL Baseline for FLARE 2025 2D Multimodal Medical Image Challenge},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- QWen team for the base model
- FLARE 2025 organizers for the dataset
- HuggingFace for the transformers library 