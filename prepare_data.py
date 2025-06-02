import os
import json
import argparse

# Set CUDA device if needed (though this script doesn't use GPU)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import pandas as pd
from pathlib import Path
import random
import logging
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from collections import Counter

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", "data_preparation.log"))
    ]
)
logger = logging.getLogger(__name__)

def validate_image(image_path):
    """Validate image can be opened and is suitable for training"""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify it's a valid image
        img = Image.open(image_path)  # Reopen after verify
        img.convert("RGB")  # Check if convertible to RGB
        return True
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False

def load_images_and_questions(base_dir, dataset_name, split, task_type, category, max_samples=None, config=None):
    """
    Load images and questions from a specific dataset and split
    """
    logger.info(f"Loading {split} data for {category}/{dataset_name} dataset")
    
    try:
        if split == "train":
            image_dir = os.path.join(base_dir, "training", category, dataset_name, "imagesTr")
            question_file = os.path.join(base_dir, "training", category, dataset_name, f"{dataset_name}_questions_train.json")
        elif split == "val":
            # Check for hidden validation data first
            hidden_val_dir = os.path.join(base_dir, "validation-hidden", category, dataset_name)
            if os.path.exists(hidden_val_dir):
                image_dir = os.path.join(hidden_val_dir, "imagesVal")
                # Look for question file with GT (ground truth)
                question_file = os.path.join(hidden_val_dir, f"{dataset_name}_questions_val_withGT.json")
                if not os.path.exists(question_file):
                    # Try without GT suffix
                    question_file = os.path.join(hidden_val_dir, f"{dataset_name}_questions_val.json")
            else:
                # Fall back to public validation
                image_dir = os.path.join(base_dir, "validation-public", category, dataset_name, "imagesVal")
                question_file = os.path.join(base_dir, "validation-public", category, dataset_name, f"{dataset_name}_questions_val.json")
        elif split == "test":
            # For test data
            test_dir = os.path.join(base_dir, "testing", category, dataset_name)
            if os.path.exists(test_dir):
                image_dir = os.path.join(test_dir, "imagesTs")
                question_file = os.path.join(test_dir, f"{dataset_name}_questions_test.json")
            else:
                logger.warning(f"Test data not found for {dataset_name}, skipping")
                return []
        else:
            logger.error(f"Unknown split: {split}")
            return []
        
        # Check if directory and file exist
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return []
            
        if not os.path.exists(question_file):
            logger.error(f"Question file not found: {question_file}")
            return []
        
        # Load questions
        with open(question_file, 'r') as f:
            questions = json.load(f)

        logger.info(f"Loaded {len(questions)} questions from {question_file}")
        
        # REMOVED task type filtering - keep ALL task types as they appear in the data
        # This ensures we preserve the full diversity of tasks in each dataset
        unique_tasks = set(q['TaskType'] for q in questions)
        logger.info(f"Found task types: {sorted(unique_tasks)}")
        
        # Limit samples for prototyping if needed
        if max_samples is not None and max_samples > 0 and len(questions) > max_samples:
            # Use deterministic sampling for reproducibility
            random.seed(42)
            questions = random.sample(questions, max_samples)
            logger.info(f"Sampled {len(questions)} questions for prototyping")
        else:
            logger.info(f"Using all {len(questions)} available samples")
        
        formatted_data = []
        skipped_images = 0
        for question in tqdm(questions, desc=f"Processing {dataset_name} {split}"):
            # Handle ImageName being either a string or a list
            image_names = question["ImageName"]
            if isinstance(image_names, list):
                # For multi-image questions, use the first image as primary
                # This is a practical solution - future enhancement could handle all images
                image_name = image_names[0]
                logger.debug(f"Using first image from list of {len(image_names)} images")
            else:
                image_name = image_names
            
            image_path = os.path.join(image_dir, os.path.basename(image_name))
            
            # Check if image exists and is valid
            if not os.path.exists(image_path):
                logger.warning(f"Image not found - {image_path}")
                skipped_images += 1
                continue
                
            if not validate_image(image_path):
                skipped_images += 1
                continue
                
            sample = {
                "image": image_path,
                "question": question["Question"],
                "answer": str(question["Answer"]),
                "task_type": question["TaskType"],  # Preserve original case and task type
                "modality": question["Modality"]
            }
            formatted_data.append(sample)
        
        if skipped_images > 0:
            logger.warning(f"Skipped {skipped_images} invalid/missing images for {dataset_name} {split}")
        
        logger.info(f"Loaded {len(formatted_data)} valid samples for {dataset_name} {split}")
        return formatted_data
        
    except Exception as e:
        logger.error(f"Error loading data for {category}/{dataset_name}: {e}", exc_info=True)
        return []

def create_formatted_prompt(sample, task_type):
    """
    Create formatted prompt based on task type
    """
    base_prompt = f"<image>\n{sample['question']}"
    return base_prompt

def get_all_dataset_configs():
    """Get configuration for all 19 datasets in FLARE25"""
    return [
        # Retinography
        {
            "name": "retino",
            "category": "Retinography", 
            "task_type": None  # Load all task types
        },
        {
            "name": "fundus",
            "category": "Retinography",
            "task_type": None  # Load all task types
        },
        # Ultrasound
        {
            "name": "BUSI-det",
            "category": "Ultrasound",
            "task_type": None  # Load all task types
        },
        {
            "name": "BUS-UCLM-det",
            "category": "Ultrasound",
            "task_type": None  # Load all task types
        },
        {
            "name": "BUSI",
            "category": "Ultrasound",
            "task_type": None  # Load all task types
        },
        {
            "name": "BUS-UCLM",
            "category": "Ultrasound",
            "task_type": None  # Load all task types
        },
        {
            "name": "iugc",
            "category": "Ultrasound",
            "task_type": None  # Load all task types - this dataset has Classification, Detection, Regression
        },
        # Xray
        {
            "name": "boneresorption",
            "category": "Xray",
            "task_type": None  # Load all task types
        },
        {
            "name": "dental",
            "category": "Xray",
            "task_type": None  # Load all task types
        },
        {
            "name": "periapical",
            "category": "Xray",
            "task_type": None  # Load all task types - has Classification, Multi-label Classification
        },
        {
            "name": "IU_XRay",
            "category": "Xray",
            "task_type": None  # Load all task types
        },
        {
            "name": "chestdr",
            "category": "Xray",
            "task_type": None  # Load all task types - has Classification, Multi-label Classification
        },
        # Clinical
        {
            "name": "neojaundice",
            "category": "Clinical",
            "task_type": None  # Load all task types
        },
        # Microscopy
        {
            "name": "chromosome",
            "category": "Microscopy",
            "task_type": None  # Load all task types
        },
        {
            "name": "neurips22cell",
            "category": "Microscopy",
            "task_type": None  # Load all task types
        },
        {
            "name": "bone_marrow",
            "category": "Microscopy",
            "task_type": None  # Load all task types
        },
        # Endoscopy
        {
            "name": "endo",
            "category": "Endoscopy",
            "task_type": None  # Load all task types
        },
        # Dermatology
        {
            "name": "bcn20000",
            "category": "Dermatology",
            "task_type": None  # Load all task types
        },
        # Mammography
        {
            "name": "CMMD",
            "category": "Mammography",
            "task_type": None  # Load all task types
        }
    ]

def validate_processed_data(output_dir):
    """Validate the processed datasets"""
    logger.info("Validating processed datasets...")
    
    # Check train dataset
    train_path = os.path.join(output_dir, "train")
    if os.path.exists(train_path):
        train_dataset = load_from_disk(train_path)
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        
        # Check a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            logger.info(f"Sample {i}: image={os.path.basename(sample['image'])}, "
                       f"question_len={len(sample['question'])}, "
                       f"task_type={sample['task_type']}")
            
            # Verify image exists
            if not os.path.exists(sample['image']):
                logger.error(f"Image not found: {sample['image']}")
    
    # Check validation dataset
    val_path = os.path.join(output_dir, "validation")
    if os.path.exists(val_path):
        val_dataset = load_from_disk(val_path)
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # Check test dataset
    test_path = os.path.join(output_dir, "test")
    if os.path.exists(test_path):
        test_dataset = load_from_disk(test_path)
        logger.info(f"Test dataset: {len(test_dataset)} samples")

def prepare_dataset(base_dir, dataset_configs, output_dir, max_samples=None):
    """
    Prepare datasets for fine-tuning
    """
    all_train_data = []
    all_val_data = []
    all_test_data = []
    
    # If dataset_configs is None, use all datasets
    if dataset_configs is None:
        dataset_configs = get_all_dataset_configs()
    
    # Track dataset statistics
    dataset_stats = {}
    
    for config in dataset_configs:
        try:
            dataset_name = config["name"]
            category = config["category"]
            task_type = config["task_type"]
            
            dataset_stats[dataset_name] = {"train": 0, "val": 0, "test": 0}
            
            # Load training data
            train_data = load_images_and_questions(base_dir, dataset_name, "train", task_type, category, max_samples, config)
            dataset_stats[dataset_name]["train"] = len(train_data)
            all_train_data.extend(train_data)
            
            # Load validation data
            val_sample_size = max_samples // 5 if max_samples else None
            val_data = load_images_and_questions(base_dir, dataset_name, "val", task_type, category, val_sample_size, config)
            dataset_stats[dataset_name]["val"] = len(val_data)
            all_val_data.extend(val_data)
            
            # Load test data if available
            test_sample_size = max_samples // 5 if max_samples else None
            test_data = load_images_and_questions(base_dir, dataset_name, "test", task_type, category, test_sample_size, config)
            dataset_stats[dataset_name]["test"] = len(test_data)
            if test_data:
                all_test_data.extend(test_data)
            
        except Exception as e:
            logger.error(f"Error processing dataset {config.get('name', 'unknown')}: {e}", exc_info=True)
            continue  # Continue with other datasets instead of failing completely
    
    # Log dataset statistics
    logger.info("Dataset loading summary:")
    for dataset_name, stats in dataset_stats.items():
        logger.info(f"  {dataset_name}: train={stats['train']}, val={stats['val']}, test={stats['test']}")
    
    # Check if we have enough data
    if len(all_train_data) == 0:
        logger.error("No training data was loaded. Cannot proceed.")
        raise ValueError("No training data available")
        
    if len(all_val_data) == 0:
        logger.warning("No validation data was loaded. Will only prepare training data.")
    
    logger.info(f"Total training samples: {len(all_train_data)}")
    logger.info(f"Total validation samples: {len(all_val_data)}")
    logger.info(f"Total test samples: {len(all_test_data)}")
    
    # Convert to HF datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(all_train_data))
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving training dataset to {output_dir}/train")
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    
    if len(all_val_data) > 0:
        val_dataset = Dataset.from_pandas(pd.DataFrame(all_val_data))
        logger.info(f"Saving validation dataset to {output_dir}/validation")
        val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    
    if len(all_test_data) > 0:
        test_dataset = Dataset.from_pandas(pd.DataFrame(all_test_data))
        logger.info(f"Saving test dataset to {output_dir}/test")
        test_dataset.save_to_disk(os.path.join(output_dir, "test"))
    
    # Save dataset info
    dataset_info = {
        "train_samples": len(all_train_data),
        "val_samples": len(all_val_data),
        "test_samples": len(all_test_data),
        "dataset_configs": dataset_configs,
        "dataset_stats": dataset_stats,
        "task_types": list(set(item["task_type"] for item in all_train_data)),
        "modalities": list(set(item["modality"] for item in all_train_data))
    }
    
    # ENHANCED: Add detailed task type and modality statistics
    task_type_counts = Counter(item["task_type"] for item in all_train_data)
    modality_counts = Counter(item["modality"] for item in all_train_data)
    
    dataset_info["task_type_distribution"] = dict(task_type_counts)
    dataset_info["modality_distribution"] = dict(modality_counts)
    
    logger.info("=== DATASET COMPOSITION ANALYSIS ===")
    logger.info(f"Task Type Distribution (Training Data):")
    for task_type, count in sorted(task_type_counts.items()):
        percentage = (count / len(all_train_data)) * 100
        logger.info(f"  {task_type}: {count} samples ({percentage:.1f}%)")
    
    logger.info(f"Modality Distribution (Training Data):")
    for modality, count in sorted(modality_counts.items()):
        percentage = (count / len(all_train_data)) * 100
        logger.info(f"  {modality}: {count} samples ({percentage:.1f}%)")
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Dataset preparation complete. Data saved to {output_dir}")
    logger.info(f"Training samples: {len(all_train_data)}")
    logger.info(f"Validation samples: {len(all_val_data)}")
    logger.info(f"Test samples: {len(all_test_data)}")
    
    # Validate the processed data
    validate_processed_data(output_dir)
    
    return len(all_train_data), len(all_val_data), len(all_test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for QwenVL fine-tuning")
    parser.add_argument("--base_dir", type=str, default="organized_dataset", help="Base directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="processed_data_qwenvl", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples per dataset (for prototyping, default: use all data)")
    parser.add_argument("--use_subset", action="store_true", help="Use only the original 3 datasets for testing")
    args = parser.parse_args()
    
    # Define datasets to use
    if args.use_subset:
        # Use only the original 3 datasets for testing
        dataset_configs = [
            {
                "name": "retino",
                "category": "Retinography",
                "task_type": None
            },
            {
                "name": "BUSI-det",
                "category": "Ultrasound",
                "task_type": None
            },
            {
                "name": "boneresorption",
                "category": "Xray",
                "task_type": None
            }
        ]
    else:
        # Use all 19 datasets
        dataset_configs = None  # Will use get_all_dataset_configs()
    
    try:
        logger.info(f"Starting dataset preparation with max_samples={args.max_samples if args.max_samples else 'ALL'}")
        train_count, val_count, test_count = prepare_dataset(args.base_dir, dataset_configs, args.output_dir, args.max_samples)
        logger.info(f"Successfully prepared {train_count} training, {val_count} validation, and {test_count} test samples")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}", exc_info=True)
        raise 