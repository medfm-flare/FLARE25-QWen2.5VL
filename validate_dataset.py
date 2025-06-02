#!/usr/bin/env python3
"""
Validate FLARE 2025 dataset structure and integrity
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import logging
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dataset_structure(base_dir):
    """Check if dataset follows expected FLARE 2025 structure"""
    base_path = Path(base_dir)
    
    issues = []
    stats = defaultdict(lambda: defaultdict(int))
    
    # Expected top-level directories
    expected_dirs = ['training', 'validation-hidden', 'validation-public', 'testing']
    found_dirs = []
    
    for dir_name in expected_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            found_dirs.append(dir_name)
            logger.info(f"✓ Found {dir_name} directory")
        else:
            logger.warning(f"✗ Missing {dir_name} directory")
    
    if not found_dirs:
        logger.error("No expected directories found. Please check the base directory path.")
        return False, stats
    
    # Check each split
    for split in found_dirs:
        split_path = base_path / split
        
        # Find all category directories
        categories = [d for d in split_path.iterdir() if d.is_dir()]
        
        for category_path in categories:
            category = category_path.name
            
            # Find all dataset directories
            datasets = [d for d in category_path.iterdir() if d.is_dir()]
            
            for dataset_path in datasets:
                dataset = dataset_path.name
                stats[split][f"{category}/{dataset}"] += 1
                
                # Check for expected subdirectories based on split
                if split == 'training':
                    image_dir = dataset_path / 'imagesTr'
                    question_file = dataset_path / f"{dataset}_questions_train.json"
                elif split in ['validation-hidden', 'validation-public']:
                    image_dir = dataset_path / 'imagesVal'
                    question_file = dataset_path / f"{dataset}_questions_val.json"
                    # Also check for GT version
                    question_file_gt = dataset_path / f"{dataset}_questions_val_withGT.json"
                    if question_file_gt.exists():
                        question_file = question_file_gt
                else:  # testing
                    image_dir = dataset_path / 'imagesTs'
                    question_file = dataset_path / f"{dataset}_questions_test.json"
                
                # Validate structure
                if not image_dir.exists():
                    issues.append(f"Missing image directory: {image_dir}")
                else:
                    image_count = len(list(image_dir.glob('*')))
                    logger.info(f"  {dataset} ({split}): {image_count} images")
                
                if not question_file.exists():
                    issues.append(f"Missing question file: {question_file}")
                else:
                    # Load and validate questions
                    try:
                        with open(question_file, 'r') as f:
                            questions = json.load(f)
                        logger.info(f"  {dataset} ({split}): {len(questions)} questions")
                        
                        # Check question format
                        if questions:
                            sample = questions[0]
                            required_fields = ['Question', 'Answer', 'TaskType', 'Modality', 'ImageName']
                            missing_fields = [f for f in required_fields if f not in sample]
                            if missing_fields:
                                issues.append(f"Missing fields in {question_file}: {missing_fields}")
                    
                    except Exception as e:
                        issues.append(f"Error reading {question_file}: {e}")
    
    # Report issues
    if issues:
        logger.error(f"\nFound {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            logger.error(f"  - {issue}")
        if len(issues) > 10:
            logger.error(f"  ... and {len(issues) - 10} more issues")
    
    return len(issues) == 0, stats


def validate_images(base_dir, sample_size=100):
    """Validate a sample of images can be loaded"""
    base_path = Path(base_dir)
    
    logger.info(f"\nValidating images (sample size: {sample_size} per dataset)...")
    
    corrupted_images = []
    total_checked = 0
    
    for split in ['training', 'validation-hidden', 'validation-public']:
        split_path = base_path / split
        if not split_path.exists():
            continue
        
        for category_path in split_path.iterdir():
            if not category_path.is_dir():
                continue
                
            for dataset_path in category_path.iterdir():
                if not dataset_path.is_dir():
                    continue
                
                # Find image directory
                if split == 'training':
                    image_dir = dataset_path / 'imagesTr'
                else:
                    image_dir = dataset_path / 'imagesVal'
                
                if not image_dir.exists():
                    continue
                
                # Sample images
                all_images = list(image_dir.glob('*'))
                sample_images = all_images[:min(sample_size, len(all_images))]
                
                for img_path in tqdm(sample_images, desc=f"Checking {dataset_path.name}", leave=False):
                    total_checked += 1
                    try:
                        img = Image.open(img_path)
                        img.verify()
                        # Reopen and convert to ensure it's fully readable
                        img = Image.open(img_path)
                        img.convert('RGB')
                    except Exception as e:
                        corrupted_images.append((str(img_path), str(e)))
    
    logger.info(f"Checked {total_checked} images")
    if corrupted_images:
        logger.error(f"Found {len(corrupted_images)} corrupted images:")
        for path, error in corrupted_images[:5]:
            logger.error(f"  - {path}: {error}")
    else:
        logger.info("All checked images are valid")
    
    return len(corrupted_images) == 0


def generate_dataset_report(base_dir, stats):
    """Generate a summary report of the dataset"""
    report_path = Path(base_dir) / 'dataset_validation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("FLARE 2025 Dataset Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        for split, datasets in stats.items():
            f.write(f"\n{split.upper()}:\n")
            f.write("-" * 30 + "\n")
            
            total_datasets = len(datasets)
            f.write(f"Total datasets: {total_datasets}\n")
            
            for dataset_name in sorted(datasets.keys()):
                f.write(f"  - {dataset_name}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Validation completed. Check logs for any issues.\n")
    
    logger.info(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate FLARE 2025 dataset structure")
    parser.add_argument("--base_dir", type=str, default="organized_dataset",
                        help="Base directory of the FLARE 2025 dataset")
    parser.add_argument("--check_images", action="store_true",
                        help="Also validate image files (slower)")
    parser.add_argument("--image_sample_size", type=int, default=100,
                        help="Number of images to check per dataset")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return 1
    
    logger.info(f"Validating dataset at: {os.path.abspath(args.base_dir)}")
    
    # Check structure
    structure_valid, stats = check_dataset_structure(args.base_dir)
    
    # Optionally check images
    images_valid = True
    if args.check_images:
        images_valid = validate_images(args.base_dir, args.image_sample_size)
    
    # Generate report
    generate_dataset_report(args.base_dir, stats)
    
    # Final status
    if structure_valid and images_valid:
        logger.info("\n✅ Dataset validation PASSED")
        return 0
    else:
        logger.error("\n❌ Dataset validation FAILED")
        return 1


if __name__ == "__main__":
    exit(main()) 