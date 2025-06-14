import os
import argparse
import json
import string


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, balanced_accuracy_score
from peft import PeftModel
import re
import logging
from PIL import Image
import time
import datetime
import ast
import inspect
from collections import defaultdict

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging with both console and file handlers from the start
def setup_logging():
    """Set up logging configuration"""
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class InfinityEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles infinity values"""
    def encode(self, obj):
        if isinstance(obj, float):
            if obj == float('inf'):
                return '"Infinity"'
            elif obj == float('-inf'):
                return '"-Infinity"'
            elif obj != obj:  # NaN check
                return '"NaN"'
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        """Encode the given object and return an iterator of string chunks."""
        if _one_shot and isinstance(obj, dict):
            obj = self._handle_dict(obj)
        elif isinstance(obj, list):
            obj = [self._handle_value(item) for item in obj]
        elif isinstance(obj, dict):
            obj = self._handle_dict(obj)
        else:
            obj = self._handle_value(obj)
        return super().iterencode(obj, _one_shot)
    
    def _handle_dict(self, obj):
        """Handle dictionary objects recursively"""
        result = {}
        for key, value in obj.items():
            result[key] = self._handle_value(value)
        return result
    
    def _handle_value(self, obj):
        """Handle individual values"""
        if isinstance(obj, float):
            if obj == float('inf'):
                return "Infinity"
            elif obj == float('-inf'):
                return "-Infinity" 
            elif obj != obj:  # NaN check
                return "NaN"
        elif isinstance(obj, dict):
            return self._handle_dict(obj)
        elif isinstance(obj, list):
            return [self._handle_value(item) for item in obj]
        return obj

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x_min, y_min, x_max, y_max] format"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_instance_detection_metrics(predictions, references, 
                                                iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    Calculate instance detection metrics following computer vision standards
    
    Args:
        predictions: List of dicts with chromosome_id -> list of boxes
        references: List of dicts with chromosome_id -> list of boxes  
        iou_thresholds: List of IoU thresholds to evaluate at
        
    Returns:
        dict: Metrics including:
            - F1, Precision, Recall at each threshold
            - Average metrics across thresholds (COCO-style)
            - Per-chromosome breakdown
            - Primary metrics for reporting
    """
    
    if not predictions or not references:
        return {
            "f1_score_at_03": 0.0,        # PRIMARY METRIC: F1@0.3 (medical imaging standard)
            "f1_score_at_05": 0.0,        # Standard computer vision metric
            "average_f1": 0.0,            # COCO-style average across thresholds
            "precision_at_03": 0.0,
            "recall_at_03": 0.0,
            "valid_samples": 0,
            "detailed_metrics": {},
            "per_chromosome_metrics": {}
        }
    
    # Calculate metrics at each threshold using the same logic as original but with different thresholds
    threshold_results = {}
    
    for iou_threshold in iou_thresholds:
        # Calculate TP, FP, FN for each chromosome identity separately at this threshold
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Count total boxes for debugging
        total_pred_boxes = 0
        total_ref_boxes = 0
        
        for i in range(len(references)):
            ref_dict = references[i]
            pred_dict = predictions[i]
            
            # Handle format issues
            if not isinstance(ref_dict, dict):
                ref_dict = {}
            if not isinstance(pred_dict, dict):
                pred_dict = {}
            
            # Get all chromosome IDs from both reference and prediction
            all_chromosome_ids = set(ref_dict.keys()) | set(pred_dict.keys())
            
            for chromosome_id in all_chromosome_ids:
                ref_boxes = ref_dict.get(chromosome_id, [])
                pred_boxes = pred_dict.get(chromosome_id, [])
                
                # Ensure boxes are lists
                if not isinstance(ref_boxes, list):
                    ref_boxes = []
                if not isinstance(pred_boxes, list):
                    pred_boxes = []
                
                # Count boxes
                total_ref_boxes += len(ref_boxes)
                total_pred_boxes += len(pred_boxes)
                
                # Match predictions to references for this specific chromosome ID
                matched_refs = set()
                matched_preds = set()
                
                # Find matches with IoU > threshold within the same chromosome ID
                for j, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_ref_idx = -1
                    
                    for k, ref_box in enumerate(ref_boxes):
                        if k in matched_refs:
                            continue
                        iou = calculate_iou(pred_box, ref_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_ref_idx = k
                    
                    if best_iou > iou_threshold:
                        matched_refs.add(best_ref_idx)
                        matched_preds.add(j)
                        total_tp += 1
                
                # False positives: unmatched predictions for this chromosome
                total_fp += len(pred_boxes) - len(matched_preds)
                
                # False negatives: unmatched references for this chromosome
                total_fn += len(ref_boxes) - len(matched_refs)
        
        # Log counts for first threshold only to avoid spam
        if iou_threshold == 0.3:
            logger.info(f"Instance detection metrics - Total ref boxes: {total_ref_boxes}, pred boxes: {total_pred_boxes}")
            logger.info(f"Instance detection metrics - TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        
        # Calculate metrics for this threshold
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        threshold_results[iou_threshold] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }
    
    # Calculate per-chromosome metrics (using 0.3 threshold for medical imaging)
    per_chromosome_metrics = calculate_per_chromosome_metrics(predictions, references, iou_threshold=0.3)
    
    # Calculate averages across thresholds
    avg_f1 = np.mean([threshold_results[t]['f1_score'] for t in iou_thresholds])
    avg_precision = np.mean([threshold_results[t]['precision'] for t in iou_thresholds])
    avg_recall = np.mean([threshold_results[t]['recall'] for t in iou_thresholds])
    
    # Return results
    return {
        # PRIMARY METRICS (for main reporting)
        "f1_score_at_03": threshold_results[0.3]['f1_score'],      # Medical imaging standard
        "f1_score_at_05": threshold_results[0.5]['f1_score'],      # Computer vision standard
        "average_f1": avg_f1,                                      # COCO-style average
        "precision_at_03": threshold_results[0.3]['precision'],
        "recall_at_03": threshold_results[0.3]['recall'],
        "valid_samples": len(predictions),
        
        # DETAILED METRICS
        "detailed_metrics": {
            f"IoU_{t}": {
                "f1_score": threshold_results[t]['f1_score'],
                "precision": threshold_results[t]['precision'], 
                "recall": threshold_results[t]['recall'],
                "tp": threshold_results[t]['tp'],
                "fp": threshold_results[t]['fp'],
                "fn": threshold_results[t]['fn']
            } for t in iou_thresholds
        },
        
        # COCO-STYLE AVERAGES
        "coco_style_metrics": {
            "average_f1": avg_f1,
            "average_precision": avg_precision,
            "average_recall": avg_recall
        },
        
        # PER-CHROMOSOME BREAKDOWN
        "per_chromosome_metrics": per_chromosome_metrics
    }

def calculate_per_chromosome_metrics(predictions, references, iou_threshold=0.3):
    """Calculate metrics for each chromosome type separately"""
    chromosome_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred_dict, ref_dict in zip(predictions, references):
        if not isinstance(pred_dict, dict) or not isinstance(ref_dict, dict):
            continue
            
        all_chromosome_ids = set(ref_dict.keys()) | set(pred_dict.keys())
        
        for chromosome_id in all_chromosome_ids:
            ref_boxes = ref_dict.get(chromosome_id, [])
            pred_boxes = pred_dict.get(chromosome_id, [])
            
            if not isinstance(ref_boxes, list):
                ref_boxes = []
            if not isinstance(pred_boxes, list):
                pred_boxes = []
            
            # Match boxes for this chromosome
            matched_refs = set()
            matched_preds = set()
            
            for i, pred_box in enumerate(pred_boxes):
                if not isinstance(pred_box, list) or len(pred_box) != 4:
                    continue
                    
                best_iou = 0
                best_ref_idx = -1
                
                for j, ref_box in enumerate(ref_boxes):
                    if j in matched_refs or not isinstance(ref_box, list) or len(ref_box) != 4:
                        continue
                    
                    iou = calculate_iou(pred_box, ref_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_ref_idx = j
                
                if best_iou >= iou_threshold:
                    chromosome_metrics[chromosome_id]['tp'] += 1
                    matched_refs.add(best_ref_idx)
                    matched_preds.add(i)
                else:
                    chromosome_metrics[chromosome_id]['fp'] += 1
            
            # Count unmatched references as false negatives
            chromosome_metrics[chromosome_id]['fn'] += len(ref_boxes) - len(matched_refs)
    
    # Calculate per-chromosome F1 scores
    chromosome_results = {}
    for chromosome_id, metrics in chromosome_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        chromosome_results[chromosome_id] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return chromosome_results

def parse_answer(output, task_type):
    """Parse model output based on task type"""
    # Handle NaN task_type values
    if task_type is None or (hasattr(task_type, '__class__') and task_type.__class__.__name__ == 'float' and str(task_type) == 'nan'):
        logger.warning(f"Invalid task_type: {task_type}, defaulting to 'classification'")
        task_type = "classification"
    
    # Ensure task_type is a string
    task_type = str(task_type)
    
    # Normalize task type for consistent parsing
    task_type_normalized = task_type.lower().strip().replace("_", " ").replace("-", " ")
    
    # Clean output first to remove chat template artifacts
    cleaned_output = output
    
    # Remove chat template artifacts
    if "assistant\n" in cleaned_output:
        cleaned_output = cleaned_output.rsplit("assistant\n", 1)[-1].strip()
    elif "assistant:" in cleaned_output.lower():
        idx = cleaned_output.lower().rfind("assistant:")
        cleaned_output = cleaned_output[idx + len("assistant:"):].strip()
    
    # Remove system/user markers
    for marker in ["system\n", "user\n"]:
        if cleaned_output.startswith(marker):
            cleaned_output = cleaned_output[len(marker):].strip()
    
    # Try to extract content after "Answer:" if present
    if "Answer:" in cleaned_output:
        cleaned_output = cleaned_output.rsplit("Answer:", 1)[-1].strip()
    
    if "classification" in task_type_normalized and "multi" not in task_type_normalized:
        # Classification (Disease Diagnosis) - extract single answer
        final_answer_str = cleaned_output
        
        # Try to find a single letter A-E, possibly surrounded by common punctuation or short phrases.
        match = re.search(r"\b([A-E])(?:[.\)\s]|$)", final_answer_str.upper())
        if match:
            final_answer_str = match.group(1)
        else:
            # If no clear A-E choice, take the last word as fallback
            words = final_answer_str.split()
            if words:
                final_answer_str = words[-1]
        
        # Final cleanup
        final_answer_str = re.sub(r'^[\s\.,:\'\"]+|[\s\.,:\'\"]+$', '', final_answer_str)
        
        logger.debug(f"Classification parsing: Raw: '{output}' -> Cleaned: '{cleaned_output}' -> Final: '{final_answer_str}'")
        return final_answer_str
    
    elif "multi" in task_type_normalized and "label" in task_type_normalized:
        # Multi-label classification - extract multiple labels
        try:
            answer_part = cleaned_output
            
            # Handle different separator formats based on dataset
            # Periapical dataset uses semicolon ";" 
            # ChestDR dataset uses comma ","
            labels = []
            
            # First try to extract from list format [label1, label2, ...]
            if "[" in answer_part and "]" in answer_part:
                match = re.search(r'\[([^\]]+)\]', answer_part)
                if match:
                    labels_str = match.group(1)
                    # Try semicolon first, then comma
                    if ";" in labels_str:
                        labels = [l.strip().strip("'\"") for l in labels_str.split(';') if l.strip()]
                    else:
                        labels = [l.strip().strip("'\"") for l in labels_str.split(',') if l.strip()]
                    return labels
            
            # Handle direct label listing - try semicolon first (periapical), then comma (chestdr)
            if ";" in answer_part:
                labels = [l.strip() for l in answer_part.split(';') if l.strip()]
            elif "," in answer_part:
                # For comma separation, be more careful to avoid splitting within labels
                # Split by comma but rejoin if it looks like we split within a single label
                potential_labels = [l.strip() for l in answer_part.split(',') if l.strip()]
                labels = potential_labels
            else:
                # Single label or unrecognized format
                labels = [answer_part.strip()] if answer_part.strip() else []
            
            # Clean up labels
            cleaned_labels = []
            for label in labels:
                cleaned_label = label.strip().strip("'\".,;")
                if cleaned_label:
                    cleaned_labels.append(cleaned_label)
            
            logger.debug(f"Multi-label parsing: Raw: '{output}' -> Cleaned: '{cleaned_output}' -> Labels: {cleaned_labels}")
            return cleaned_labels if cleaned_labels else [cleaned_output.strip()]
            
        except Exception as e:
            logger.warning(f"Failed to parse multi-label output: {output}. Error: {e}")
            return [cleaned_output.strip()]
    
    elif "instance" in task_type_normalized and "detection" in task_type_normalized:
        # Instance Detection - extract identity-mapped bounding boxes (e.g., chromosomes)
        try:
            # Instance detection expects JSON format: {"1": [[x,y,x,y], ...], "2": [[x,y,x,y]], ...}
            
            # Simple approach: just try to parse the JSON as-is
            if cleaned_output.startswith("{") and cleaned_output.endswith("}"):
                try:
                    json_data = json.loads(cleaned_output)
                    if isinstance(json_data, dict):
                        # Convert to standardized format: dict with string keys and list of [x,y,x,y] boxes
                        result = {}
                        for key, boxes in json_data.items():
                            if isinstance(boxes, list):
                                # Handle both single box and multiple boxes per chromosome
                                if len(boxes) > 0 and isinstance(boxes[0], list) and len(boxes[0]) == 4:
                                    # Multiple boxes: [[x,y,x,y], [x,y,x,y]]
                                    result[str(key)] = [[int(coord) for coord in box] for box in boxes]
                                elif len(boxes) == 4 and all(isinstance(x, (int, float)) for x in boxes):
                                    # Single box: [x,y,x,y]
                                    result[str(key)] = [[int(coord) for coord in boxes]]
                        return result
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse instance detection JSON: {e}")
            
            # If not valid JSON, return empty dict
            logger.warning(f"Instance detection output not in expected JSON format: {cleaned_output[:100]}...")
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to parse instance detection output: {cleaned_output}. Error: {e}")
            return {}
    
    elif "detection" in task_type_normalized:
        # Detection - extract bounding boxes
        try:
            # Handle JSON format detection references like:
            # {"1": [[x1,y1,w1,h1], [x2,y2,w2,h2]], "2": [[x3,y3,w3,h3]], ...}
            
            # First try to parse as JSON (for ground truth references)
            if cleaned_output.startswith("{") and cleaned_output.endswith("}"):
                try:
                    json_data = json.loads(cleaned_output)
                    if isinstance(json_data, dict):
                        # Extract all bounding boxes from all keys
                        all_boxes = []
                        for key, boxes in json_data.items():
                            if isinstance(boxes, list):
                                for box in boxes:
                                    if isinstance(box, list) and len(box) == 4:
                                        all_boxes.append([int(coord) for coord in box])
                        return all_boxes
                except json.JSONDecodeError:
                    pass
            
            # Detection expects [x_min, y_min, x_max, y_max] format (same as instance detection)
            # Try the exact format: [[x_min, y_min, x_max, y_max], ...]
            bracket_pattern = r'\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]'
            matches = re.findall(bracket_pattern, cleaned_output)
            if matches:
                return [[int(x_min), int(y_min), int(x_max), int(y_max)] for x_min, y_min, x_max, y_max in matches]
            
            # Try to find any list of 4 numbers in [x_min, y_min, x_max, y_max] format
            number_pattern = r'\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?'
            matches = re.findall(number_pattern, cleaned_output)
            if matches:
                return [[int(x_min), int(y_min), int(x_max), int(y_max)] for x_min, y_min, x_max, y_max in matches]
            
            return []
        except Exception as e:
            logger.warning(f"Failed to parse detection output: {cleaned_output}. Error: {e}")
            return []
    
    elif "counting" in task_type_normalized:
        # Cell Counting - extract count number
        try:
            answer_part = cleaned_output
            
            # First, check if this looks like an incomplete instruction/explanation
            incomplete_indicators = [
                "to accurately count", "to provide an accurate count", "to count the cells",
                "we need to", "i would need to", "to determine", "to estimate",
                "follow these steps", "carefully examine", "inspect each"
            ]
            
            if any(indicator in answer_part.lower() for indicator in incomplete_indicators):
                logger.debug(f"Detected incomplete counting response: {answer_part[:100]}...")
                return None
            
            # Look for explicit count statements first
            explicit_patterns = [
                r'count.*?(?:is|are|equals?|totals?)\s*:?\s*(\d+)',
                r'(?:total|number|count)\s*(?:is|are|equals?|of)?\s*:?\s*(\d+)',
                r'(\d+)\s*cells?\s*(?:visible|present|identified|counted)',
                r'approximately\s*(\d+)',
                r'estimated?\s*(\d+)'
            ]
            
            for pattern in explicit_patterns:
                matches = re.findall(pattern, answer_part.lower())
                if matches:
                    # Take the largest number if multiple matches (likely the actual count)
                    numbers = [int(m) for m in matches]
                    return max(numbers)
            
            # Only extract standalone numbers, avoid numbered lists
            # Look for numbers not preceded by step indicators
            number_pattern = r'(?<!step\s)(?<!\d\.\s)(?<!procedure\s)(\d+)(?!\.\s)'
            matches = re.findall(number_pattern, cleaned_output, re.IGNORECASE)
            
            if matches:
                # Convert to integers and filter out obvious outliers for cell counting
                numbers = []
                for match in matches:
                    num = int(match)
                    # Filter out obvious non-count numbers (like procedure steps 1-10)
                    if num > 10 or (num <= 10 and len(matches) == 1):  # Allow small numbers only if it's the only number
                        numbers.append(num)
                
                if numbers:
                    # For counting, usually the largest reasonable number is the count
                    return max(numbers)
            
            # Fallback: try to convert the whole cleaned output
            try:
                return int(float(cleaned_output))
            except ValueError:
                pass
                
            return None  # Could not parse any count
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse counting answer: '{cleaned_output}'. Error: {e}")
            return None
    
    elif "regression" in task_type_normalized:
        # Regression - extract numeric value (can be float)
        try:
            # Look for percentage values first (common in regression tasks)
            percentage_pattern = r'(-?\d+\.?\d*)\s*%'
            matches = re.findall(percentage_pattern, cleaned_output)
            if matches:
                return float(matches[0])
            
            # Look for any decimal number
            number_pattern = r'(-?\d+\.?\d*)'
            matches = re.findall(number_pattern, cleaned_output)
            if matches:
                return float(matches[0])
            
            return float(cleaned_output)
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse regression answer: '{cleaned_output}'. Error: {e}")
            return None
    
    elif "report" in task_type_normalized or "generation" in task_type_normalized:
        # Report Generation - return cleaned text
        return cleaned_output.strip()
    
    # Default case - return cleaned output
    return cleaned_output

def calculate_metrics(predictions, references, task_type):
    """Calculate metrics based on task type - using metrics from the FLARE25 specification"""
    # Handle NaN task_type values
    if task_type is None or (hasattr(task_type, '__class__') and task_type.__class__.__name__ == 'float' and str(task_type) == 'nan'):
        logger.warning(f"Invalid task_type: {task_type}, defaulting to 'classification'")
        task_type = "classification"
    
    # Ensure task_type is a string
    task_type = str(task_type)
    
    # Normalize task type for consistent metric calculation
    task_type_normalized = task_type.lower().strip().replace("_", " ").replace("-", " ")
    
    if "classification" in task_type_normalized and "multi" not in task_type_normalized:
        # Classification (Disease Diagnosis) - PRIMARY METRIC: Balanced Accuracy
        try:
            if not predictions or not references:
                logger.warning("Empty predictions or references for classification task")
                return {"balanced_accuracy": 0.0, "accuracy": 0.0, "valid_samples": 0, "f1_score": 0.0}
            
            # Simple label normalization - just clean and compare directly
            def _normalize_label(lbl):
                if lbl is None:
                    return ""
                
                s = str(lbl).strip()
                
                # Basic cleanup - normalize whitespace
                s = ' '.join(s.split())
                
                # If it's a single letter, make it uppercase (for A-E format)
                if len(s) == 1 and s.isalpha():
                    return s.upper()
                
                # Otherwise, lowercase for consistency (for descriptive format)
                return s.lower()
            
            cleaned_preds = [_normalize_label(p) for p in predictions]
            cleaned_refs = [_normalize_label(r) for r in references]
            
            # Calculate metrics - Balanced Accuracy is the PRIMARY metric for classification
            balanced_acc = balanced_accuracy_score(cleaned_refs, cleaned_preds)
            accuracy = accuracy_score(cleaned_refs, cleaned_preds)
            
            # Calculate F1 score as secondary metric
            unique_classes = sorted(list(set(cleaned_refs + cleaned_preds)))
            if len(unique_classes) <= 2:  # Binary classification
                f1 = f1_score(cleaned_refs, cleaned_preds, average='binary', zero_division=0)
            else:  # Multi-class
                f1 = f1_score(cleaned_refs, cleaned_preds, average='weighted', zero_division=0)
                
            # Log class distribution for debugging
            class_distribution = {cls: cleaned_refs.count(cls) for cls in unique_classes}
            pred_distribution = {cls: cleaned_preds.count(cls) for cls in unique_classes}
            logger.info(f"Class distribution in references: {class_distribution}")
            logger.info(f"Class distribution in predictions: {pred_distribution}")
            
            return {
                "balanced_accuracy": balanced_acc,  # PRIMARY METRIC
                "accuracy": accuracy, 
                "f1_score": f1,
                "valid_samples": len(predictions)
            }
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}", exc_info=True)
            return {"balanced_accuracy": 0.0, "accuracy": 0.0, "valid_samples": 0, "f1_score": 0.0}
    
    elif "multi" in task_type_normalized and "label" in task_type_normalized:
        # Multi-label Classification - PRIMARY METRIC: F1 Score
        try:
            if not predictions or not references:
                logger.warning("Empty predictions or references for multi-label task")
                return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
            
            # Convert predictions and references to sets for comparison
            f1_scores = []
            precisions = []
            recalls = []
            
            for pred, ref in zip(predictions, references):
                # Ensure they are lists
                if not isinstance(pred, list):
                    pred = [pred] if pred else []
                if not isinstance(ref, list):
                    ref = [ref] if ref else []
                
                pred_set = set(str(p).strip() for p in pred if p)
                ref_set = set(str(r).strip() for r in ref if r)
                
                # Calculate metrics for this sample
                if len(ref_set) == 0:
                    continue
                    
                tp = len(pred_set & ref_set)
                fp = len(pred_set - ref_set)
                fn = len(ref_set - pred_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            return {
                "f1_score": np.mean(f1_scores) if f1_scores else 0.0,  # PRIMARY METRIC
                "precision": np.mean(precisions) if precisions else 0.0,
                "recall": np.mean(recalls) if recalls else 0.0,
                "valid_samples": len(f1_scores)
            }
        except Exception as e:
            logger.error(f"Error calculating multi-label metrics: {e}", exc_info=True)
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
    
    elif "instance" in task_type_normalized and "detection" in task_type_normalized:
        # Instance Detection - PRIMARY METRIC: F1 Score matching via IoU > 0.5 (identity-aware)
        def calculate_iou_xmin_ymin_xmax_ymax(box1, box2):
            # Calculate IoU between two bounding boxes in [x_min, y_min, x_max, y_max] format
            try:
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
                
                # Calculate coordinates of intersection
                x_left = max(x1_min, x2_min)
                y_top = max(y1_min, y2_min)
                x_right = min(x1_max, x2_max)
                y_bottom = min(y1_max, y2_max)
                
                # Check if there is no intersection
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                # Calculate areas
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                union_area = box1_area + box2_area - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0
                return iou
            except Exception as e:
                logger.error(f"Error calculating IoU for instance detection: {e}")
                return 0.0
        
        try:
            if not predictions or not references:
                logger.warning("Empty predictions or references for instance detection task")
                return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
            
            # ORIGINAL METRICS (F1@0.5) - KEEPING THESE FOR BACKWARD COMPATIBILITY
            # Calculate TP, FP, FN for each chromosome identity separately
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for i in range(len(references)):
                ref_dict = references[i]
                pred_dict = predictions[i]
                
                # Handle format issues
                if not isinstance(ref_dict, dict):
                    ref_dict = {}
                if not isinstance(pred_dict, dict):
                    pred_dict = {}
                
                # Get all chromosome IDs from both reference and prediction
                all_chromosome_ids = set(ref_dict.keys()) | set(pred_dict.keys())
                
                for chromosome_id in all_chromosome_ids:
                    ref_boxes = ref_dict.get(chromosome_id, [])
                    pred_boxes = pred_dict.get(chromosome_id, [])
                    
                    # Ensure boxes are lists
                    if not isinstance(ref_boxes, list):
                        ref_boxes = []
                    if not isinstance(pred_boxes, list):
                        pred_boxes = []
                    
                    # Match predictions to references for this specific chromosome ID
                    matched_refs = set()
                    matched_preds = set()
                    
                    # Find matches with IoU > 0.5 within the same chromosome ID
                    for j, pred_box in enumerate(pred_boxes):
                        best_iou = 0
                        best_ref_idx = -1
                        
                        for k, ref_box in enumerate(ref_boxes):
                            if k in matched_refs:
                                continue
                            iou = calculate_iou_xmin_ymin_xmax_ymax(pred_box, ref_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_ref_idx = k
                            
                        if best_iou > 0.5:
                            matched_refs.add(best_ref_idx)
                            matched_preds.add(j)
                            total_tp += 1
                    
                    # False positives: unmatched predictions for this chromosome
                    total_fp += len(pred_boxes) - len(matched_preds)
                    
                    # False negatives: unmatched references for this chromosome
                    total_fn += len(ref_boxes) - len(matched_refs)
            
            # Calculate original metrics (F1@0.5)
            precision_original = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall_original = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1_original = 2 * precision_original * recall_original / (precision_original + recall_original) if (precision_original + recall_original) > 0 else 0
            
            logger.info(f"Instance detection original metrics (F1@0.5): TP={total_tp}, FP={total_fp}, FN={total_fn}")
            
            try:
                if len(predictions) > 0 and len(references) > 0:
                    logger.info(f"Calling improved metrics with {len(predictions)} predictions and {len(references)} references")
                    logger.info(f"Sample prediction type: {type(predictions[0])}")
                    logger.info(f"Sample reference type: {type(references[0])}")
                
                improved_metrics = calculate_instance_detection_metrics(
                    predictions, references,
                    iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]  # Medical imaging appropriate thresholds
                )
                
                logger.info(f"Instance detection improved metrics: "
                           f"F1@0.3={improved_metrics['f1_score_at_03']:.3f}, "
                           f"F1@0.5={improved_metrics['f1_score_at_05']:.3f}, "
                           f"Avg_F1={improved_metrics['average_f1']:.3f}")
                
                combined_metrics = {
                    "f1_score": f1_original,  # PRIMARY METRIC (F1@0.5)
                    "precision": precision_original,
                    "recall": recall_original,
                    "valid_samples": len(predictions),
                    
                    "f1_score_at_03": improved_metrics["f1_score_at_03"],
                    "f1_score_at_05": improved_metrics["f1_score_at_05"],  # Computer vision standard
                    "average_f1": improved_metrics["average_f1"],          # COCO-style average
                    "precision_at_03": improved_metrics["precision_at_03"],
                    "recall_at_03": improved_metrics["recall_at_03"],
                    
                    # DETAILED BREAKDOWN
                    "detailed_metrics": improved_metrics["detailed_metrics"],
                    "coco_style_metrics": improved_metrics["coco_style_metrics"],
                    "per_chromosome_metrics": improved_metrics["per_chromosome_metrics"],
                    
                    # METADATA
                    "metrics_version": "v2",
                    "primary_metric_original": "f1_score",  # F1@0.5
                    "primary_metric_improved": "f1_score_at_03"  # F1@0.3 for medical imaging
                }
                
                return combined_metrics
                
            except Exception as e:
                logger.error(f"Error calculating instance detection metrics at IoU 0.3: {e}")
                return {
                    "f1_score": f1_original,
                    "precision": precision_original,
                    "recall": recall_original,
                    "valid_samples": len(predictions),
                    "metrics_version": "v1_original_only",
                    "error": str(e)
                }
            
        except Exception as e:
            logger.error(f"Error calculating instance detection metrics: {e}")
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
    
    elif "detection" in task_type_normalized:
        # Detection - PRIMARY METRIC: F1 Score matching via IoU > 0.5
        def calculate_iou(box1, box2):
            # Calculate IoU between two bounding boxes [x_min, y_min, x_max, y_max] format
            try:
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
                
                # Calculate coordinates of intersection
                x_left = max(x1_min, x2_min)
                y_top = max(y1_min, y2_min)
                x_right = min(x1_max, x2_max)
                y_bottom = min(y1_max, y2_max)
                
                # Check if there is no intersection
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                # Calculate areas
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                union_area = box1_area + box2_area - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0
                return iou
            except Exception as e:
                logger.error(f"Error calculating IoU: {e}")
                return 0.0
        
        try:
            if not predictions or not references:
                logger.warning("Empty predictions or references for detection task")
                return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
            
            # Calculate TP, FP, FN for F1 score
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for i in range(len(references)):
                ref_boxes = references[i]
                pred_boxes = predictions[i]
                
                # Handle format issues
                if not isinstance(ref_boxes, list):
                    ref_boxes = []
                if not isinstance(pred_boxes, list):
                    pred_boxes = []
                
                # Match predictions to references
                matched_refs = set()
                matched_preds = set()
                
                # Find matches with IoU > 0.5
                for j, pred_box in enumerate(pred_boxes):
                    best_iou = 0
                    best_ref_idx = -1
                    
                    for k, ref_box in enumerate(ref_boxes):
                        if k in matched_refs:
                            continue
                        iou = calculate_iou(pred_box, ref_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_ref_idx = k
                    
                    if best_iou > 0.5:
                        matched_refs.add(best_ref_idx)
                        matched_preds.add(j)
                        total_tp += 1
                
                # False positives: unmatched predictions
                total_fp += len(pred_boxes) - len(matched_preds)
                
                # False negatives: unmatched references
                total_fn += len(ref_boxes) - len(matched_refs)
            
            # Calculate metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "f1_score": f1,  # PRIMARY METRIC
                "precision": precision,
                "recall": recall,
                "valid_samples": len(predictions)
            }
        except Exception as e:
            logger.error(f"Error calculating detection metrics: {e}")
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "valid_samples": 0}
    
    elif "counting" in task_type_normalized:
        # Cell Counting - PRIMARY METRIC: Mean Absolute Error
        try:
            if not predictions or not references:
                logger.warning(f"Empty predictions or references for counting task")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            # Filter out None values
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
            
            if not valid_pairs:
                logger.warning(f"No valid prediction-reference pairs for counting task")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            valid_preds, valid_refs = zip(*valid_pairs)
            
            # Convert to numeric values for counting
            try:
                valid_preds = [float(p) for p in valid_preds]
                valid_refs = [float(r) for r in valid_refs]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting counting values to numeric: {e}")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            mae = mean_absolute_error(valid_refs, valid_preds)
            
            # Also calculate RMSE for additional insight
            mse = np.mean([(p - r) ** 2 for p, r in zip(valid_preds, valid_refs)])
            rmse = np.sqrt(mse)
            
            return {
                "mean_absolute_error": mae,  # PRIMARY METRIC
                "root_mean_squared_error": rmse,
                "valid_samples": len(valid_pairs)
            }
        except Exception as e:
            logger.error(f"Error calculating counting metrics: {e}")
            return {"mean_absolute_error": float('inf'), "valid_samples": 0}
    
    elif "regression" in task_type_normalized:
        # Regression - PRIMARY METRIC: Mean Absolute Error
        try:
            if not predictions or not references:
                logger.warning(f"Empty predictions or references for regression task")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            # Filter out None values
            valid_pairs = [(p, r) for p, r in zip(predictions, references) if p is not None and r is not None]
            
            if not valid_pairs:
                logger.warning(f"No valid prediction-reference pairs for regression task")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            valid_preds, valid_refs = zip(*valid_pairs)
            
            # Convert to numeric values
            try:
                valid_preds = [float(p) for p in valid_preds]
                valid_refs = [float(r) for r in valid_refs]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting regression values to numeric: {e}")
                return {"mean_absolute_error": float('inf'), "valid_samples": 0}
            
            mae = mean_absolute_error(valid_refs, valid_preds)
            
            # Also calculate RMSE for additional insight
            mse = np.mean([(p - r) ** 2 for p, r in zip(valid_preds, valid_refs)])
            rmse = np.sqrt(mse)
            
            return {
                "mean_absolute_error": mae,  # PRIMARY METRIC
                "root_mean_squared_error": rmse,
                "valid_samples": len(valid_pairs)
            }
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            return {"mean_absolute_error": float('inf'), "valid_samples": 0}
    
    elif "report" in task_type_normalized or "generation" in task_type_normalized:
        # Report Generation - PRIMARY METRIC: GREEN Score
        try:
            if not predictions or not references:
                logger.warning("Empty predictions or references for report generation task")
                return {"green_score": 0.0, "bleu_score": 0.0, "valid_samples": 0}
            
            # Import advanced metrics
            try:
                from utils.metrics import calculate_green_score, calculate_bleu_score, calculate_clinical_efficacy_score
            except ImportError:
                logger.warning("Advanced metrics not available, using simplified version")
                # Fallback to simple word overlap
                green_scores = []
                
                for pred, ref in zip(predictions, references):
                    if not pred or not ref:
                        continue
                    
                    # Tokenize
                    pred_words = set(pred.lower().split())
                    ref_words = set(ref.lower().split())
                    
                    # Calculate overlap (simplified GREEN score)
                    overlap = len(pred_words & ref_words)
                    total = len(ref_words)
                    
                    score = overlap / total if total > 0 else 0
                    green_scores.append(score)
                
                return {
                    "green_score": np.mean(green_scores) if green_scores else 0.0,  # PRIMARY METRIC
                    "valid_samples": len(green_scores)
                }
            
            # Calculate proper GREEN score
            green_results = calculate_green_score(predictions, references)
            
            # Calculate BLEU score
            bleu_score = calculate_bleu_score(predictions, references)
            
            # Calculate clinical efficacy
            ce_score = calculate_clinical_efficacy_score(predictions, references)
            
            return {
                "green_score": green_results.get("overall_mean", 0.0),  # PRIMARY METRIC
                "green_entity_matching": green_results.get("entity_matching_mean", 0.0),
                "green_location_accuracy": green_results.get("location_accuracy_mean", 0.0),
                "green_negation_handling": green_results.get("negation_handling_mean", 0.0),
                "green_temporal_accuracy": green_results.get("temporal_accuracy_mean", 0.0),
                "green_measurement_accuracy": green_results.get("measurement_accuracy_mean", 0.0),
                "green_clinical_significance": green_results.get("clinical_significance_mean", 0.0),
                "green_structure_completeness": green_results.get("structure_completeness_mean", 0.0),
                "green_severity_correlation": green_results.get("severity_correlation_mean", 0.0),
                "bleu_score": bleu_score,
                "clinical_efficacy": ce_score,
                "valid_samples": len(predictions),
                # Additional statistics
                "avg_ref_entities": green_results.get("avg_ref_entities", 0),
                "avg_gen_entities": green_results.get("avg_gen_entities", 0),
                "most_missed_entities": green_results.get("most_missed_entities", {})
            }
        except Exception as e:
            logger.error(f"Error calculating report generation metrics: {e}")
            return {"green_score": 0.0, "valid_samples": 0}
    
    # Default case for unknown task types
    logger.warning(f"Unknown task type: {task_type}. Using default metrics.")
    return {"valid_samples": 0}

def create_evaluation_prompt(row):
    """Create evaluation prompt that matches training format but WITHOUT the answer"""
    # Handle NaN task_type values
    task_type_raw = row["task_type"]
    if task_type_raw is None or (hasattr(task_type_raw, '__class__') and task_type_raw.__class__.__name__ == 'float' and str(task_type_raw) == 'nan'):
        logger.warning(f"Invalid task_type: {task_type_raw}, defaulting to 'classification'")
        task_type_raw = "classification"
    
    # Same task type normalization as training
    task_type = str(task_type_raw).lower().strip().replace("_", " ").replace("-", " ")
    question = row['question'].strip()
    
    # Same task-specific instructions as training
    if task_type == "classification":
        instruction = "Look at the image carefully and classify what you see. Provide a clear, specific answer."
    elif task_type == "multi label classification":
        instruction = "Examine the image and identify all relevant labels or categories that apply. List them clearly."
    elif task_type == "detection":
        instruction = "Examine the image and detect the specified objects or features. Be precise in your response."
    elif task_type == "instance detection":
        instruction = "Look at the image and identify specific instances of the target objects. Provide precise detection results."
    elif task_type == "counting":
        instruction = "Count the specified objects or features in the image. Provide an accurate numerical answer."
    elif task_type == "regression":
        instruction = "Analyze the image and provide a quantitative assessment. Give your answer with appropriate precision."
    elif task_type == "report generation":
        instruction = "Examine the medical image thoroughly and generate a report describing your findings."
    else:
        instruction = "Look at the image and answer the question accurately based on what you observe."
        logger.warning(f"Unknown task type '{row['task_type']}' (normalized: '{task_type}') - using default instruction")
    
    # Create exact same format as training but stop at user message
    full_question = f"{instruction}\n\n{question}"
    
    # Create content array with all images following Qwen2.5-VL format
    content = []
    
    # Add all images to content
    images = row.get('images', [])
    for image_path in images:
        content.append({"type": "image", "image": image_path})
    
    # Add text question
    content.append({"type": "text", "text": full_question})
    
    return {
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

def evaluate_model(args):
    try:
        # Store evaluation start time for summary
        eval_start_time = datetime.datetime.now()
        
        # Set up file logging immediately
        timestamp = eval_start_time.strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join("logs", f"evaluation_{timestamp}.log")
        
        # Add file handler to the logger
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        logger.info("="*60)
        logger.info("STARTING MODEL EVALUATION ON VALIDATION SET")
        logger.info("="*60)
        logger.info(f"Log file: {log_filename}")
        logger.info(f"Arguments: {vars(args)}")
        if args.task_type:
            logger.info(f"🎯 Evaluating only task type: {args.task_type}")
        else:
            logger.info("📋 Evaluating all task types")
        
        # Validate critical arguments early
        if not os.path.exists(args.processed_data_dir):
            logger.error(f"Processed data directory not found: {args.processed_data_dir}")
            logger.error("Please run prepare_data.py first to create the processed dataset")
            raise FileNotFoundError(f"Processed data directory not found: {args.processed_data_dir}")
        
        if args.lora_weights and not os.path.exists(args.lora_weights):
            logger.error(f"LoRA weights not found: {args.lora_weights}")
            logger.error("Please ensure the fine-tuning is complete and weights are saved")
            raise FileNotFoundError(f"LoRA weights not found: {args.lora_weights}")
        
        # Log validation success
        logger.info("✅ Argument validation passed")
        
        # Log GPU device being used
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            logger.info(f"Using GPU device {current_device}: {torch.cuda.get_device_name(current_device)}")
            # Get GPU memory info for dynamic batch sizing
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)  # Convert to GB
            logger.info(f"Total GPU memory: {gpu_memory_gb:.2f} GB")
            
            # Clear cache before starting
            torch.cuda.empty_cache()
            
            # Determine batch size based on available GPU memory
            # Conservative approach: start with a base batch size and adjust
            if gpu_memory_gb > 32:
                dynamic_batch_size = 2  # Reduced from 8
            elif gpu_memory_gb > 24:
                dynamic_batch_size = 1  # Reduced from 4  
            elif gpu_memory_gb > 16:
                dynamic_batch_size = 1  # Reduced from 2
            else:
                dynamic_batch_size = 1
                
            # Override with user-specified batch size if provided
            eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else dynamic_batch_size
            # Safety check: ensure batch size is never zero
            if eval_batch_size <= 0:
                eval_batch_size = 1
                logger.warning("Batch size was set to zero, defaulting to 1")
            
            # For vision-language models, always use batch size 1 to avoid memory issues
            eval_batch_size = 1
            logger.info(f"Using conservative evaluation batch size: {eval_batch_size} (forced to 1 for memory safety)")
        else:
            logger.warning("CUDA not available! Evaluation will be very slow.")
            eval_batch_size = 1
            if not args.force_cpu:
                logger.error("Aborting due to no GPU. Use --force_cpu to override.")
                return
    
        # Load validation dataset (includes both hidden and public validation data)
        logger.info(f"Loading validation dataset from {args.processed_data_dir}")
        
        
        dataset_path = os.path.join(args.processed_data_dir, "validation")
        if not os.path.exists(dataset_path):
            logger.error(f"Validation dataset not found at {dataset_path}")
            logger.error("Please run prepare_data.py first to create the validation dataset")
            raise FileNotFoundError(f"Validation dataset not found at {dataset_path}")
        
        val_dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded validation dataset with {len(val_dataset)} examples")
        logger.info("Note: This includes both hidden and public validation data as processed by prepare_data.py")
        
        model_results = {}
        
        models_to_evaluate = []
        
        # Add base model if requested
        if args.evaluate_base_model:
            models_to_evaluate.append({
                "name": "base_model",
                "model_path": args.base_model_path,
                "use_lora": False,
                "lora_weights": None,
                "output_dir": os.path.join(args.output_dir, "base_model")
            })
        
        # Add fine-tuned model
        models_to_evaluate.append({
            "name": "finetuned_model",
            "model_path": args.model_name_or_path,
            "use_lora": args.lora_weights is not None,
            "lora_weights": args.lora_weights,
            "output_dir": os.path.join(args.output_dir, "finetuned_model")
        })
        
        # Evaluate each model
        for model_config in models_to_evaluate:
            logger.info(f"Starting evaluation for {model_config['name']}")
            
            # Create output directory
            os.makedirs(model_config['output_dir'], exist_ok=True)
            
            # Load model
            logger.info(f"Loading model from {model_config['model_path']}")
            if model_config['use_lora']:
                # Load base model
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_config['model_path'],
                    torch_dtype=torch.bfloat16,
                    device_map={"": current_device} if torch.cuda.is_available() else "auto",
                    trust_remote_code=True,
                    attn_implementation="sdpa"  # Use scaled dot product attention
                )
                # Load LoRA weights
                logger.info(f"Loading LoRA weights from {model_config['lora_weights']}")
                model = PeftModel.from_pretrained(model, model_config['lora_weights'])
            else:
                # Load model directly
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_config['model_path'],
                    torch_dtype=torch.bfloat16,
                    device_map={"": current_device} if torch.cuda.is_available() else "auto",
                    trust_remote_code=True,
                    attn_implementation="sdpa"  # Use scaled dot product attention
                )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['model_path'],
                trust_remote_code=True,
                padding_side='left'  # Explicitly set for decoder-only models
            )
            
            # Ensure padding side is set to left for decoder-only models
            tokenizer.padding_side = 'left'
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token # Common practice for decoder-only
                logger.info(f"Tokenizer pad_token not set. Using eos_token as pad_token: {tokenizer.pad_token}")

            processor = AutoProcessor.from_pretrained(
                model_config['model_path'], 
                trust_remote_code=True,
                use_fast=True  # Address slow processor warning
            )
            
            # Ensure our generation kwargs override any problematic defaults
            # For Qwen2.5-VL, we'll use explicit kwargs to avoid warnings
            logger.info("Model loaded. Will use explicit generation kwargs to ensure deterministic evaluation.")
            
            # Group validation examples by task type
            task_groups = {}
            for example in val_dataset:
                task_type_raw = example["task_type"]
                
                # Better task type normalization to match training script and image
                task_type = task_type_raw.lower().strip().replace("_", " ").replace("-", " ")
                
                # Normalize to standard task types with correct metrics
                if "classification" in task_type and ("multi" in task_type or "label" in task_type):
                    normalized_task = "multi-label"  # F1 Score
                elif "classification" in task_type:
                    normalized_task = "classification"  # Balanced Accuracy  
                elif "instance" in task_type and "detection" in task_type:
                    normalized_task = "instance_detection"  # F1 Score matching via IoU > 0.5 (identity-aware)
                elif "detection" in task_type:
                    normalized_task = "detection"  # F1 Score via IoU > 0.5
                elif "counting" in task_type:
                    normalized_task = "counting"  # Mean Absolute Error
                elif "regression" in task_type:
                    normalized_task = "regression"  # Mean Absolute Error
                elif "report" in task_type or "generation" in task_type:
                    normalized_task = "report_generation"  # GREEN Score
                else:
                    # Keep original for logging but normalize spaces/underscores
                    normalized_task = task_type
                    logger.warning(f"Unknown task type encountered: '{task_type_raw}' -> normalized: '{task_type}'")
                
                if normalized_task not in task_groups:
                    task_groups[normalized_task] = []
                task_groups[normalized_task].append(example)
            
            logger.info(f"Found task types: {list(task_groups.keys())} with counts: {[(k, len(v)) for k, v in task_groups.items()]}")
            
            # Filter task groups if specific task is requested
            if args.task_type:
                if args.task_type in task_groups:
                    filtered_task_groups = {args.task_type: task_groups[args.task_type]}
                    logger.info(f"Filtering to evaluate only task type: {args.task_type} with {len(task_groups[args.task_type])} examples")
                    task_groups = filtered_task_groups
                else:
                    logger.error(f"Requested task type '{args.task_type}' not found in dataset")
                    logger.error(f"Available task types: {list(task_groups.keys())}")
                    raise ValueError(f"Task type '{args.task_type}' not found in dataset")
            
            # Evaluate by task type
            all_results = {}
            
            start_time = time.time()
            
            for task_type, all_task_examples in task_groups.items():
                num_total_task_examples = len(all_task_examples)
                
                # Apply max_eval_samples limit if specified
                current_task_examples = all_task_examples
                if args.max_eval_samples > 0 and num_total_task_examples > args.max_eval_samples:
                    logger.info(f"Evaluating task: {task_type} with {args.max_eval_samples} samples (out of {num_total_task_examples}) for model {model_config['name']}")
                    current_task_examples = all_task_examples[:args.max_eval_samples]
                else:
                    logger.info(f"Evaluating task: {task_type} with all {len(current_task_examples)} examples for model {model_config['name']}")

                task_predictions = []
                task_references = []
                task_details = []
                task_skipped_count = 0

                # Double-check that batch size is not zero before using in range()
                current_batch_size = eval_batch_size
                if current_batch_size <= 0:
                    current_batch_size = 1
                    logger.warning(f"Batch size for task {task_type} was invalid, using batch size 1 instead")

                for i in tqdm(range(0, len(current_task_examples), current_batch_size), desc=f"Evaluating {task_type} with {model_config['name']}"):
                    batch_examples = current_task_examples[i:i+current_batch_size]
                    batch_prompts = []
                    batch_images_pil = []
                    
                                        # Process examples using official Qwen2.5-VL format
                    batch_messages = []
                    
                    for example in batch_examples:
                        try:
                            # Create evaluation prompt in Qwen2.5-VL message format
                            prompt_data = create_evaluation_prompt(example)
                            batch_messages.append(prompt_data["messages"])
                            
                            # Apply chat template
                            prompt_text = tokenizer.apply_chat_template(
                                prompt_data["messages"],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            batch_prompts.append(prompt_text)
                            
                        except Exception as e_img:
                            image_info = example.get('images', [example.get('image', 'unknown')])
                            logger.error(f"Error loading images {image_info}: {e_img}")
                            task_skipped_count += 1
                            continue
                    
                    if not batch_messages: # All examples in batch failed to process
                        continue

                    try:
                        # Use standard processor approach - manually extract images from messages
                        all_images = []
                        for messages in batch_messages:
                            for message in messages:
                                if message["role"] == "user":
                                    for content in message["content"]:
                                        if content["type"] == "image":
                                            image_path = content["image"]
                                            try:
                                                image = Image.open(image_path).convert("RGB")
                                                # Ensure consistent size for QWen2.5-VL
                                                image = image.resize((448, 448), Image.LANCZOS)
                                                all_images.append(image)
                                            except Exception as img_err:
                                                logger.error(f"Error loading image {image_path}: {img_err}")
                                                continue
                        
                        # Process with the processor using the extracted images
                        data_inputs = processor(
                            text=batch_prompts,
                            images=all_images if all_images else None,
                            return_tensors="pt",
                            padding=True,
                            truncation=True
                        )

                        for k, v in data_inputs.items():
                            if torch.is_tensor(v):
                                data_inputs[k] = v.to(model.device)

                        # Use only essential generation parameters to avoid warnings
                        gen_kwargs = {
                            "max_new_tokens": 150,  # INCREASED from 100 to allow complete responses for counting
                            "do_sample": False,  # Deterministic generation
                            "num_beams": 1,      # Greedy decoding
                            "repetition_penalty": 1.0,
                            "use_cache": True,
                            "pad_token_id": tokenizer.pad_token_id,  # Ensure pad token is set
                        }
                        
                        # Increase tokens for instance detection due to complex JSON output
                        if task_type == "instance_detection":
                            gen_kwargs["max_new_tokens"] = 10000  # Allow complete chromosome inventory JSON
                            
                        
                        # Filter gen_kwargs to only include parameters that the model's generate method accepts
                        try:
                            generate_signature = inspect.signature(model.generate)
                            valid_params = set(generate_signature.parameters.keys())
                            filtered_gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in valid_params}
                            if task_type == "instance_detection":
                                # For instance detection, keep original parameters to ensure complete JSON generation
                                logger.debug("Using full generation parameters for instance detection")
                                gen_kwargs = gen_kwargs  # Keep original parameters
                            else:
                                logger.debug(f"Filtered generation kwargs: {list(filtered_gen_kwargs.keys())}")
                                gen_kwargs = filtered_gen_kwargs
                        except Exception as e:
                            logger.debug(f"Could not filter generation kwargs: {e}. Using all kwargs.")
                        
                        with torch.no_grad():
                            # Clear cache before generation
                            torch.cuda.empty_cache()
                            
                            generated_ids_batch = model.generate(
                                input_ids=data_inputs["input_ids"],
                                attention_mask=data_inputs.get("attention_mask"),
                                pixel_values=data_inputs.get("pixel_values"),
                                image_grid_thw=data_inputs.get("image_grid_thw"),
                                **gen_kwargs,
                            )
                            
                            # Clear cache after generation
                            torch.cuda.empty_cache()

                        if generated_ids_batch is None:
                            logger.error(f"Model generation failed for a batch. Skipping {len(batch_examples)} examples.")
                            task_skipped_count += len(batch_examples)
                            continue

                        outputs_batch_decoded = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

                        for j, decoded_output in enumerate(outputs_batch_decoded):
                            original_example = batch_examples[j]
                            raw_reference_single = original_example["answer"]

                            parsed_output_single = parse_answer(decoded_output, task_type)
                            sample_valid = True
                            parsed_reference_single = None # Initialize

                            try:
                                if task_type.lower() == "detection":
                                    # Parse detection reference which may be in JSON format
                                    parsed_reference_single = parse_answer(raw_reference_single, "detection")
                                    if not isinstance(parsed_reference_single, list):
                                        logger.warning(f"Could not parse detection reference '{raw_reference_single}' to list of bounding boxes. Skipping.")
                                        task_skipped_count +=1
                                        sample_valid = False
                                elif task_type.lower() == "instance_detection":
                                    # Parse instance detection reference which should be in JSON format
                                    parsed_reference_single = parse_answer(raw_reference_single, "instance detection")
                                    if not isinstance(parsed_reference_single, dict):
                                        logger.warning(f"Could not parse instance detection reference '{raw_reference_single}' to dict of chromosome mappings. Skipping.")
                                        task_skipped_count += 1
                                        sample_valid = False
                                elif task_type.lower() == "regression" or "regression" in task_type.lower():
                                    try:
                                        parsed_reference_single = float(raw_reference_single)
                                        if parsed_output_single is None:
                                            logger.warning(f"Could not parse model output for regression. Q: {original_example['question']}, Output: {decoded_output}")
                                            task_skipped_count += 1
                                            sample_valid = False
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not parse regression reference '{raw_reference_single}' to float. Skipping.")
                                        task_skipped_count += 1
                                        sample_valid = False
                                elif "multi" in task_type.lower() and "label" in task_type.lower():
                                    # Parse multi-label reference using the same logic as predictions
                                    parsed_reference_single = parse_answer(raw_reference_single, task_type)
                                    if not isinstance(parsed_reference_single, list):
                                        parsed_reference_single = [parsed_reference_single] if parsed_reference_single else []
                                elif "counting" in task_type.lower():
                                    try:
                                        parsed_reference_single = int(float(raw_reference_single))
                                        if parsed_output_single is None:
                                            logger.warning(f"Could not parse model output for counting. Q: {original_example['question']}, Output: {decoded_output}")
                                            task_skipped_count += 1
                                            sample_valid = False
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not parse counting reference '{raw_reference_single}' to int. Skipping.")
                                        task_skipped_count += 1
                                        sample_valid = False
                                else: # Single-label Classification
                                    parsed_reference_single = raw_reference_single
                            except (ValueError, SyntaxError) as e_ref_parse:
                                logger.error(f"Error parsing reference '{raw_reference_single}' for task {task_type}: {e_ref_parse}. Skipping.")
                                task_skipped_count += 1
                                sample_valid = False

                            if sample_valid:
                                task_predictions.append(parsed_output_single)
                                task_references.append(parsed_reference_single)

                            if args.save_predictions:
                                task_details.append({
                                    "question": original_example["question"],
                                    "model_output": decoded_output,
                                    "parsed_output": str(parsed_output_single),
                                    "reference": str(parsed_reference_single),
                                    "valid_for_metrics": sample_valid,
                                    "image_path": original_example.get("images", [original_example.get("image", "unknown")])
                                })

                            # Log occasional examples (consider adjusting frequency for batches)
                            if ((i + j) % 20 == 0) or not sample_valid: # Log every 20 samples
                                log_level = logging.DEBUG if sample_valid else logging.INFO
                                logger.log(log_level, f"Example {len(task_predictions) + task_skipped_count}")
                                logger.log(log_level, f"  Question: {original_example['question']}")
                                logger.log(log_level, f"  Raw output: {decoded_output}")
                                logger.log(log_level, f"  Parsed output: {parsed_output_single}")
                                logger.log(log_level, f"  Reference: {parsed_reference_single}")
                                logger.log(log_level, f"  Valid for metrics: {sample_valid}")

                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"CUDA out of memory for batch in task {task_type}. Skipping batch with {len(batch_examples)} examples.")
                        logger.error(f"OOM Error: {str(e)}")
                        # Clear cache aggressively
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        task_skipped_count += len(batch_examples)
                        continue
                    except Exception as e_batch_proc:
                        logger.error(f"Error processing batch for task {task_type}: {e_batch_proc}", exc_info=True)
                        task_skipped_count += len(batch_examples)
                        continue # Skip to next batch
                
                # Calculate metrics for this task
                metrics = calculate_metrics(task_predictions, task_references, task_type)
                
                result = {
                    "metrics": metrics,
                    "num_examples": len(current_task_examples),
                    "valid_examples": len(task_predictions),
                    "skipped_examples": task_skipped_count
                }
                
                # Add detailed predictions if requested
                if args.save_predictions and task_details:
                    result["predictions"] = task_details
                    
                all_results[task_type] = result
                logger.info(f"Task {task_type} results for {model_config['name']}: {metrics}")
                logger.info(f"Valid examples: {len(task_predictions)}/{len(current_task_examples)} (skipped: {task_skipped_count})")
            
            # Add overall stats
            elapsed_time = time.time() - start_time
            all_results["overall"] = {
                "total_examples": sum(result["num_examples"] for task, result in all_results.items() if task != "overall"),
                "valid_examples": sum(result["valid_examples"] for task, result in all_results.items() if task != "overall"),
                "skipped_examples": sum(result["skipped_examples"] for task, result in all_results.items() if task != "overall"),
                "evaluation_time": str(datetime.timedelta(seconds=int(elapsed_time))),
                "evaluation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save results
            results_file = os.path.join(model_config['output_dir'], "evaluation_results.json")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, cls=InfinityEncoder)
            
            logger.info(f"Evaluation for {model_config['name']} complete. Results saved to {results_file}")
            
            # Store results for comparison
            model_results[model_config['name']] = all_results
            
            # Free up memory
            del model
            del processor
            del tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            logger.info(f"Memory cleanup completed for {model_config['name']}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU memory allocated after cleanup: {allocated:.2f} GB")
        
        # Compare models if multiple models were evaluated
        if len(model_results) > 1 and "base_model" in model_results and "finetuned_model" in model_results:
            logger.info("\n" + "="*60)
            logger.info("GENERATING BASE vs FINE-TUNED COMPARISON")
            logger.info("="*60)
            
            comparison = {}
            overall_improvements = {
                "improved_tasks": 0,
                "total_tasks": 0,
                "task_details": {}
            }
            
            # Get tasks to compare (either all tasks or just the specified one)
            tasks_to_compare = list(model_results["base_model"].keys())
            tasks_to_compare = [t for t in tasks_to_compare if t != "overall"]
            
            if args.task_type:
                logger.info(f"Comparing only task type: {args.task_type}")
            
            for task_type in tasks_to_compare:
                if task_type != "overall":
                    base_metrics = model_results["base_model"][task_type].get("metrics", {})
                    finetuned_metrics = model_results["finetuned_model"][task_type].get("metrics", {})
                    
                    # Get task-specific primary metric
                    primary_metric_map = {
                        "classification": "balanced_accuracy",
                        "multi-label": "f1_score", 
                        "detection": "f1_score",
                        "instance_detection": "f1_score",
                        "counting": "mean_absolute_error",
                        "regression": "mean_absolute_error",
                        "report_generation": "green_score"
                    }
                    
                    primary_metric = primary_metric_map.get(task_type, None)
                    
                    # Compare metrics
                    metric_comparison = {}
                    for metric_name in set(base_metrics.keys()) & set(finetuned_metrics.keys()):
                        if isinstance(base_metrics[metric_name], (int, float)) and isinstance(finetuned_metrics[metric_name], (int, float)):
                            base_value = float(base_metrics[metric_name])
                            finetuned_value = float(finetuned_metrics[metric_name])
                            diff = finetuned_value - base_value
                            rel_diff = (diff / base_value * 100) if base_value != 0 else None
                            
                            # Determine improvement based on metric type
                            if metric_name in ["mean_absolute_error", "root_mean_squared_error"]:
                                # Lower is better
                                improvement = bool(diff < 0)
                                # Handle special cases for magnitude calculation
                                if base_value == 0:
                                    # When base is 0, use absolute difference as magnitude
                                    improvement_magnitude = abs(diff) * 100  # Scale to percentage-like value
                                elif base_value == float('inf') or base_value == float('-inf') or base_value != base_value:
                                    # When base is infinity or NaN, can't calculate meaningful percentage
                                    improvement_magnitude = float('inf') if diff != 0 else 0
                                else:
                                    improvement_magnitude = abs(diff / base_value * 100)
                            else:
                                # Higher is better
                                improvement = bool(diff > 0)
                                # Handle special cases for magnitude calculation
                                if base_value == 0:
                                    # When base is 0, use absolute difference as magnitude
                                    improvement_magnitude = abs(diff) * 100  # Scale to percentage-like value
                                elif base_value == float('inf') or base_value == float('-inf') or base_value != base_value:
                                    # When base is infinity or NaN, can't calculate meaningful percentage
                                    improvement_magnitude = float('inf') if diff != 0 else 0
                                else:
                                    improvement_magnitude = abs(diff / base_value * 100)
                            
                            metric_comparison[metric_name] = {
                                "base": base_value,
                                "finetuned": finetuned_value,
                                "absolute_diff": diff,
                                "relative_diff_percent": rel_diff,
                                "improvement": improvement,
                                "improvement_magnitude_percent": improvement_magnitude,
                                "is_primary_metric": metric_name == primary_metric
                            }
                    
                    comparison[task_type] = {
                        "metrics": metric_comparison,
                        "primary_metric": primary_metric,
                        "base_samples": model_results["base_model"][task_type].get("valid_examples", 0),
                        "finetuned_samples": model_results["finetuned_model"][task_type].get("valid_examples", 0)
                    }
                    
                    # Track overall improvements using primary metric
                    if primary_metric and primary_metric in metric_comparison:
                        overall_improvements["total_tasks"] += 1
                        primary_improved = metric_comparison[primary_metric]["improvement"]
                        if primary_improved:
                            overall_improvements["improved_tasks"] += 1
                        
                        overall_improvements["task_details"][task_type] = {
                            "primary_metric": primary_metric,
                            "improved": primary_improved,
                            "improvement_magnitude": metric_comparison[primary_metric]["improvement_magnitude_percent"]
                        }
            
            # Add overall summary to comparison
            if overall_improvements["total_tasks"] > 0:
                improvement_rate = (overall_improvements["improved_tasks"] / overall_improvements["total_tasks"]) * 100
                overall_improvements["improvement_rate_percent"] = improvement_rate
            
            comparison["overall_summary"] = overall_improvements
            
            # Save comparison
            comparison_file = os.path.join(args.output_dir, "model_comparison.json")
            with open(comparison_file, "w") as f:
                json.dump(comparison, f, indent=2, cls=InfinityEncoder)
            
            # Create detailed comparison report
            create_detailed_comparison_report(comparison, comparison_file, eval_start_time, args)
                
            # Log comparison summary
            logger.info("\n========== MODEL COMPARISON SUMMARY ==========")
            for task_type, task_data in comparison.items():
                if task_type == "overall_summary":
                    continue
                    
                logger.info(f"\nTask: {task_type.upper()}")
                logger.info(f"Samples: Base={task_data['base_samples']}, Fine-tuned={task_data['finetuned_samples']}")
                
                metrics = task_data["metrics"]
                primary_metric = task_data["primary_metric"]
                
                # Show primary metric first if available
                if primary_metric and primary_metric in metrics:
                    values = metrics[primary_metric]
                    base_val = values["base"]
                    finetuned_val = values["finetuned"]
                    abs_diff = abs(values["absolute_diff"])
                    improvement_mag = values["improvement_magnitude_percent"]
                    
                    # Handle infinity and NaN values for display
                    if base_val == float('inf') or base_val == float('-inf') or base_val != base_val:
                        base_str = str(base_val)
                    else:
                        base_str = f"{base_val:.4f}"
                    
                    if finetuned_val == float('inf') or finetuned_val == float('-inf') or finetuned_val != finetuned_val:
                        finetuned_str = str(finetuned_val)
                    else:
                        finetuned_str = f"{finetuned_val:.4f}"
                    
                    if abs_diff == float('inf') or abs_diff == float('-inf') or abs_diff != abs_diff:
                        diff_str = str(abs_diff)
                    else:
                        diff_str = f"{abs_diff:.4f}"
                    
                    if improvement_mag == float('inf') or improvement_mag == float('-inf') or improvement_mag != improvement_mag:
                        mag_str = str(improvement_mag)
                    else:
                        mag_str = f"{improvement_mag:.1f}%"
                    
                    if primary_metric in ["mean_absolute_error", "root_mean_squared_error"]:
                        # Lower is better metrics
                        direction = "↓" if values["absolute_diff"] < 0 else "↑"
                        improvement_indicator = " (IMPROVED)" if values["improvement"] else " (WORSE)"
                        logger.info(f"  🎯 {primary_metric} (PRIMARY): {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str}){improvement_indicator}")
                    else:
                        # Higher is better metrics
                        direction = "↑" if values["improvement"] else "↓"
                        improvement_indicator = " (IMPROVED)" if values["improvement"] else " (WORSE)"
                        logger.info(f"  🎯 {primary_metric} (PRIMARY): {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str}){improvement_indicator}")
                
                # Show other metrics
                for metric_name, values in metrics.items():
                    if metric_name == primary_metric:
                        continue  # Already shown above
                    
                    base_val = values["base"]
                    finetuned_val = values["finetuned"]
                    abs_diff = abs(values["absolute_diff"])
                    improvement_mag = values["improvement_magnitude_percent"]
                    
                    # Handle infinity and NaN values for display
                    if base_val == float('inf') or base_val == float('-inf') or base_val != base_val:
                        base_str = str(base_val)
                    else:
                        base_str = f"{base_val:.4f}"
                    
                    if finetuned_val == float('inf') or finetuned_val == float('-inf') or finetuned_val != finetuned_val:
                        finetuned_str = str(finetuned_val)
                    else:
                        finetuned_str = f"{finetuned_val:.4f}"
                    
                    if abs_diff == float('inf') or abs_diff == float('-inf') or abs_diff != abs_diff:
                        diff_str = str(abs_diff)
                    else:
                        diff_str = f"{abs_diff:.4f}"
                    
                    if improvement_mag == float('inf') or improvement_mag == float('-inf') or improvement_mag != improvement_mag:
                        mag_str = str(improvement_mag)
                    else:
                        mag_str = f"{improvement_mag:.1f}%"
                        
                    if metric_name in ["accuracy", "f1_score", "balanced_accuracy", "precision", "recall", "green_score"]:
                        # Higher is better metrics
                        direction = "↑" if values["improvement"] else "↓"
                        logger.info(f"  {metric_name}: {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str})")
                    elif metric_name in ["mean_absolute_error", "root_mean_squared_error"]:
                        # Lower is better metrics
                        direction = "↓" if values["absolute_diff"] < 0 else "↑"
                        logger.info(f"  {metric_name}: {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str})")
            logger.info("\n==============================================")
            
            # Log overall performance summary
            summary = comparison["overall_summary"]
            logger.info("\n========== OVERALL PERFORMANCE SUMMARY ==========")
            if summary["total_tasks"] > 0:
                improvement_rate = summary["improvement_rate_percent"]
                logger.info(f"🏆 Fine-tuning improved {summary['improved_tasks']}/{summary['total_tasks']} tasks ({improvement_rate:.1f}%)")
                
                # Show task-by-task improvement details
                logger.info("\nTask-by-task improvement summary:")
                for task_name, details in summary["task_details"].items():
                    status = "✅ IMPROVED" if details["improved"] else "❌ WORSE"
                    magnitude = details["improvement_magnitude"]
                    primary = details["primary_metric"]
                    
                    # Handle infinity and NaN values for display
                    if magnitude == float('inf') or magnitude == float('-inf') or magnitude != magnitude:
                        magnitude_str = str(magnitude)
                    else:
                        magnitude_str = f"{magnitude:.1f}%"
                    
                    logger.info(f"  {task_name}: {status} ({magnitude_str} change in {primary})")
                    
            logger.info("================================================")
            
            logger.info(f"\n📊 Comparison saved to: {comparison_file}")
        
        elif len(model_results) == 1:
            logger.warning("Only one model evaluated - no comparison possible")
            logger.info("To enable comparison, run with base model evaluation enabled (default behavior)")
        
        else:
            logger.warning("Base model and/or fine-tuned model results missing - comparison not possible")
        
        logger.info(f"Total evaluation time: {datetime.timedelta(seconds=int(time.time() - start_time))}")
        
        # Create evaluation summary
        create_evaluation_summary(args, model_results, eval_start_time, log_filename)
        
        # Return results
        return model_results

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def create_evaluation_summary(args, model_results, start_time, log_filename):
    """Create a summary file of evaluation results in the logs folder"""
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    summary_filename = os.path.join("logs", f"evaluation_summary_{timestamp}.txt")
    
    with open(summary_filename, "w") as f:
        f.write("Evaluation Summary\n")
        f.write("=================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        if args.lora_weights:
            f.write(f"LoRA weights: {args.lora_weights}\n")
        f.write(f"Base model evaluated: {args.evaluate_base_model}\n")
        if args.evaluate_base_model:
            f.write(f"Base model path: {args.base_model_path}\n")
        f.write(f"Dataset: {args.processed_data_dir}\n")
        f.write(f"Max evaluation samples: {args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'}\n")
        if args.task_type:
            f.write(f"Task type filter: {args.task_type} (evaluating only this task)\n")
        else:
            f.write("Task type filter: None (evaluating all tasks)\n")
        f.write("\n")
        
        if args.evaluate_base_model and "base_model" in model_results and "finetuned_model" in model_results:
            f.write("Performance Comparison\n")
            f.write("=====================\n")
            for task_type in model_results["base_model"].keys():
                if task_type != "overall":
                    f.write(f"\nTask: {task_type}\n")
                    base_metrics = model_results["base_model"][task_type].get("metrics", {})
                    finetuned_metrics = model_results["finetuned_model"][task_type].get("metrics", {})
                    
                    for metric_name in set(base_metrics.keys()) & set(finetuned_metrics.keys()):
                        if isinstance(base_metrics[metric_name], (int, float)) and isinstance(finetuned_metrics[metric_name], (int, float)):
                            base_value = float(base_metrics[metric_name])
                            finetuned_value = float(finetuned_metrics[metric_name])
                            diff = finetuned_value - base_value
                            direction = "↑" if diff > 0 else "↓"
                            f.write(f"  {metric_name}: {base_value:.4f} → {finetuned_value:.4f} ({direction} {abs(diff):.4f})\n")
        
        # Add individual model results
        for model_name, results in model_results.items():
            f.write(f"\n{model_name.upper()} Results\n")
            f.write("=" * (len(model_name) + 8) + "\n")
            
            for task_type, task_results in results.items():
                if task_type != "overall":
                    f.write(f"\nTask: {task_type}\n")
                    metrics = task_results.get("metrics", {})
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric_name}: {value:.4f}\n")
                    
                    f.write(f"  Valid examples: {task_results.get('valid_examples', 0)}/{task_results.get('num_examples', 0)}\n")
                    f.write(f"  Skipped examples: {task_results.get('skipped_examples', 0)}\n")
    
    logger.info(f"Evaluation summary saved to {summary_filename}")

def create_detailed_comparison_report(comparison, report_file, eval_start_time, args):
    """Create a detailed comparison report in the logs folder"""
    timestamp = eval_start_time.strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join("logs", f"detailed_comparison_report_{timestamp}.txt")
    
    with open(report_filename, "w") as f:
        f.write("Detailed Comparison Report\n")
        f.write("==========================\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_name_or_path}\n")
        if args.lora_weights:
            f.write(f"LoRA weights: {args.lora_weights}\n")
        f.write(f"Base model evaluated: {args.evaluate_base_model}\n")
        if args.evaluate_base_model:
            f.write(f"Base model path: {args.base_model_path}\n")
        f.write(f"Dataset: {args.processed_data_dir}\n")
        f.write(f"Max evaluation samples: {args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'}\n")
        if args.task_type:
            f.write(f"Task type filter: {args.task_type} (evaluating only this task)\n")
        else:
            f.write("Task type filter: None (evaluating all tasks)\n")
        f.write("\n")
        
        f.write("Comparison Summary\n")
        f.write("==================\n")
        for task_type, task_data in comparison.items():
            if task_type == "overall_summary":
                continue
            
            f.write(f"\nTask: {task_type.upper()}\n")
            f.write("=" * (len(task_type) + 8) + "\n")
            f.write(f"Samples: Base={task_data['base_samples']}, Fine-tuned={task_data['finetuned_samples']}\n\n")
            
            metrics = task_data["metrics"]
            primary_metric = task_data["primary_metric"]
            
            # Show primary metric first if available
            if primary_metric and primary_metric in metrics:
                values = metrics[primary_metric]
                base_val = values["base"]
                finetuned_val = values["finetuned"]
                abs_diff = abs(values["absolute_diff"])
                improvement_mag = values["improvement_magnitude_percent"]
                
                # Handle infinity and NaN values for display
                if base_val == float('inf') or base_val == float('-inf') or base_val != base_val:
                    base_str = str(base_val)
                else:
                    base_str = f"{base_val:.4f}"
                
                if finetuned_val == float('inf') or finetuned_val == float('-inf') or finetuned_val != finetuned_val:
                    finetuned_str = str(finetuned_val)
                else:
                    finetuned_str = f"{finetuned_val:.4f}"
                
                if abs_diff == float('inf') or abs_diff == float('-inf') or abs_diff != abs_diff:
                    diff_str = str(abs_diff)
                else:
                    diff_str = f"{abs_diff:.4f}"
                
                if improvement_mag == float('inf') or improvement_mag == float('-inf') or improvement_mag != improvement_mag:
                    mag_str = str(improvement_mag)
                else:
                    mag_str = f"{improvement_mag:.1f}%"
                
                if primary_metric in ["mean_absolute_error", "root_mean_squared_error"]:
                    # Lower is better metrics
                    direction = "↓" if values["absolute_diff"] < 0 else "↑"
                    improvement_indicator = " (IMPROVED)" if values["improvement"] else " (WORSE)"
                    f.write(f"🎯 {primary_metric} (PRIMARY): {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str}){improvement_indicator}\n")
                else:
                    # Higher is better metrics
                    direction = "↑" if values["improvement"] else "↓"
                    improvement_indicator = " (IMPROVED)" if values["improvement"] else " (WORSE)"
                    f.write(f"🎯 {primary_metric} (PRIMARY): {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str}){improvement_indicator}\n")
            
            # Show other metrics
            for metric_name, values in metrics.items():
                if metric_name == primary_metric:
                    continue  # Already shown above
                    
                base_val = values["base"]
                finetuned_val = values["finetuned"]
                abs_diff = abs(values["absolute_diff"])
                improvement_mag = values["improvement_magnitude_percent"]
                
                # Handle infinity and NaN values for display
                if base_val == float('inf') or base_val == float('-inf') or base_val != base_val:
                    base_str = str(base_val)
                else:
                    base_str = f"{base_val:.4f}"
                
                if finetuned_val == float('inf') or finetuned_val == float('-inf') or finetuned_val != finetuned_val:
                    finetuned_str = str(finetuned_val)
                else:
                    finetuned_str = f"{finetuned_val:.4f}"
                
                if abs_diff == float('inf') or abs_diff == float('-inf') or abs_diff != abs_diff:
                    diff_str = str(abs_diff)
                else:
                    diff_str = f"{abs_diff:.4f}"
                
                if improvement_mag == float('inf') or improvement_mag == float('-inf') or improvement_mag != improvement_mag:
                    mag_str = str(improvement_mag)
                else:
                    mag_str = f"{improvement_mag:.1f}%"
                    
                if metric_name in ["accuracy", "f1_score", "balanced_accuracy", "precision", "recall", "green_score"]:
                    # Higher is better metrics
                    direction = "↑" if values["improvement"] else "↓"
                    f.write(f"  {metric_name}: {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str})\n")
                elif metric_name in ["mean_absolute_error", "root_mean_squared_error"]:
                    # Lower is better metrics
                    direction = "↓" if values["absolute_diff"] < 0 else "↑"
                    f.write(f"  {metric_name}: {base_str} → {finetuned_str} ({direction} {diff_str}, {mag_str})\n")
        
        f.write("\nOverall Performance Summary\n")
        f.write("===========================\n")
        if "overall_summary" in comparison:
            summary = comparison["overall_summary"]
            if summary["total_tasks"] > 0:
                improvement_rate = summary["improvement_rate_percent"]
                f.write(f"🏆 Fine-tuning improved {summary['improved_tasks']}/{summary['total_tasks']} tasks ({improvement_rate:.1f}%)\n\n")
                
                f.write("Task-by-task improvement summary:\n")
                for task_name, details in summary["task_details"].items():
                    status = "✅ IMPROVED" if details["improved"] else "❌ WORSE"
                    magnitude = details["improvement_magnitude"]
                    primary = details["primary_metric"]
                    
                    # Handle infinity and NaN values for display
                    if magnitude == float('inf') or magnitude == float('-inf') or magnitude != magnitude:
                        magnitude_str = str(magnitude)
                    else:
                        magnitude_str = f"{magnitude:.1f}%"
                    
                    f.write(f"  {task_name}: {status} ({magnitude_str} change in {primary})\n")
    
    logger.info(f"Detailed comparison report saved to {report_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned QwenVL model on validation set (includes both hidden and public validation data)")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model path")
    parser.add_argument("--lora_weights", type=str, default="finetuned_qwenvl/final", help="Path to LoRA weights")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data_qwenvl", help="Directory with processed datasets")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/validation", help="Output directory for evaluation results")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Maximum number of samples to evaluate per task (0 = evaluate all)")
    parser.add_argument("--force_cpu", action="store_true", help="Force evaluation on CPU if no GPU is available")
    parser.add_argument("--save_predictions", action="store_true", help="Save individual predictions with details")
    parser.add_argument("--evaluate_base_model", action="store_true", default=True, help="Evaluate the base model for comparison (default: True)")
    parser.add_argument("--skip_base_model", action="store_true", help="Skip base model evaluation (only evaluate fine-tuned model)")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the base model")
    parser.add_argument("--eval_batch_size", type=int, default=0, help="Batch size for evaluation (0 = auto-determine based on GPU memory)")
    parser.add_argument("--task_type", type=str, default=None, 
                       choices=["classification", "multi-label", "detection", "instance_detection", 
                               "counting", "regression", "report_generation"],
                       help="Evaluate only a specific task type. If not specified, evaluates all tasks.")
    args = parser.parse_args()
    
    # Override evaluate_base_model if skip_base_model is specified
    if args.skip_base_model:
        args.evaluate_base_model = False
        logger.info("Base model evaluation skipped as requested")
    else:
        args.evaluate_base_model = True
        logger.info("Base model evaluation enabled for comparison on validation set")
    
    evaluate_model(args) 