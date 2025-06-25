#!/usr/bin/env python3
"""
FLARE 2025 Challenge Inference Script
====================================

This script runs inference on the FLARE 2025 validation-hidden dataset
and generates predictions.json file for challenge submission.

The script automatically:
- Finds all *_questions_val.json files in validation-hidden/ folder
- Excludes files ending with "withGT" 
- Processes all question-image pairs using the FLARE dataset format
- Outputs a single predictions.json file with all predicted answers

Usage:
    # Using fine-tuned model
    python inference.py --dataset_path organized_dataset --lora_weights finetuned_qwenvl/final
    
    # Using base model only
    python inference.py --dataset_path organized_dataset --model_name Qwen/Qwen2.5-VL-7B-Instruct
    
    # With custom settings
    python inference.py --dataset_path organized_dataset --lora_weights finetuned_qwenvl/final --max_tokens 128 --verbose
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalVLMInference:
    """Inference class for medical vision-language model."""
    
    def __init__(
        self, 
        model_name_or_path: str,
        device: str = "auto",
        quantize: bool = True,
        max_memory: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model_name_or_path: HuggingFace model name or local checkpoint path
            device: Device to use ('cuda', 'cpu', or 'auto')
            quantize: Whether to use 4-bit quantization
            max_memory: Maximum memory per device
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.quantize = quantize
        
        # Set up device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processors
        self._load_model(max_memory)
        
        # Task-specific configurations
        self.task_configs = self._get_task_configs()
    
    def _load_model(self, max_memory: Optional[Dict[str, str]] = None):
        """Load the model, tokenizer, and processor."""
        logger.info(f"Loading model from {self.model_name_or_path}")
        
        # Determine if using LoRA checkpoint (local or HuggingFace) or base model
        base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        is_local_lora = Path(self.model_name_or_path).exists() and (
            Path(self.model_name_or_path) / "adapter_config.json"
        ).exists()
        is_hf_lora = not is_local_lora and self.model_name_or_path != base_model_name
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.processor = AutoProcessor.from_pretrained(base_model_name)
        
        # Quantization config
        quantization_config = None
        if self.quantize and self.device != "cpu":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load base model
        logger.info("Loading base model...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                max_memory=max_memory
            )
        except ImportError:
            # Fallback to AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                max_memory=max_memory
            )
        
        # Load LoRA adapter if provided
        if is_local_lora:
            logger.info(f"Loading local LoRA adapter from {self.model_name_or_path}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                self.model_name_or_path,
                device_map=self.device if self.device != "cpu" else None
            )
        elif is_hf_lora:
            logger.info(f"Loading HuggingFace LoRA adapter from {self.model_name_or_path}")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    self.model_name_or_path,
                    device_map=self.device if self.device != "cpu" else None
                )
            except Exception as e:
                logger.error(f"Failed to load HuggingFace LoRA adapter: {e}")
                logger.warning("Falling back to base model only")
        elif self.model_name_or_path != base_model_name:
            logger.warning(f"Model path {self.model_name_or_path} does not appear to be a LoRA adapter. Using base model only.")
        
        # Set to evaluation mode
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def _get_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get task-specific configurations and prompts."""
        return {
            "classification": {
                "prompt_template": "Analyze this medical image and classify it. What is the primary diagnosis or finding?",
                "max_tokens": 128,
                "parse_fn": self._parse_classification
            },
            "multi_label": {
                "prompt_template": "List all abnormalities and findings present in this medical image. Be comprehensive.",
                "max_tokens": 256,
                "parse_fn": self._parse_multi_label
            },
            "detection": {
                "prompt_template": "Identify and locate all abnormalities in this medical image. Provide bounding box coordinates in format [x1,y1,x2,y2] for each finding.",
                "max_tokens": 512,
                "parse_fn": self._parse_detection
            },
            "counting": {
                "prompt_template": "Count the number of {target} in this medical image. Provide only the numerical count.",
                "max_tokens": 32,
                "parse_fn": self._parse_counting
            },
            "report_generation": {
                "prompt_template": "Generate a comprehensive medical report for this image. Include findings, impressions, and recommendations.",
                "max_tokens": 1024,
                "parse_fn": self._parse_report
            },
            "vqa": {
                "prompt_template": "{question}",
                "max_tokens": 256,
                "parse_fn": lambda x: x.strip()
            }
        }
    
    def inference(
        self,
        image: Union[str, Image.Image, List[Union[str, Image.Image]]],
        task: str = "report_generation",
        prompt: Optional[str] = None,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run inference on single or multiple images.
        
        Args:
            image: Image path, PIL Image, or list of images
            task: Task type (classification, detection, report_generation, etc.)
            prompt: Custom prompt (overrides task default)
            **kwargs: Additional generation parameters
        
        Returns:
            Inference results as dictionary or list of dictionaries
        """
        # Handle batch vs single inference
        if isinstance(image, list):
            return self.batch_inference(image, task, prompt, **kwargs)
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
            image_path = image
        else:
            pil_image = image.convert('RGB')
            image_path = "uploaded_image"
        
        # Get task configuration
        task_config = self.task_configs.get(task, self.task_configs["vqa"])
        
        # Prepare prompt
        if prompt is None:
            prompt = task_config["prompt_template"]
        
        # Handle special prompts
        if "{target}" in prompt and "target" in kwargs:
            prompt = prompt.format(target=kwargs["target"])
        elif "{question}" in prompt and "question" in kwargs:
            prompt = prompt.format(question=kwargs["question"])
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=pil_image,
            text=text,
            return_tensors="pt"
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", task_config["max_tokens"]),
                top_p=kwargs.get("top_p", 0.9),
                num_beams=kwargs.get("num_beams", 1),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse response based on task
        parsed_result = task_config["parse_fn"](response)
        
        # Return structured result
        return {
            "image_path": image_path,
            "task": task,
            "prompt": prompt,
            "raw_response": response,
            "parsed_result": parsed_result,
            "metadata": {
                "model": self.model_name_or_path,
                "max_tokens": kwargs.get("max_tokens", task_config["max_tokens"])
            }
        }
    
    def batch_inference(
        self,
        images: List[Union[str, Image.Image]],
        task: str = "report_generation",
        prompt: Optional[str] = None,
        batch_size: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple images in batches."""
        results = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Processing images"):
            batch = images[i:i + batch_size]
            
            # Process each image in batch
            for img in batch:
                try:
                    result = self.inference(img, task, prompt, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    results.append({
                        "image_path": str(img) if isinstance(img, str) else "image",
                        "error": str(e)
                    })
        
        return results
    
    # Parsing functions for different tasks
    def _parse_classification(self, response: str) -> Dict[str, Any]:
        """Parse classification response."""
        # Extract diagnosis/finding
        diagnosis = response.strip()
        
        # Try to extract confidence if mentioned
        confidence_match = re.search(r'(\d+)%|confidence:?\s*(\d*\.?\d+)', response.lower())
        confidence = None
        if confidence_match:
            confidence = float(confidence_match.group(1) or confidence_match.group(2))
            if confidence > 1:
                confidence /= 100
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence
        }
    
    def _parse_multi_label(self, response: str) -> Dict[str, Any]:
        """Parse multi-label classification response."""
        # Extract findings (usually listed)
        findings = []
        
        # Try different patterns
        lines = response.strip().split('\n')
        for line in lines:
            # Remove bullet points, numbers, etc.
            cleaned = re.sub(r'^[-•*\d.]+\s*', '', line.strip())
            if cleaned and len(cleaned) > 3:
                findings.append(cleaned)
        
        return {
            "findings": findings,
            "count": len(findings)
        }
    
    def _parse_detection(self, response: str) -> Dict[str, Any]:
        """Parse detection response with bounding boxes."""
        detections = []
        
        # Pattern for bounding boxes [x1,y1,x2,y2]
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        
        # Find all bounding boxes
        matches = re.finditer(bbox_pattern, response)
        
        for match in matches:
            x1, y1, x2, y2 = map(int, match.groups())
            
            # Try to find associated label
            text_before = response[:match.start()].strip()
            label_match = re.search(r'(\w+[\w\s]*)\s*:?\s*$', text_before)
            label = label_match.group(1) if label_match else "abnormality"
            
            detections.append({
                "label": label.strip(),
                "bbox": [x1, y1, x2, y2],
                "confidence": None  # Model doesn't provide confidence
            })
        
        return {
            "detections": detections,
            "count": len(detections)
        }
    
    def _parse_counting(self, response: str) -> Dict[str, Any]:
        """Parse counting response."""
        # Extract number
        numbers = re.findall(r'\d+', response)
        count = int(numbers[0]) if numbers else 0
        
        return {
            "count": count,
            "raw_response": response.strip()
        }
    
    def _parse_report(self, response: str) -> Dict[str, Any]:
        """Parse medical report generation response."""
        report_sections = {
            "findings": "",
            "impression": "",
            "recommendations": ""
        }
        
        # Try to extract sections
        current_section = "findings"
        lines = response.strip().split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in ["findings:", "finding:"]):
                current_section = "findings"
                continue
            elif any(keyword in line_lower for keyword in ["impression:", "impressions:", "conclusion:"]):
                current_section = "impression"
                continue
            elif any(keyword in line_lower for keyword in ["recommendation:", "recommendations:", "suggest:"]):
                current_section = "recommendations"
                continue
            
            if line.strip():
                report_sections[current_section] += line.strip() + " "
        
        # Clean up sections
        for key in report_sections:
            report_sections[key] = report_sections[key].strip()
        
        # If no sections found, put everything in findings
        if not any(report_sections.values()):
            report_sections["findings"] = response.strip()
        
        return {
            "report": report_sections,
            "full_text": response.strip()
        }

def main():
    parser = argparse.ArgumentParser(description="FLARE 2025 Challenge Inference")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model name on HuggingFace or local checkpoint path")
    parser.add_argument("--lora_weights", type=str,
                       help="Path to LoRA weights/adapter")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cuda", "cpu", "auto"],
                       help="Device to use for inference")
    parser.add_argument("--no_quantize", action="store_true",
                       help="Disable 4-bit quantization")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to organized_dataset folder")
    parser.add_argument("--output_file", type=str, default="predictions.json",
                       help="Output predictions file")
    
    # Generation arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    
    # Utility arguments
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("FLARE 2025 Challenge Inference Started")
    logger.info(f"Dataset path: {args.dataset_path}")
    
    # Initialize model
    logger.info("Initializing model...")
    if args.lora_weights:
        model = MedicalVLMInference(
            model_name_or_path=args.lora_weights,
            device=args.device,
            quantize=not args.no_quantize
        )
    else:
        model = MedicalVLMInference(
            model_name_or_path=args.model_name,
            device=args.device,
            quantize=not args.no_quantize
        )
    
    # Find all question files in the dataset path
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return
    
    # Find all question files (exclude withGT files)
    # Search recursively for any JSON files that look like question files
    question_files = []
    
    # Search patterns for question files
    search_patterns = [
        "*_questions_val.json",
        "*_questions.json", 
        "*questions*.json",
        "questions*.json"
    ]
    
    logger.info(f"Searching for question files in: {dataset_path}")
    
    # Search recursively through all subdirectories
    for pattern in search_patterns:
        for file_path in dataset_path.rglob(pattern):
            if "withGT" not in file_path.name and file_path.is_file():
                question_files.append(file_path)
                logger.info(f"Found question file: {file_path}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_question_files = []
    for file_path in question_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_question_files.append(file_path)
    
    question_files = unique_question_files
    
    if not question_files:
        logger.error("No validation question files found")
        return
    
    logger.info(f"Found {len(question_files)} question files to process")
    
    # Process all question files
    all_predictions = []
    
    for question_file in tqdm(question_files, desc="Processing datasets"):
        logger.info(f"Processing {question_file}")
        
        try:
            # Load questions
            with open(question_file, 'r') as f:
                questions = json.load(f)
            
            # Get dataset info
            dataset_name = question_file.parent.name
            
            # Try to find image directory - check multiple possible locations
            possible_image_dirs = [
                question_file.parent / "imagesVal",
                question_file.parent / "imagesTs",  # Test images
                question_file.parent / "images", 
                question_file.parent / "Images",
                question_file.parent / "img",
                question_file.parent,  # Images might be in same directory as questions
            ]
            
            image_dir = None
            for possible_dir in possible_image_dirs:
                if possible_dir.exists():
                    # Check if this directory contains image files
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
                    has_images = any(
                        f.suffix.lower() in image_extensions 
                        for f in possible_dir.iterdir() 
                        if f.is_file()
                    )
                    if has_images:
                        image_dir = possible_dir
                        logger.info(f"Using image directory: {image_dir}")
                        break
            
            if image_dir is None:
                logger.error(f"No image directory found for {question_file}")
                continue
            
            logger.info(f"Processing {len(questions)} questions from {dataset_name}")
            
            # Process each question
            for question_data in tqdm(questions, desc=f"{dataset_name}", leave=False):
                try:
                    # Get image path
                    image_name = question_data["ImageName"]
                    
                    # Clean up image name - remove common prefixes
                    for prefix in ["imagesVal/", "imagesTs/", "images/", "Images/", "img/"]:
                        if image_name.startswith(prefix):
                            image_name = image_name.replace(prefix, "")
                            break
                    
                    # Try different ways to find the image
                    possible_image_paths = [
                        image_dir / image_name,
                        image_dir / Path(image_name).name,  # Just filename without subdirs
                    ]
                    
                    # Also try with different subdirectories if image_name contains path separators
                    if "/" in image_name or "\\" in image_name:
                        base_name = Path(image_name).name
                        possible_image_paths.append(image_dir / base_name)
                    
                    image_path = None
                    for possible_path in possible_image_paths:
                        if possible_path.exists():
                            image_path = possible_path
                            break
                    
                    if image_path is None:
                        # If still not found, use the first possibility for error reporting
                        image_path = image_dir / image_name
                    
                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}")
                        # Add empty prediction
                        prediction = question_data.copy()
                        prediction["Answer"] = ""
                        all_predictions.append(prediction)
                        continue
                    
                    # Map TaskType to internal task format
                    task_mapping = {
                        'classification': 'classification',
                        'Classification': 'classification',
                        'Multi-label Classification': 'multi_label',
                        'Detection': 'detection',
                        'Instance Detection': 'detection',
                        'Cell Counting': 'counting',
                        'Regression': 'vqa',
                        'Report Generation': 'report_generation'
                    }
                    
                    task_type = question_data.get("TaskType", "classification")
                    internal_task = task_mapping.get(task_type, "classification")
                    
                    # Run inference
                    result = model.inference(
                        image=str(image_path),
                        task=internal_task,
                        prompt=question_data["Question"],
                        max_tokens=args.max_tokens
                    )
                    
                    # Extract answer based on task type
                    if internal_task == "classification":
                        # For classification, try to extract the letter option
                        raw_answer = result["parsed_result"].get("diagnosis", result["raw_response"])
                        
                        # Try to find option letter (A, B, C, etc.) in the response
                        import re
                        option_match = re.search(r'\b([A-K])\b', raw_answer)
                        if option_match:
                            answer = option_match.group(1)
                        else:
                            # If no letter found, use the raw answer
                            answer = raw_answer.strip()
                    elif internal_task == "multi_label":
                        findings = result["parsed_result"].get("findings", [])
                        answer = "; ".join(findings) if findings else result["raw_response"].strip()
                    elif internal_task == "detection":
                        detections = result["parsed_result"].get("detections", [])
                        if detections:
                            # Format as coordinate list
                            bbox_strings = []
                            for det in detections:
                                bbox = det.get("bbox", [])
                                if len(bbox) == 4:
                                    bbox_strings.append(f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
                            answer = "; ".join(bbox_strings) if bbox_strings else result["raw_response"].strip()
                        else:
                            answer = result["raw_response"].strip()
                    elif internal_task == "counting":
                        count = result["parsed_result"].get("count", 0)
                        answer = str(count)
                    else:
                        answer = result["raw_response"].strip()
                    
                    # Create prediction entry
                    prediction = question_data.copy()
                    prediction["Answer"] = answer
                    all_predictions.append(prediction)
                    
                    if args.verbose:
                        logger.info(f"Q: {question_data['Question'][:100]}...")
                        logger.info(f"A: {answer}")
                
                except Exception as e:
                    logger.error(f"Error processing question: {e}")
                    # Add empty prediction for failed cases
                    prediction = question_data.copy()
                    prediction["Answer"] = ""
                    all_predictions.append(prediction)
        
        except Exception as e:
            logger.error(f"Error processing question file {question_file}: {e}")
    
    # Save all predictions
    logger.info(f"Saving {len(all_predictions)} predictions to {args.output_file}")
    
    with open(args.output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    logger.info("FLARE 2025 Challenge Inference Completed!")
    
    # Print summary
    task_summary = {}
    for pred in all_predictions:
        task_type = pred.get("TaskType", "Unknown")
        task_summary[task_type] = task_summary.get(task_type, 0) + 1
    
    logger.info("Summary by task type:")
    for task_type, count in task_summary.items():
        logger.info(f"  {task_type}: {count} predictions")

if __name__ == "__main__":
    main() 