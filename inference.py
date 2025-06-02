#!/usr/bin/env python3
"""
Inference Script for Fine-tuned QWen2.5VL Medical Model
======================================================

This script demonstrates how to use the fine-tuned QWen2.5VL model for various
medical imaging tasks including classification, detection, and report generation.

Usage:
    # Basic inference
    python inference.py --image_path xray.jpg --task classification --prompt "What abnormalities are present?"
    
    # Batch inference
    python inference.py --image_folder ./images --task report_generation --output results.json
    
    # Using HuggingFace model
    python inference.py --model_name leoyinn/qwen2.5vl-flare2025 --image_path mri.jpg

    # Using question file
    python inference.py --output results.json --question_file ./questions.json --verbose
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
        
        # Determine if using local checkpoint or HF model
        is_local = Path(self.model_name_or_path).exists()
        
        if is_local:
            # Load from local checkpoint
            base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            adapter_path = self.model_name_or_path
        else:
            # Load from HuggingFace
            base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            adapter_path = self.model_name_or_path
        
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
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map=self.device if self.device != "cpu" else None,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            max_memory=max_memory
        )
        
        # Load adapter
        logger.info("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(
            self.model, 
            adapter_path,
            device_map=self.device if self.device != "cpu" else None
        )
        
        # Set to evaluation mode
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def _get_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get task-specific configurations and prompts."""
        return {
            "classification": {
                "prompt_template": "Analyze this medical image and classify it. What is the primary diagnosis or finding?",
                "max_tokens": 128,
                "temperature": 0.1,
                "parse_fn": self._parse_classification
            },
            "multi_label": {
                "prompt_template": "List all abnormalities and findings present in this medical image. Be comprehensive.",
                "max_tokens": 256,
                "temperature": 0.2,
                "parse_fn": self._parse_multi_label
            },
            "detection": {
                "prompt_template": "Identify and locate all abnormalities in this medical image. Provide bounding box coordinates in format [x1,y1,x2,y2] for each finding.",
                "max_tokens": 512,
                "temperature": 0.1,
                "parse_fn": self._parse_detection
            },
            "counting": {
                "prompt_template": "Count the number of {target} in this medical image. Provide only the numerical count.",
                "max_tokens": 32,
                "temperature": 0.0,
                "parse_fn": self._parse_counting
            },
            "report_generation": {
                "prompt_template": "Generate a comprehensive medical report for this image. Include findings, impressions, and recommendations.",
                "max_tokens": 1024,
                "temperature": 0.3,
                "parse_fn": self._parse_report
            },
            "vqa": {
                "prompt_template": "{question}",
                "max_tokens": 256,
                "temperature": 0.2,
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
                temperature=kwargs.get("temperature", task_config["temperature"]),
                do_sample=kwargs.get("temperature", task_config["temperature"]) > 0,
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
                "temperature": kwargs.get("temperature", task_config["temperature"]),
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

def create_example_prompts() -> Dict[str, List[Dict[str, str]]]:
    """Create example prompts for different medical imaging scenarios."""
    return {
        "chest_xray": [
            {
                "task": "classification",
                "prompt": "Classify this chest X-ray. Is it normal or abnormal? If abnormal, what is the primary finding?"
            },
            {
                "task": "multi_label",
                "prompt": "List all abnormalities visible in this chest X-ray including pneumonia, effusion, cardiomegaly, or other findings."
            },
            {
                "task": "report_generation",
                "prompt": "Generate a radiology report for this chest X-ray including findings and impression."
            }
        ],
        "fundus": [
            {
                "task": "classification",
                "prompt": "Analyze this fundus image for diabetic retinopathy. What grade would you assign?"
            },
            {
                "task": "detection",
                "prompt": "Identify and locate any microaneurysms, hemorrhages, or exudates in this fundus image."
            }
        ],
        "pathology": [
            {
                "task": "classification",
                "prompt": "Classify this histopathology image. Is it benign or malignant?"
            },
            {
                "task": "counting",
                "prompt": "Count the number of mitotic figures in this pathology image."
            }
        ],
        "ultrasound": [
            {
                "task": "classification",
                "prompt": "Analyze this ultrasound image. What organ is being examined and are there any abnormalities?"
            },
            {
                "task": "vqa",
                "prompt": "Is there evidence of gallstones in this ultrasound image?"
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Medical VLM Inference")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model name on HuggingFace or local checkpoint path")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cuda", "cpu", "auto"],
                       help="Device to use for inference")
    parser.add_argument("--no_quantize", action="store_true",
                       help="Disable 4-bit quantization")
    
    # Input arguments
    parser.add_argument("--image_path", type=str,
                       help="Path to single image")
    parser.add_argument("--image_folder", type=str,
                       help="Path to folder containing images")
    parser.add_argument("--file_pattern", type=str, default="*.{jpg,jpeg,png,dcm}",
                       help="File pattern for batch processing")
    parser.add_argument("--questions_file", type=str,
                       help="JSON file with image-question pairs for batch processing")
    
    # Task arguments
    parser.add_argument("--task", type=str, default="report_generation",
                       choices=["classification", "multi_label", "detection", 
                               "counting", "report_generation", "vqa"],
                       help="Task to perform")
    parser.add_argument("--prompt", type=str,
                       help="Custom prompt (overrides task default)")
    parser.add_argument("--question", type=str,
                       help="Question for VQA task")
    parser.add_argument("--target", type=str,
                       help="Target object for counting task")
    
    # Generation arguments
    parser.add_argument("--max_tokens", type=int,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float,
                       help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing multiple images")
    
    # Output arguments
    parser.add_argument("--output", type=str,
                       help="Output file path (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize model
    logger.info("Initializing model...")
    model = MedicalVLMInference(
        model_name_or_path=args.model_name,
        device=args.device,
        quantize=not args.no_quantize
    )
    
    # Prepare images and questions
    images = []
    questions_data = []
    
    if args.questions_file:
        # Load questions from file
        logger.info(f"Loading questions from {args.questions_file}")
        with open(args.questions_file, 'r') as f:
            questions_data = json.load(f)
        
        # Extract images from questions data (support multiple formats)
        images = []
        for item in questions_data:
            # Support different field names for image path
            image_path = (item.get('image_path') or 
                         item.get('image') or 
                         item.get('ImageName') or
                         item.get('imageName'))
            images.append(image_path)
        logger.info(f"Found {len(images)} image-question pairs")
        
    elif args.image_path:
        images = [args.image_path]
    elif args.image_folder:
        folder = Path(args.image_folder)
        patterns = args.file_pattern.strip("{}").split(",")
        for pattern in patterns:
            images.extend(folder.glob(f"**/{pattern}"))
        images = [str(img) for img in images]
        logger.info(f"Found {len(images)} images")
    else:
        logger.error("Please provide --image_path, --image_folder, or --questions_file")
        return
    
    # Run inference
    logger.info(f"Running {args.task} inference on {len(images)} image(s)...")
    
    # Prepare kwargs
    inference_kwargs = {}
    if args.max_tokens:
        inference_kwargs["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        inference_kwargs["temperature"] = args.temperature
    if args.question:
        inference_kwargs["question"] = args.question
    if args.target:
        inference_kwargs["target"] = args.target
    
    # Process images
    results = []
    
    if args.questions_file:
        # Process with individual questions from file
        logger.info(f"Processing {len(questions_data)} image-question pairs...")
        for item in tqdm(questions_data, desc="Processing"):
            try:
                # Support multiple field names for image path
                image_path = (item.get('image_path') or 
                             item.get('image') or 
                             item.get('ImageName') or
                             item.get('imageName'))
                
                # Support multiple field names for question
                question = (item.get('question') or 
                           item.get('prompt') or 
                           item.get('Question'))
                
                # Map TaskType to task (with fallback)
                task_mapping = {
                    'Detection': 'detection',
                    'Classification': 'classification', 
                    'Multi-label Classification': 'multi_label',
                    'Instance Detection': 'detection',
                    'Cell Counting': 'counting',
                    'Regression': 'vqa',
                    'Report Generation': 'report_generation'
                }
                
                task_type = item.get('TaskType', item.get('task', args.task))
                task = task_mapping.get(task_type, task_type.lower() if isinstance(task_type, str) else args.task)
                
                # Override inference_kwargs with item-specific values
                item_kwargs = inference_kwargs.copy()
                if 'max_tokens' in item:
                    item_kwargs['max_tokens'] = item['max_tokens']
                if 'temperature' in item:
                    item_kwargs['temperature'] = item['temperature']
                if 'target' in item:
                    item_kwargs['target'] = item['target']
                
                result = model.inference(
                    image_path,
                    task=task,
                    prompt=question,
                    **item_kwargs
                )
                
                # Add original question data to result
                result['original_item'] = item
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                image_path = (item.get('image_path') or 
                             item.get('image') or 
                             item.get('ImageName') or
                             item.get('imageName') or 'unknown')
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "original_item": item
                })
    else:
        # Standard processing
        if len(images) == 1:
            result = model.inference(
                images[0],
                task=args.task,
                prompt=args.prompt,
                **inference_kwargs
            )
            results = [result]
        else:
            results = model.batch_inference(
                images,
                task=args.task,
                prompt=args.prompt,
                batch_size=args.batch_size,
                **inference_kwargs
            )
    
    # Display results
    for result in results:
        if "error" in result:
            logger.error(f"Error processing {result['image_path']}: {result['error']}")
        else:
            logger.info(f"\nImage: {result['image_path']}")
            logger.info(f"Task: {result['task']}")
            logger.info(f"Result: {json.dumps(result['parsed_result'], indent=2)}")
            if args.verbose:
                logger.info(f"Raw response: {result['raw_response']}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 