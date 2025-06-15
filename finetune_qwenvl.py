import os
import json
import argparse
import time
import io
import gc
from collections import OrderedDict
import signal, psutil, sys
import warnings

# Suppress the specific FutureWarning from torch.utils.checkpoint, which is not actionable since it is a warning from the library
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*`torch.cpu.amp.autocast\(args\.\.\.\)` is deprecated.*")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable CUDA debugging and stability settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
import logging
from datasets import load_from_disk, Dataset
from PIL import Image
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader, IterableDataset
import threading
from queue import Queue
import cv2  # For faster image loading

try:
    from PIL.Image import Resampling
    LANCZOS_RESAMPLING = Resampling.LANCZOS
except (ImportError, AttributeError):
    LANCZOS_RESAMPLING = Image.LANCZOS

# Enable Flash Attention 2 for memory efficiency and speed
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

os.makedirs("logs", exist_ok=True)

# Remove all existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a FileHandler for all logs (INFO and above)
log_filename = os.path.join("logs", f"finetune_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Create a StreamHandler for only errors (ERROR and above)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_filename}")

# Avoid fork-related tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress gradient checkpointing warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# Add warning filters
def worker_init_fn(worker_id):
    """Worker initialization for multiprocessing with PEFT models"""
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class FastImageLoader:
    """Fast image loader using cv2 with caching (multiprocessing-safe)"""
    def __init__(self, cache_size_mb=2048):
        self.cache = OrderedDict()
        self.cache_size_mb = cache_size_mb
        self.current_size_mb = 0
        self.hits = 0
        self.misses = 0
        
    def load_image(self, image_path):
        """Load image with caching (multiprocessing-safe)"""
        if image_path in self.cache:
            self.hits += 1
            self.cache.move_to_end(image_path)
            return self.cache[image_path].copy()
        
        self.misses += 1
        
        # Load image
        try:
            # Use cv2 for faster loading; fall back to PIL if cv2 fails
            img = cv2.imread(image_path)
            if img is None:
                # cv2 could not read – open directly with PIL
                pil_img = Image.open(image_path).convert("RGB")
            else:
                # cv2 succeeds – convert to RGB first, then wrap in PIL
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
            # Always store PIL.Image in cache
            # Estimate size in MB (RGB only)
            size_mb = (pil_img.width * pil_img.height * 3) / (1024 * 1024)
            # Evict oldest entries if cache is full or would exceed size
            while self.cache and (len(self.cache) >= 5000 or self.current_size_mb + size_mb > self.cache_size_mb):
                oldest_path, oldest_img = self.cache.popitem(last=False)
                # Estimate size of evicted image
                if isinstance(oldest_img, Image.Image):
                    old_size_mb = (oldest_img.width * oldest_img.height * 3) / (1024 * 1024)
                else:
                    old_size_mb = 0
                self.current_size_mb -= old_size_mb
            # Add the new image if there's now space (or if it was already small enough)
            if self.current_size_mb + size_mb <= self.cache_size_mb:
                self.cache[image_path] = pil_img
                self.current_size_mb += size_mb
            # else: image is too large for current cache state, don't cache but still return
            return pil_img
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "cache_size": len(self.cache),
            "size_mb": self.current_size_mb
        }

class MultimodalDataCollator:
    """Data collator with proper Qwen2.5-VL image/token handling"""
    def __init__(self, tokenizer, processor, model=None, max_seq_length=2048, 
                 image_cache_size_mb=2048, image_size=448, max_images_per_sample=0):
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.image_loader = FastImageLoader(cache_size_mb=image_cache_size_mb)
        
        # Processing cache for tokenized results
        self.processing_cache = OrderedDict()
        self.max_cache_entries = 5000
        
        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Ensure consistent processor configuration
        self._configure_processor()
        
        # Performance monitoring
        self.timing_stats = {
            "image_loading": [],
            "processing": [],
        }
        self._last_stat_log = time.time()
        
        # Error tracking
        self.error_count = 0
        self.total_batches = 0
        
        self.max_images_per_sample = max_images_per_sample
    
    def _configure_processor(self):
        """Properly configure processor for consistent tokenization"""
        if hasattr(self.processor, 'image_processor'):
            # Set consistent image size
            try:
                # Prefer explicit dict
                self.processor.image_processor.size = {"height": self.image_size, "width": self.image_size}
                logger.info("Set processor size to dict format")
                
                # Set pixel limits safely
                if hasattr(self.processor.image_processor, 'max_pixels'):
                    self.processor.image_processor.max_pixels = self.image_size * self.image_size
                if hasattr(self.processor.image_processor, 'min_pixels'):
                    self.processor.image_processor.min_pixels = self.image_size * self.image_size
                
                # Set processing flags safely
                if hasattr(self.processor.image_processor, 'do_resize'):
                    self.processor.image_processor.do_resize = True
                if hasattr(self.processor.image_processor, 'do_center_crop'):
                    self.processor.image_processor.do_center_crop = False
                    
            except Exception as config_error:
                logger.warning(f"Could not fully configure processor: {config_error}")
                logger.info("Will rely on manual image preprocessing")
            
            logger.info(f"Processor configured with {self.image_size}x{self.image_size} resolution for stable tokenization")
    
    def _validate_inputs(self, inputs):
        """Validate inputs to catch token/feature mismatches early"""
        try:
            # Check if required keys exist
            if "input_ids" not in inputs:
                raise ValueError("Missing input_ids in processor output")
            
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"]
                input_ids = inputs["input_ids"]
                
                # Basic shape validation
                batch_size = input_ids.shape[0]
                if len(pixel_values.shape) >= 4:
                    # pixel_values shape: [batch_size, channels, height, width] or similar
                    pixel_batch_size = pixel_values.shape[0]
                    if pixel_batch_size != batch_size:
                        logger.warning(f"Batch size mismatch: input_ids={batch_size}, pixel_values={pixel_batch_size}")
                
                # Log token statistics for debugging
                for i in range(batch_size):
                    ids = input_ids[i]
                    # Count image tokens (specific to Qwen2.5-VL tokenizer)
                    image_token_count = 0
                    if hasattr(self.tokenizer, 'im_start_id'):
                        image_token_count = (ids == self.tokenizer.im_start_id).sum().item()
                    elif hasattr(self.tokenizer, 'image_token_id'):
                        image_token_count = (ids == self.tokenizer.image_token_id).sum().item()
                    
                    logger.debug(f"Sample {i}: sequence_length={len(ids)}, estimated_image_tokens={image_token_count}")
            
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def __call__(self, examples):
        # Start timing
        total_start = time.time()
        self.total_batches += 1
        
        batch_size = len(examples)
        batch_images = []
        batch_messages = []
        valid_indices = []
        
        # Process examples - extract images and messages
        img_start = time.time()
        for idx, example in enumerate(examples):
            try:
                # Get image paths - handle both old and new formats
                image_paths = []
                if "images" in example:
                    # Handle list of image paths (multi-image support)
                    images_field = example["images"]
                    if isinstance(images_field, list):
                        image_paths = images_field
                    else:
                        image_paths = [images_field]  # Single image as string
                elif "image" in example:
                    # Legacy single image format
                    image_paths = [example["image"]]
                else:
                    logger.error(f"No image field found in example {idx}")
                    continue
                
                # Load all images for this example
                example_images = []
                for img_path in image_paths:
                    try:
                        # Load and validate image
                        image = self.image_loader.load_image(img_path)
                        
                        # Ensure consistent image size before processing
                        target_size = (self.image_size, self.image_size)
                        
                        # Safely get image size
                        if isinstance(image, np.ndarray):
                            if image.ndim >= 2:
                                current_size = (image.shape[1], image.shape[0])  # (width, height)
                            else:
                                logger.error(f"Invalid numpy image shape: {image.shape}")
                                raise ValueError(f"Invalid numpy image shape: {image.shape}")
                        elif isinstance(image, Image.Image):
                            current_size = image.size
                        else:
                            logger.error(f"Unsupported image object type: {type(image)}")
                            raise ValueError(f"Unsupported image object type: {type(image)}")
                        
                        # Resize if needed
                        if current_size != target_size:
                            logger.debug(f"Resizing image from {current_size} to {target_size}")
                            if isinstance(image, Image.Image):
                                try:
                                    from PIL import Image as PILImage
                                    resample_const = getattr(PILImage, "Resampling", PILImage).LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS
                                    image = image.resize(target_size, resample_const)
                                except Exception as resize_err:
                                    logger.warning(f"PIL resize failed ({resize_err}); retrying with default resample")
                                    image = image.resize(target_size)
                            else:
                                # Convert numpy array to PIL Image and resize
                                from PIL import Image as PILImage
                                if image.ndim == 2:
                                    image = PILImage.fromarray(image, mode='L').convert('RGB')
                                else:
                                    image = PILImage.fromarray(image)
                                try:
                                    resample_const = getattr(PILImage, "Resampling", PILImage).LANCZOS if hasattr(PILImage, "Resampling") else PILImage.LANCZOS
                                    image = image.resize(target_size, resample_const)
                                except Exception as resize_err:
                                    logger.warning(f"PIL resize (numpy convert) failed ({resize_err}); resizing with default")
                                    image = image.resize(target_size)
                        
                        # Final validation
                        if not isinstance(image, Image.Image):
                            logger.error(f"Processed image is not a PIL Image (type={type(image)}) -> skipping image")
                            continue
                        
                        example_images.append(image)
                    except Exception as img_error:
                        logger.warning(f"Failed to load image {img_path}: {img_error}")
                        continue
                
                # Skip example if no valid images
                if not example_images:
                    logger.warning(f"No valid images found for example {idx}, skipping")
                    continue
                
                # Limit images per sample if configured
                if self.max_images_per_sample > 0 and len(example_images) > self.max_images_per_sample:
                    example_images = example_images[:self.max_images_per_sample]
                    logger.debug(f"Limited example {idx} to {self.max_images_per_sample} images")
                
                # Support multi-image examples
                batch_images.append(example_images)  # Keep all images, not just first
                
                # Get messages - handle both old and new formats
                if "messages" in example:
                    messages = example["messages"]
                else:
                    # Build on-the-fly from raw fields
                    messages = create_prompt_with_answer({
                        "question": example["question"],
                        "answer": example["answer"],
                        "task_type": example["task_type"],
                        "num_images": len(example_images)  # Pass number of images
                    })["messages"]
                
                batch_messages.append(messages)
                valid_indices.append(idx)
                
            except Exception as e:
                logger.error(f"Failed to process example {idx}: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                self.error_count += 1
                continue
        
        img_time = time.time() - img_start
        self.timing_stats["image_loading"].append(img_time)
        
        if not batch_images:
            raise ValueError("All images in batch failed to load")
        
        # Process messages with proper label masking
        proc_start = time.time()
        try:
            # Apply chat template to each message set
            batch_texts = []
            batch_assistant_starts = []  # Track where assistant response starts for label masking
            batch_processed_images = []  # Store processed images per example
            
            for idx, (messages, images_list) in enumerate(zip(batch_messages, batch_images)):
                try:
                    # For multi-image support, we need to properly format the messages
                    # with the actual image placeholders
                    if isinstance(images_list, list) and len(images_list) > 1:
                        # Multi-image case - ensure message content has correct number of image entries
                        user_message = messages[0]
                        if "content" in user_message:
                            content = user_message["content"]
                            # Count existing image entries
                            image_entries = [item for item in content if item.get("type") == "image"]
                            
                            # If mismatch, rebuild content with correct number of images
                            if len(image_entries) != len(images_list):
                                new_content = []
                                # Add all images
                                for _ in images_list:
                                    new_content.append({"type": "image"})
                                # Add text content
                                for item in content:
                                    if item.get("type") == "text":
                                        new_content.append(item)
                                messages[0]["content"] = new_content
                    
                    # Apply chat template with proper formatting
                    full_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False  # We want the full conversation
                    )
                    
                    # Find where the assistant response starts for label masking
                    # Look for assistant marker in the formatted text
                    assistant_markers = ["<|im_start|>assistant\n", "<|im_start|>assistant", "assistant\n", "assistant:"]
                    assistant_start_pos = -1
                    
                    for marker in assistant_markers:
                        pos = full_text.find(marker)
                        if pos != -1:
                            assistant_start_pos = pos + len(marker)
                            break
                    
                    if assistant_start_pos == -1:
                        logger.warning("Could not find assistant start position - will mask entire sequence")
                        assistant_start_pos = len(full_text)  # Mask everything
                    

                    batch_texts.append(full_text)
                    batch_assistant_starts.append(assistant_start_pos)
                    batch_processed_images.append(images_list)
                    
                except Exception as template_error:
                    logger.error(f"Chat template error: {template_error}")
                    # Fallback format
                    user_content = messages[0]["content"]
                    text_content = ""
                    num_images = 0
                    for content in user_content:
                        if content["type"] == "text":
                            text_content = content["text"]
                        elif content["type"] == "image":
                            num_images += 1
                    
                    assistant_content = messages[1]["content"] if len(messages) > 1 else ""
                    
                    # Build image placeholders for multi-image support
                    image_placeholders = ""
                    for i in range(max(1, num_images)):  # At least one image placeholder
                        image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>"
                    
                    full_text = f"<|im_start|>user\n{image_placeholders}{text_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
                    assistant_start_pos = full_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
                    
                    batch_texts.append(full_text)
                    batch_assistant_starts.append(assistant_start_pos)
                    batch_processed_images.append(images_list)
            
            # Flatten images for processor (it expects a flat list)
            # batch_images is now a list of lists, we need to flatten it
            flattened_images = []
            image_counts = []  # Track how many images per example
            for images_list in batch_images:
                if isinstance(images_list, list):
                    flattened_images.extend(images_list)
                    image_counts.append(len(images_list))
                else:
                    # Single image case (backward compatibility)
                    flattened_images.append(images_list)
                    image_counts.append(1)
            
            # Process with the tokenizer and processor
            inputs = self.processor(
                text=batch_texts,
                images=flattened_images if flattened_images else None,
                return_tensors="pt",
                padding=True,
                truncation=False,  # Disabled to avoid processor bug with multi-image
                # max_length=self.max_seq_length, # Let SFTTrainer handle truncation
            )
            
            # Create proper labels that only learn from assistant responses
            labels = inputs["input_ids"].clone()
            
            # Mask everything except assistant responses
            for i, (text, assistant_start_pos) in enumerate(zip(batch_texts, batch_assistant_starts)):
                # Tokenize just the prefix (everything before assistant response)
                prefix_text = text[:assistant_start_pos]
                # --- Efficient prefix length retrieval with LRU cache ---
                if prefix_text in self.processing_cache:
                    prefix_length = self.processing_cache[prefix_text]
                    # Move to end to mark as recently used
                    self.processing_cache.move_to_end(prefix_text)
                else:
                    prefix_tokens = self.tokenizer(
                        prefix_text,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"]
                    prefix_length = prefix_tokens.shape[1]
                    # Maintain LRU cache
                    self.processing_cache[prefix_text] = prefix_length
                    if len(self.processing_cache) > self.max_cache_entries:
                        self.processing_cache.popitem(last=False)
                
                # Mask the prefix tokens with -100 (ignore in loss)
                if prefix_length < labels.shape[1]:
                    labels[i, :prefix_length] = -100
                else:
                    # If prefix is longer than sequence, mask everything
                    labels[i, :] = -100
                    logger.warning(f"Prefix longer than sequence for sample {i}, masking entire sequence")
            
            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Add labels to inputs
            inputs["labels"] = labels
            
            # Validate
            if not self._validate_inputs(inputs):
                logger.error("Input validation failed after processing")
                raise ValueError("Input validation failed")
            
            proc_time = time.time() - proc_start
            self.timing_stats["processing"].append(proc_time)
            
            # Log sample for debugging (first sample only)
            if self.total_batches % 100 == 1:  # Every 100 batches
                sample_text = batch_texts[0]
                sample_labels = labels[0]
                sample_input_ids = inputs["input_ids"][0]
                sample_images = batch_processed_images[0] if batch_processed_images else []
                
                # Count masked vs unmasked tokens
                total_tokens = len(sample_labels)
                masked_tokens = (sample_labels == -100).sum().item()
                learning_tokens = total_tokens - masked_tokens
                
                # Count images in this sample
                num_images = len(sample_images) if isinstance(sample_images, list) else 1
                
                logger.info(f"TRAINING SAMPLE DEBUG (batch {self.total_batches}):")
                logger.info(f"  Number of images: {num_images}")
                logger.info(f"  Total tokens: {total_tokens}")
                logger.info(f"  Masked tokens (ignored): {masked_tokens}")
                logger.info(f"  Learning tokens (assistant): {learning_tokens}")
                logger.info(f"  Learning ratio: {learning_tokens/total_tokens:.2%}")
                logger.info(f"  Text preview: {sample_text[:200]}...")
                
                # Log multi-image statistics for the batch
                if any(len(imgs) > 1 if isinstance(imgs, list) else False for imgs in batch_processed_images):
                    multi_image_counts = [len(imgs) if isinstance(imgs, list) else 1 for imgs in batch_processed_images]
                    logger.info(f"  Multi-image samples in batch: {sum(1 for c in multi_image_counts if c > 1)}")
                    logger.info(f"  Image counts per sample: {multi_image_counts}")
                
                if learning_tokens == 0:
                    logger.error("WARNING: No tokens to learn from! All tokens masked!")
            
            # Log performance stats periodically
            current_time = time.time()
            if current_time - self._last_stat_log > 30:  # Every 30 seconds
                self._log_performance_stats()
                self._last_stat_log = current_time
            
            return inputs
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory - emergency cleanup")
            torch.cuda.empty_cache()
            gc.collect()
            raise
        except Exception as e:
            logger.error(f"Data collator error: {e}")
            logger.error(f"Error details - batch_size: {len(batch_texts)}, image_count: {len(batch_images)}")
            
            # Emergency cleanup
            torch.cuda.empty_cache()
            gc.collect()
            self.error_count += 1
            raise
    
    def _log_performance_stats(self):
        """Log performance statistics with error tracking"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        logger.info("=== Data Pipeline Performance Stats ===")
        logger.info(f"Total batches processed: {self.total_batches}")
        logger.info(f"Errors encountered: {self.error_count}")
        logger.info(f"Error rate: {(self.error_count/self.total_batches*100):.2f}%" if self.total_batches > 0 else "0%")
        logger.info(f"Image loading: avg={avg(self.timing_stats['image_loading']):.3f}s")
        logger.info(f"Processing: avg={avg(self.timing_stats['processing']):.3f}s")
        
        # Image cache stats
        cache_stats = self.image_loader.get_stats()
        logger.info(f"Image cache: hit_rate={cache_stats['hit_rate']:.1f}%, "
                   f"size={cache_stats['cache_size']} images, "
                   f"{cache_stats['size_mb']:.1f} MB")
        
        # Clear old stats to prevent memory growth
        for key in self.timing_stats:
            self.timing_stats[key] = self.timing_stats[key][-100:]  # Keep last 100

class CustomSFTTrainer(SFTTrainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,  # [2] Disable pin_memory for stability
            # "worker_init_fn": worker_init_fn,  # Not needed with num_workers=0
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
             dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
        return DataLoader(train_dataset, **dataloader_params)

def create_prompt_with_answer(row):
    """Create proper chat format for training - only learn to predict answers, not repeat questions"""
    # Better task type normalization to handle all case and separator variations
    # Guard against missing/NaN task_type
    raw_task_type = row.get("task_type")
    if raw_task_type is None or (hasattr(raw_task_type, "__class__") and raw_task_type.__class__.__name__ == 'float' and str(raw_task_type) == 'nan'):
        task_type = "classification"
    else:
        task_type = str(raw_task_type).lower().strip().replace("_", " ").replace("-", " ")
    question = row['question'].strip()
    answer = row['answer'].strip()
    num_images = row.get('num_images', 1)  # Get number of images, default to 1
    
    # Create task-specific instructions for better performance
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
        instruction = "Examine the medical image thoroughly and generate a comprehensive report describing your findings."
    else:
        instruction = "Look at the image and answer the question accurately based on what you observe."
        logger.warning(f"Unknown task type '{row['task_type']}' (normalized: '{task_type}') - using default instruction")
    
    # Adjust instruction for multiple images
    if num_images > 1:
        instruction = instruction.replace("the image", "the images").replace("this image", "these images")
    
    # Create proper chat format for training
    # The tokenizer.apply_chat_template will handle the proper format
    full_question = f"{instruction}\n\n{question}"
    
    # Build content list with multiple images if needed
    content = []
    for i in range(num_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": full_question})
    
    return {
        "messages": [
            {
                "role": "user",
                "content": content
            },
            {
                "role": "assistant", 
                "content": answer
            }
        ]
    }

def load_processed_datasets(processed_data_dir):
    """Load processed datasets without modifying columns (messages built on-the-fly)"""
    logger.info(f"Loading datasets from {processed_data_dir}")
    train_dataset = load_from_disk(os.path.join(processed_data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(processed_data_dir, "validation"))
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    return train_dataset, val_dataset

def get_available_gpu_memory():
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
        cached_memory = torch.cuda.memory_reserved(device) / (1024**3)
        available_memory = total_memory - allocated_memory
        
        logger.info(f"GPU Memory Status:")
        logger.info(f"  Total: {total_memory:.2f} GB")
        logger.info(f"  Allocated: {allocated_memory:.2f} GB")
        logger.info(f"  Cached: {cached_memory:.2f} GB")
        logger.info(f"  Available: {available_memory:.2f} GB")
        
        return available_memory, total_memory
    return 0, 0

def cleanup_memory():
    """Aggressive memory cleanup"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("Memory cleanup completed")



LOCK_PATH = "/tmp/qwenvl_train.lock"

def ensure_single_instance():
    if os.path.exists(LOCK_PATH):
        with open(LOCK_PATH) as f:
            pid = int(f.read().strip())
        if psutil.pid_exists(pid):
            logger.error(f"Another training run (PID {pid}) is already active.  Aborting.")
            sys.exit(1)
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    def _cleanup(*_):
        try:
            os.remove(LOCK_PATH)
        finally:
            sys.exit(0)
    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

def main(args):
    try:
        ensure_single_instance()
        # Log GPU device being used
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            logger.info(f"Using GPU device {current_device}: {torch.cuda.get_device_name(current_device)}")
            
            # Check available memory
            available_memory, total_memory = get_available_gpu_memory()
            
            if available_memory < 4:
                logger.error(f"Insufficient GPU memory available: {available_memory:.2f} GB")
                return
            
            cleanup_memory()
        else:
            logger.error("CUDA not available!")
            return

        # Set torch settings for STABLE training (prioritize stability over speed)
        torch.backends.cudnn.benchmark = False  # DISABLED: Can cause memory access issues
        torch.backends.cuda.matmul.allow_tf32 = False  # [5] Set TF32 OFF for determinism
        torch.backends.cudnn.allow_tf32 = False  # [5] Set TF32 OFF for determinism
        
        # Set conservative memory allocation strategy
        # torch.cuda.set_per_process_memory_fraction(0.90)  # REMOVED: Prevent memory pressure
        
        # Dynamically select compute dtype based on GPU capability (bf16 needs SM >= 8.0)
        cc_major, cc_minor = (0, 0)
        if torch.cuda.is_available():
            cc_major, cc_minor = torch.cuda.get_device_capability()
        bf16_supported = cc_major >= 8
        compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,   # Nested quantisation
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        
        # Load model and tokenizer
        logger.info(f"Loading model {args.model_name_or_path}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map={"": current_device},
            trust_remote_code=True,
            use_cache=False,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        logger.info("Configuring processor to prevent token/feature mismatch...")
        if hasattr(processor, 'image_processor'):
            # Set fixed resolution to prevent dynamic resizing
            fixed_size = args.image_size
            processor.image_processor.size = {"height": fixed_size, "width": fixed_size}
            processor.image_processor.max_pixels = fixed_size * fixed_size
            processor.image_processor.min_pixels = fixed_size * fixed_size
            processor.image_processor.do_resize = True
            processor.image_processor.do_center_crop = False
            
            # Disable any dynamic processing that could change token count
            if hasattr(processor.image_processor, 'do_rescale'):
                processor.image_processor.do_rescale = True
            if hasattr(processor.image_processor, 'do_normalize'):
                processor.image_processor.do_normalize = True
                
            logger.info(f"Processor locked to {fixed_size}x{fixed_size} - prevents dynamic token count changes")
        else:
            logger.error("Cannot configure image processor - this WILL cause token/feature mismatches!")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        
        # Define LoRA config for better convergence
        peft_config = LoraConfig(
            r=args.lora_rank,  # Use command-line argument
            lora_alpha=args.lora_alpha,  # Use command-line argument
            lora_dropout=args.lora_dropout,  # Use command-line argument
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
        )
        
        # Apply LoRA
        logger.info("Applying LoRA adapters")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        model.train()
        
        # Re-check memory after model loading
        cleanup_memory()
        available_memory_post, _ = get_available_gpu_memory()
        logger.info(f"Available GPU memory after model loading: {available_memory_post:.2f} GB")
        
        # Warn about potential OOM with multi-image data
        if args.batch_size > 1:
            logger.warning(f"Using batch_size={args.batch_size} with multi-image data. "
                         f"If you encounter OOM errors, consider reducing to batch_size=1 "
                         f"or limiting images with --max_images_per_sample")
        
        logger.info(f"Training settings: batch_size={args.batch_size}, "
                   f"grad_acc={args.gradient_accumulation_steps}, "
                   f"effective_batch={args.batch_size * args.gradient_accumulation_steps}")
        
        # Load datasets
        train_dataset, val_dataset = load_processed_datasets(args.processed_data_dir)
        
        # Calculate image cache size based on available memory
       
        image_cache_size_mb = min(8192, int(available_memory_post * 1024 * 0.20))
        logger.info(f"Using optimized {image_cache_size_mb} MB for image cache")
        
        # Initialize data collator
        data_collator = MultimodalDataCollator(
            tokenizer=tokenizer,
            processor=processor,
            model=model,
            max_seq_length=args.max_seq_length,
            image_cache_size_mb=image_cache_size_mb,
            image_size=args.image_size,
            max_images_per_sample=args.max_images_per_sample,
        )
        
        # Calculate training steps
        num_training_steps = len(train_dataset) * args.num_epochs // (args.batch_size * args.gradient_accumulation_steps)
        logger.info(f"Total training steps: {num_training_steps}")
        
        # Initialize callbacks
        callbacks = []
        if args.early_stopping_patience > 0:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            )
            callbacks.append(early_stopping_callback)
        
        # Create SFTConfig
        logger.info("Creating training configuration")
        sft_config = SFTConfig(
            # SFT specific arguments
            dataset_text_field="question",
            max_seq_length=args.max_seq_length,
            packing=False,
            
            # Training arguments
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,  # Use command-line argument
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,  # Set to 1 for memory safety
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,  # Use command-line argument
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            logging_dir=f"{args.output_dir}/logs",
            logging_steps=max(10, num_training_steps // 100),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="wandb" if args.use_wandb else "none",
            run_name=args.run_name if args.use_wandb else None,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            resume_from_checkpoint=args.resume_from_checkpoint,
            
            # dataloader settings
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            dataloader_pin_memory=False,
            
            # Performance settings
            tf32=False,
            skip_memory_metrics=True,
            # eval_accumulation_steps=8, # Removed for simplicity and safety
            save_safetensors=True,
            
            # Additional optimizations
            max_grad_norm=0.5,
            group_by_length=False,
            # neftune_noise_alpha=5,
            prediction_loss_only=True,
            include_inputs_for_metrics=False,
            label_names=["labels"]
        )
        
        # Initialize SFT trainer
        logger.info("Initializing SFT trainer")
        trainer = CustomSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            peft_config=peft_config,
        )
        
        # Train the model with robust error handling
        logger.info("Starting training with CUDA error recovery")
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Clear CUDA cache before training attempt
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Add CUDA error checking
                logger.info(f"Training attempt {retry_count + 1}/{max_retries}")
                start_time = time.time()
                
                trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
                break  # Success - exit retry loop
                
            except RuntimeError as e:
                if "illegal memory access" in str(e).lower() or "cuda" in str(e).lower() or "out of memory" in str(e).lower():
                    retry_count += 1
                    logger.error(f"CUDA error encountered (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        logger.info("Attempting recovery...")
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        gc.collect()
                        del trainer
                        trainer = None
                        # Reduce batch size for stability
                        if hasattr(sft_config, 'per_device_train_batch_size'):
                            old_batch_size = sft_config.per_device_train_batch_size
                            sft_config.per_device_train_batch_size = max(1, old_batch_size // 2)
                            sft_config.gradient_accumulation_steps *= 2
                            logger.info(f"Reduced batch size from {old_batch_size} to {sft_config.per_device_train_batch_size}")
                        trainer = CustomSFTTrainer(
                            model=model,
                            args=sft_config,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            data_collator=data_collator,
                            callbacks=callbacks,
                            peft_config=peft_config,
                        )
                        time.sleep(5)
                    else:
                        logger.error("Max retries exceeded. Training failed.")
                        raise
                else:
                    raise
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"GPU OOM during training: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                raise
        
        # Save the final model
        logger.info(f"Saving model to {args.output_dir}/final")
        trainer.save_model(f"{args.output_dir}/final")
        
        logger.info(f"Training complete. Model saved to {args.output_dir}/final")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune QwenVL")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data_qwenvl")
    parser.add_argument("--output_dir", type=str, default="finetuned_qwenvl")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size per device. Default 1 for multi-image stability.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Gradient accumulation steps. Default 32 for effective batch size of 32.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_images_per_sample", type=int, default=0, 
                        help="Maximum images per sample (0=unlimited). Set to 1 or 2 to reduce memory usage.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="qwenvl-medical")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001)
    args = parser.parse_args()
    
    main(args) 