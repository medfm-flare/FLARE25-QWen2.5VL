#!/usr/bin/env python3
"""
Download FLARE 2025 fine-tuned QwenVL models for Docker container
"""

import os
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftConfig

def download_models():
    """Download base model and FLARE 2025 fine-tuned adapter"""
    
    print('=' * 50)
    print('Downloading FLARE 2025 fine-tuned model...')
    print('=' * 50)
    
    # Download base model
    base_model = 'Qwen/Qwen2.5-VL-7B-Instruct'
    print(f'📥 Downloading base model: {base_model}')
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        processor = AutoProcessor.from_pretrained(base_model)
        print(f'✅ Base model downloaded successfully')
    except Exception as e:
        print(f'❌ Failed to download base model: {e}')
        raise
    
    # Download FLARE 2025 fine-tuned adapter
    adapter_model = 'leoyinn/flare25-qwen2.5vl'
    print(f'📥 Downloading FLARE adapter: {adapter_model}')
    
    try:
        config = PeftConfig.from_pretrained(adapter_model)
        print(f'✅ FLARE adapter downloaded successfully')
    except Exception as e:
        print(f'❌ Failed to download FLARE adapter: {e}')
        raise
    
    print('=' * 50)
    print('🎉 All models downloaded successfully!')
    print('=' * 50)

if __name__ == "__main__":
    download_models() 