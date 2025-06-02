#!/usr/bin/env python3
"""
Test script to validate FLARE 2025 QWen2.5-VL baseline installation
"""

import sys
import os
from importlib import import_module

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        import_module(module_name)
        print(f"✅ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ {package_name} import failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Device 0: {device_name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("⚠️  CUDA not available - training will be very slow")
            return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_access():
    """Test if model can be accessed from HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        model_info = api.model_info("Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"✅ Model accessible: Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"   Model size: {model_info.safetensors.total / (1024**3):.1f} GB")
        return True
    except Exception as e:
        print(f"⚠️  Model access test failed: {e}")
        print("   You may need to authenticate with HuggingFace")
        return False

def test_directory_structure():
    """Test if required directories exist or can be created"""
    required_dirs = ["logs", "organized_dataset"]
    all_good = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ Directory created: {dir_name}")
                os.rmdir(dir_name)  # Clean up test directory
            except Exception as e:
                print(f"❌ Cannot create directory {dir_name}: {e}")
                all_good = False
    
    return all_good

def main():
    print("=" * 60)
    print("FLARE 2025 QWen2.5-VL Baseline Installation Test")
    print("=" * 60)
    print()
    
    # Test core dependencies
    print("Testing core dependencies...")
    core_deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("bitsandbytes", "BitsAndBytes"),
    ]
    
    core_success = all(test_import(mod, name) for mod, name in core_deps)
    print()
    
    # Test image processing
    print("Testing image processing dependencies...")
    image_deps = [
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
    ]
    
    image_success = all(test_import(mod, name) for mod, name in image_deps)
    print()
    
    # Test CUDA
    print("Testing CUDA...")
    cuda_success = test_cuda()
    print()
    
    # Test model access
    print("Testing model access...")
    model_success = test_model_access()
    print()
    
    # Test directory structure
    print("Testing directory structure...")
    dir_success = test_directory_structure()
    print()
    
    # Test custom modules
    print("Testing custom modules...")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils.metrics import GREENScorer
        print("✅ Custom metrics module imported successfully")
        custom_success = True
    except Exception as e:
        print(f"❌ Custom metrics module import failed: {e}")
        custom_success = False
    print()
    
    # Summary
    print("=" * 60)
    print("Installation Test Summary")
    print("=" * 60)
    
    all_success = core_success and image_success and dir_success and custom_success
    
    if all_success:
        print("✅ All required dependencies are installed correctly!")
        if not cuda_success:
            print("⚠️  Warning: CUDA is not available. Training will be slow.")
        if not model_success:
            print("⚠️  Warning: Model access failed. You may need to:")
            print("   1. Check your internet connection")
            print("   2. Authenticate with HuggingFace: huggingface-cli login")
    else:
        print("❌ Some dependencies are missing. Please install them:")
        print("   pip install -r requirements.txt")
    
    print()
    print("Next steps:")
    print("1. Organize your FLARE 2025 dataset in the 'organized_dataset' folder")
    print("2. Run: python prepare_data.py")
    print("3. Run: python finetune_qwenvl.py")
    print("4. Run: python evaluate_model.py")
    print()
    print("Or use the pipeline runner: ./run_pipeline.sh")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main()) 