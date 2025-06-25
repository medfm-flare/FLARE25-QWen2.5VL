# FLARE 2025 QwenVL Docker Deployment

This folder contains Docker deployment files for the FLARE 2025 Medical Multimodal VQA Challenge using QwenVL with fine-tuned adapters.

## Quick Start with Pre-built Image

If you have the pre-built Docker image (`qwenvl-flare2025.tar.gz`) from the [Docker Imgae Archive](https://huggingface.co/leoyinn/flare25-qwen2.5vl/blob/main/qwenvl-flare2025.tar.gz), you can skip the build step:

### Load Pre-built Image

```bash
# Load the Docker image (if you have the tar.gz file)
docker load -i qwenvl-flare2025.tar.gz
```

### Run Inference

```bash
# Run inference on your dataset
docker run --gpus all \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm qwenvl-inference:latest
```

## Build Docker from Source

### Prerequisites

- Docker installed with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- At least 24GB disk space for the Docker image
- Internet connection for downloading models

### Build Docker Image

```bash
# Build the Docker image
docker build -f Dockerfile -t qwenvl-inference .
```

> Note: Don't forget the `.` at the end. The build process will automatically download the FLARE 2025 fine-tuned model and base QwenVL model.

### Alternative: Use Build Script

```bash
# Make the build script executable and run it
chmod +x docker_build.sh
./docker_build.sh
```

## Running Inference

### Method 1: Direct Docker Run

```bash
# Run inference with volume mounts for input and output
docker run --gpus all \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm qwenvl-inference:latest
```

### Method 2: Interactive Mode

```bash
# Run in interactive mode for debugging
docker run --gpus all -it \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    qwenvl-inference:latest /bin/bash
```

### Output
The inference will generate a `predictions.json` file in the output directory containing the model's answers to all questions.

## Configuration Options

The Docker container supports several environment variables for configuration:

```bash
# Custom model configuration
docker run --gpus all \
    -e FLARE_MODEL_PATH="/app/models/flare25-qwen2.5vl" \
    -e BASE_MODEL_PATH="/app/models/Qwen/Qwen2.5-VL-7B-Instruct" \
    -e MAX_NEW_TOKENS="512" \
    -e TEMPERATURE="0.0" \
    -v $(pwd)/organized_dataset:/app/input/organized_dataset \
    -v $(pwd)/predictions:/app/output \
    --rm qwenvl-inference:latest
```

## Memory Requirements

- **GPU Memory**: Minimum 24GB VRAM recommended
- **System RAM**: At least 16GB
- **Docker Container**: Limited to 8GB by default (configurable)

## Testing the Installation

Run the test script to verify everything is working:

```bash
# Test the Docker container
docker run --gpus all --rm qwenvl-inference:latest python test_installation.py
```

## Troubleshooting

### Permission Issues
If you encounter permission denied errors:
```bash
chmod -R 777 ./organized_dataset ./predictions
```

### GPU Not Detected
Ensure NVIDIA Container Toolkit is installed:
```bash
# Check if GPU is available in Docker
docker run --gpus all --rm nvidia/cuda:11.8-base nvidia-smi
```

### Out of Memory Errors
- Reduce batch size by setting environment variable: `-e BATCH_SIZE="1"`
- Ensure no other GPU processes are running
- Check available GPU memory: `nvidia-smi`

### Model Download Issues
If model download fails during build, try:
```bash
# Build with no cache to force re-download
docker build --no-cache -f Dockerfile -t qwenvl-inference .
```

## Saving and Distributing the Image

### Save Docker Image

```bash
# Save the Docker image for distribution
docker save qwenvl-inference:latest -o qwenvl-flare2025.tar
gzip qwenvl-flare2025.tar
```

### Load Docker Image on Another Machine

```bash
# Load the Docker image on target machine
docker load -i qwenvl-flare2025.tar.gz
```

## Model Information

This Docker container uses:
- **Base Model**: Qwen/Qwen2.5-VL-7B-Instruct
- **Fine-tuned Adapter**: leoyinn/flare25-qwen2.5vl
- **Training Data**: 19 medical datasets across 8 modalities
- **LoRA Configuration**: r=16, alpha=32

## Files in this Directory

- `Dockerfile`: Main Docker configuration
- `docker_build.sh`: Automated build script
- `docker_inference.sh`: Container entrypoint script
- `inference.py`: Main inference script
- `download_models.py`: Model download utility
- `requirements.txt`: Python dependencies
- `test_docker.sh`: Docker testing script
- `.dockerignore`: Docker build ignore patterns

## Citation

If you use this Docker container or the FLARE 2025 fine-tuned model, please cite:

```bibtex
@misc{flare2025-qwenvl,
  title={FLARE 2025 QwenVL Fine-tuned Model for Medical Multimodal VQA},
  author={[Authors]},
  year={2025},
  howpublished={\url{https://github.com/medfm-flare/FLARE25-QWen2.5VL}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [FLARE25-QWen2.5VL Issues](https://github.com/medfm-flare/FLARE25-QWen2.5VL/issues)
- Model Hub: [leoyinn/flare25-qwen2.5vl](https://huggingface.co/leoyinn/flare25-qwen2.5vl) 