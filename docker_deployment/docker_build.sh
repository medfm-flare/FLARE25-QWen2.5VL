#!/bin/bash

# QwenVL Docker Build Script
# FLARE 2025 Medical Multimodal VQA Challenge

set -e

echo "========================================="
echo "Building QwenVL Inference Docker Image"
echo "========================================="

# Default values
IMAGE_NAME=${IMAGE_NAME:-"qwenvl-inference"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
CONTEXT_DIR=${CONTEXT_DIR:-"."}
NO_CACHE=${NO_CACHE:-"false"}
PLATFORM=${PLATFORM:-"linux/amd64"}

echo "Build Configuration:"
echo "  Image Name: $IMAGE_NAME"
echo "  Image Tag: $IMAGE_TAG"
echo "  Context Directory: $CONTEXT_DIR"
echo "  No Cache: $NO_CACHE"
echo "  Platform: $PLATFORM"
echo "========================================="

# Check if required files exist
REQUIRED_FILES=("Dockerfile" "requirements.txt" "inference.py" "docker_inference.sh")

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$CONTEXT_DIR/$file" ]; then
        echo "ERROR: Required file not found: $file"
        echo "Please ensure all required files are in the context directory: $CONTEXT_DIR"
        exit 1
    fi
done

echo "All required files found. Starting build..."

# Build Docker arguments
BUILD_ARGS=(
    "build"
    "--platform" "$PLATFORM"
    "-t" "$IMAGE_NAME:$IMAGE_TAG"
    "-f" "$CONTEXT_DIR/Dockerfile"
)

# Add no-cache flag if requested
if [ "$NO_CACHE" == "true" ]; then
    BUILD_ARGS+=("--no-cache")
fi

# Add build context
BUILD_ARGS+=("$CONTEXT_DIR")

echo "Running Docker build command:"
echo "docker ${BUILD_ARGS[*]}"
echo "========================================="

# Run the build
docker "${BUILD_ARGS[@]}"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Docker image built successfully!"
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    
    # Show image info
    echo ""
    echo "Image information:"
    docker images "$IMAGE_NAME:$IMAGE_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    echo "To run the container:"
    echo "1. With base model only:"
    echo "   docker run --gpus all -v /path/to/dataset:/app/input/organized_dataset -v /path/to/output:/app/output $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "2. With LoRA weights:"
    echo "   docker run --gpus all -v /path/to/dataset:/app/input/organized_dataset -v /path/to/lora:/app/lora -v /path/to/output:/app/output -e LORA_WEIGHTS=/app/lora $IMAGE_NAME:$IMAGE_TAG"
    echo ""
    echo "3. With custom settings:"
    echo "   docker run --gpus all -v /path/to/dataset:/app/input/organized_dataset -v /path/to/output:/app/output -e MAX_TOKENS=512 -e VERBOSE=true $IMAGE_NAME:$IMAGE_TAG"
    
    echo "========================================="
else
    echo "ERROR: Docker build failed!"
    exit 1
fi 