#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
LORA_CHECKPOINT_DIR="${LORA_CHECKPOINT_DIR:-${PROJECT_ROOT}/lora_checkpoints}"
REQUIRED_SPACE_MB="${REQUIRED_SPACE_MB:-1000}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

# Function to handle errors
handle_error() {
    print_error "$1"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check disk space
check_disk_space() {
    local available_space_mb=$(df -m . | awk 'NR==2 {print $4}')
    if [ "$available_space_mb" -lt "$REQUIRED_SPACE_MB" ]; then
        print_warning "Low disk space. Only ${available_space_mb}MB available, ${REQUIRED_SPACE_MB}MB required."
        return 1
    fi
    return 0
}

# Function to check GPU memory
check_gpu_memory() {
    if command_exists nvidia-smi; then
        local total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        local used_memory=$((total_memory - free_memory))
        print_status "GPU Memory: ${used_memory}MB used, ${free_memory}MB free of ${total_memory}MB total"
    fi
}

# Function to create project structure
create_project_structure() {
    print_status "Creating project structure..."
    mkdir -p "${PROJECT_ROOT}/src/data" \
            "${PROJECT_ROOT}/src/model" \
            "${PROJECT_ROOT}/src/training" \
            "${CHECKPOINT_DIR}" \
            "${LORA_CHECKPOINT_DIR}" || handle_error "Failed to create directories"
}

# Function to setup virtual environment
setup_virtual_env() {
    print_status "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}" || handle_error "Failed to create virtual environment"
    source "${VENV_PATH}/bin/activate" || handle_error "Failed to activate virtual environment"
    
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt || handle_error "Failed to install requirements"
}

# Function to prepare dataset
prepare_dataset() {
    print_status "Preparing dataset..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    # Create a Python script to process the data
    cat > process_data.py << 'EOF'
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import DataProcessor

def main():
    print("[+] Processing dataset into binary files...")
    processor = DataProcessor()
    processor.prepare_dataset()
    print("[+] Data processing completed successfully!")

if __name__ == "__main__":
    main()
EOF

    # Run the data processing script
    python3 process_data.py || handle_error "Failed to process dataset"
    
    # Verify the files were created
    if [ ! -f "${PROJECT_ROOT}/src/data/train.bin" ] || [ ! -f "${PROJECT_ROOT}/src/data/validation.bin" ]; then
        handle_error "Data processing failed - required files not created"
    fi
}

# Function to train base model
train_base_model() {
    print_status "Starting base model training..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    python3 src/run_training.py \
        --batch-size "${BATCH_SIZE:-12}" \
        --max-iters "${MAX_ITERS:-20000}" \
        --eval-interval "${EVAL_INTERVAL:-1000}" \
        --eval-iters "${EVAL_ITERS:-200}" \
        --learning-rate "${LEARNING_RATE:-6e-4}" \
        --weight-decay "${WEIGHT_DECAY:-0.1}" \
        --warmup-iters "${WARMUP_ITERS:-2000}" \
        --lr-decay-iters "${LR_DECAY_ITERS:-20000}" \
        --min-lr "${MIN_LR:-6e-5}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" || handle_error "Base model training failed"
}

# Function to perform LoRA finetuning
finetune_lora() {
    while true; do
        read -p "Do you want to perform LoRA finetuning? (y/n) " do_finetune
        case $do_finetune in
            [Yy]* )
                print_status "Starting LoRA finetuning..."
                cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
                python3 src/finetune_lora.py \
                    --base-model "${CHECKPOINT_DIR}/Tiny-Children-Stories-Collection-model.pt" \
                    --batch-size "${LORA_BATCH_SIZE:-8}" \
                    --learning-rate "${LORA_LEARNING_RATE:-1e-4}" \
                    --max-iters "${LORA_MAX_ITERS:-5000}" \
                    --checkpoint-dir "${LORA_CHECKPOINT_DIR}" || handle_error "LoRA finetuning failed"
                break
                ;;
            [Nn]* )
                print_status "Skipping LoRA finetuning..."
                break
                ;;
            * )
                echo "Please answer 'y' or 'n'"
                ;;
        esac
    done
}

# Function to test the trained model
test_model() {
    while true; do
        read -p "Do you want to test the trained model? (y/n) " do_test
        case $do_test in
            [Yy]* )
                print_status "Testing the trained model..."
                cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
                
                # Create test prompts
                prompts=(
                    "Once upon a time"
                    "In a magical forest"
                    "The little robot"
                    "The brave knight"
                )
                
                # Test each prompt
                for prompt in "${prompts[@]}"; do
                    print_status "Testing with prompt: '$prompt'"
                    python3 src/generate.py \
                        --model_path "${CHECKPOINT_DIR}/Tiny-Children-Stories-Collection-model.pt" \
                        --prompt "$prompt" \
                        --max_tokens 100 \
                        --temperature 0.8 \
                        --top_k 40
                    echo
                done
                break
                ;;
            [Nn]* )
                print_status "Skipping model testing..."
                break
                ;;
            * )
                echo "Please answer 'y' or 'n'"
                ;;
        esac
    done
}

# Main execution
main() {
    # Check Python installation
    if ! command_exists python3; then
        handle_error "Python 3 is not installed. Please install Python 3.8 or higher."
    fi

    # Check disk space
    check_disk_space || handle_error "Insufficient disk space"

    # Create project structure
    create_project_structure

    # Check for CUDA availability
    if command_exists nvidia-smi; then
        print_status "CUDA is available. Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        DEVICE="cuda"
        check_gpu_memory
    else
        print_warning "CUDA not found. Training will be slower on CPU."
        DEVICE="cpu"
    fi

    # Setup virtual environment and dependencies
    setup_virtual_env

    # Start the training process
    print_status "Starting training process..."
    START_TIME=$(date +%s)

    # Prepare and process dataset
    prepare_dataset

    # Train base model
    train_base_model

    # Perform LoRA finetuning if requested
    finetune_lora

    # Test the trained model
    test_model

    # Calculate and display total time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    SECONDS=$((DURATION % 60))

    print_status "Training process completed!"
    print_status "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

    # Deactivate virtual environment
    deactivate

    print_status "Setup and training completed successfully!"
    print_status "You can find the trained models in:"
    echo "  - Base model: ${CHECKPOINT_DIR}/Tiny-Children-Stories-Collection-model.pt"
    echo "  - LoRA model: ${LORA_CHECKPOINT_DIR}/Tiny-Children-Stories-Collection-LoRA-model.pt"

    # Print final GPU memory status
    if command_exists nvidia-smi; then
        print_status "Final GPU Memory Status:"
        nvidia-smi
    fi
}

# Run main function
main 