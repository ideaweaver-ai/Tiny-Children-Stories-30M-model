import os
import sys
import torch
import time
import datetime
import traceback
import argparse
from model.gpt import GPT, GPTConfig
from training.trainer import Trainer
import numpy as np

def get_project_root():
    """Get the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tiny-DevOps-GPT model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size for training')
    parser.add_argument('--max-iters', type=int, default=20000, help='Maximum number of training iterations')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Interval for evaluation')
    parser.add_argument('--eval-iters', type=int, default=200, help='Number of iterations for evaluation')
    parser.add_argument('--learning-rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-iters', type=int, default=2000, help='Number of warmup iterations')
    parser.add_argument('--lr-decay-iters', type=int, default=20000, help='Number of iterations for learning rate decay')
    parser.add_argument('--min-lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    return parser.parse_args()

def check_requirements():
    """Check if all required files and directories exist"""
    project_root = get_project_root()
    required_dirs = [
        os.path.join(project_root, 'src', 'data'),
        os.path.join(project_root, 'src', 'model'),
        os.path.join(project_root, 'src', 'training'),
        os.path.join(project_root, 'checkpoints')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    required_files = [
        os.path.join(project_root, 'src', 'data', 'train.bin'),
        os.path.join(project_root, 'src', 'data', 'validation.bin')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

def check_gpu():
    """Check GPU availability and return device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set memory allocation
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("No GPU available. Using CPU (this will be very slow!)")
    return device

def prepare_dataset():
    """Prepare the dataset"""
    try:
        from datasets import load_dataset
        dataset = load_dataset('ajibawa-2023/Children-Stories-Collection')
        print(f"Dataset loaded with {len(dataset['train'])} examples")
        return True
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return False

def train_base_model(device, args):
    """Train the base model"""
    try:
        # Model configuration
        config = GPTConfig(
            vocab_size=50257,  # GPT-2 vocabulary size
            block_size=1024,   # Context window
            n_layer=6,         # Number of transformer layers
            n_head=8,          # Number of attention heads
            n_embd=512,        # Embedding dimension
            dropout=0.1,       # Dropout rate
            bias=True,         # Use bias in linear layers
        )
        torch.serialization.add_safe_globals([GPTConfig])
        
        # Initialize model
        model = GPT(config)
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            min_lr=args.min_lr,
            checkpoint_dir=args.checkpoint_dir
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            if not trainer.load_checkpoint(args.resume):
                print("Failed to load checkpoint. Starting from scratch.")
        
        # Start training
        print("\nStarting base model training...")
        best_model_path = trainer.train()
        
        # Plot training metrics
        trainer.plot_metrics()
        
        print(f"\nBase model training completed. Best model saved at: {best_model_path}")
        return best_model_path
        
    except Exception as e:
        print(f"\n[!] Error during base model training: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return None

def finetune_lora(base_model_path, device):
    """Perform LoRA finetuning"""
    try:
        print("\nStarting LoRA finetuning...")
        # Import here to avoid circular imports
        from finetune_lora import finetune
        lora_model_path = finetune(base_model_path, device)
        print(f"\nLoRA finetuning completed. Model saved at: {lora_model_path}")
        return lora_model_path
    except Exception as e:
        print(f"\n[!] Error during LoRA finetuning: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        return None

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Check requirements
        print("Checking requirements...")
        check_requirements()
        
        # Check GPU
        print("\nChecking GPU availability...")
        device = check_gpu()
        
        # Prepare dataset
        print("\nPreparing dataset...")
        if not prepare_dataset():
            raise Exception("Dataset preparation failed")
        
        # Train base model
        base_model_path = train_base_model(device, args)
        if not base_model_path:
            raise Exception("Base model training failed")
        
        # Ask about finetuning
        while True:
            response = input("\nDo you want to perform LoRA finetuning? (y/n): ").lower()
            if response in ['y', 'n']:
                break
            print("Please answer 'y' or 'n'")
        
        if response == 'y':
            lora_model_path = finetune_lora(base_model_path, device)
            if not lora_model_path:
                raise Exception("LoRA finetuning failed")
        
        print("\nTraining process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Unexpected error: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
