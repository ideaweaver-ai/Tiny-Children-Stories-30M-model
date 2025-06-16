import torch
from data.data_processor import DataProcessor
from model.gpt import GPT
from config import GPTConfig, TrainingConfig
from training.trainer import Trainer
import tiktoken
import os

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("lora_checkpoints", exist_ok=True)
    
    # Initialize data processor and prepare dataset
    print("Preparing dataset...")
    data_processor = DataProcessor()
    data_processor.prepare_dataset()
    
    # Initialize model
    print("Initializing model...")
    model_config = GPTConfig()
    model = GPT(model_config)
    model = model.to(device)
    
    # Initialize trainer
    print("Starting training...")
    training_config = TrainingConfig()
    trainer = Trainer(model, training_config, device)
    
    # Train the model
    best_model_path = trainer.train()
    
    # Load the best model for inference
    print("Loading best model for inference...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Example generation
    print("\nGenerating example story...")
    enc = tiktoken.get_encoding("gpt2")
    sentence = "Once upon a time there was a pumpkin."
    context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim=0).to(device)
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))
    
    print("\nTraining completed! You can now run LoRA finetuning using:")
    print("python src/finetune_lora.py")

if __name__ == "__main__":
    main() 