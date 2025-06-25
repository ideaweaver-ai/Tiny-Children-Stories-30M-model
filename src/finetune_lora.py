import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os
from tqdm.auto import tqdm
from datetime import datetime
import time
from model.gpt import GPTConfig

# Load the base model
print("Loading base model...")
torch.serialization.add_safe_globals([GPTConfig])
base_model_path = "Tiny-Children-Stories-Collection-model.pt"
checkpoint = torch.load(base_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.load_state_dict(checkpoint['model'])

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Get PEFT model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and prepare dataset
print("Loading dataset...")
data = torch.load("finetune.bin")
train_data = data['input_ids']

# Training configuration
batch_size = 4
learning_rate = 1e-4
num_epochs = 3
gradient_accumulation_steps = 4

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create checkpoints directory
os.makedirs("lora_checkpoints", exist_ok=True)

# Record start time
start_time = time.time()
start_datetime = datetime.now()
print(f"\nLoRA Finetuning started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

# Training loop
print("Starting LoRA finetuning...")
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(range(0, len(train_data), batch_size))
    
    for i in progress_bar:
        batch = train_data[i:i + batch_size]
        input_ids = torch.stack(batch).to(model.device)
        
        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        # Gradient accumulation
        if (i + batch_size) % (batch_size * gradient_accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(progress_bar)
    print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    # Save checkpoint if it's the best so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = "lora_checkpoints/Tiny-Children-Stories-Collection-LoRA-model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, checkpoint_path)
        print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Save regular checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"lora_checkpoints/Tiny-Children-Stories-Collection-LoRA-model_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

# Record end time and calculate duration
end_time = time.time()
end_datetime = datetime.now()
duration = end_time - start_time
hours, remainder = divmod(duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\nLoRA Finetuning completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total finetuning time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

print("LoRA finetuning completed!") 
