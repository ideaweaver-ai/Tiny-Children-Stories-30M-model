import tiktoken
import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

def load_encoder_decoder():
    """Load the encoder and decoder for text processing"""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc

class DataProcessor:
    def __init__(self):
        # Initialize tokenizer with specific settings for children's stories
        self.enc = tiktoken.get_encoding("gpt2")
        # Add special tokens for story structure
        self.special_tokens = {
            "prompt_start": "<|prompt|>",
            "prompt_end": "</|prompt|>",
            "story_start": "<|story|>",
            "story_end": "</|story|>"
        }
        # Ensure data directory exists
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory: {self.data_dir}")
        
    def preprocess_text(self, text):
        # Basic text cleaning
        text = text.lower()  # Convert to lowercase for consistency
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())  # Normalize whitespace
        return text
        
    def process(self, example):
        # Preprocess both prompt and story
        prompt = self.preprocess_text(example['prompt'])
        story = self.preprocess_text(example['text'])
        
        # Create structured text with special tokens
        full_text = (
            f"{self.special_tokens['prompt_start']} {prompt} {self.special_tokens['prompt_end']} "
            f"{self.special_tokens['story_start']} {story} {self.special_tokens['story_end']}"
        )
        
        # Tokenize with error handling
        try:
            ids = self.enc.encode_ordinary(full_text)
            # Ensure the sequence isn't too long
            if len(ids) > 1024:  # GPT-2's max context length
                ids = ids[:1024]
            out = {'ids': ids, 'len': len(ids)}
            return out
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            # Return empty sequence in case of error
            return {'ids': [], 'len': 0}
        
    def prepare_dataset(self):
        # Load the Children Stories Collection dataset
        ds = load_dataset("ajibawa-2023/Children-Stories-Collection")
        
        train_bin_path = os.path.join(self.data_dir, "train.bin")
        print(f"Checking for existing train.bin at: {train_bin_path}")
        
        if not os.path.exists(train_bin_path):
            print("train.bin not found, processing dataset...")
            # Filter out examples that are too short or too long
            def filter_by_length(example):
                return 50 <= example['text_token_length'] <= 1000
            
            ds = ds.filter(filter_by_length)
            
            # Split the dataset into train, validation, and finetune sets
            train_val_test = ds["train"].train_test_split(test_size=0.2, seed=42)
            val_finetune = train_val_test["test"].train_test_split(test_size=0.5, seed=42)
            
            # Create a new dataset dictionary with all splits
            ds = {
                "train": train_val_test["train"],
                "validation": val_finetune["train"],
                "finetune": val_finetune["test"]
            }
            
            print(f"Dataset split sizes:")
            print(f"Training set: {len(ds['train'])} examples")
            print(f"Validation set: {len(ds['validation'])} examples")
            print(f"Finetune set: {len(ds['finetune'])} examples")
            
            # Process each split
            for split_name, split_data in ds.items():
                print(f"\nProcessing {split_name} split...")
                tokenized = split_data.map(
                    self.process,
                    remove_columns=['text', 'prompt', 'text_token_length'],
                    desc=f"tokenizing {split_name} split",
                    num_proc=8,
                )
                
                # Save each split to its own binary file
                filename = os.path.join(self.data_dir, f"{split_name}.bin")
                print(f"Saving {split_name} split to: {filename}")
                
                arr_len = np.sum(tokenized['len'], dtype=np.uint64)
                dtype = np.uint16
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
                total_batches = 1024

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                    batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
                
                # Verify file was created
                if os.path.exists(filename):
                    print(f"Successfully created {filename}")
                    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
                else:
                    raise RuntimeError(f"Failed to create {filename}")
                
        else:
            print(f"Found existing train.bin at: {train_bin_path}")
            print(f"File size: {os.path.getsize(train_bin_path) / (1024*1024):.2f} MB")
                
        return ds 