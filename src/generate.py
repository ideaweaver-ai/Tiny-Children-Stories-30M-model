import os
import torch
import argparse
import tiktoken
from model.gpt import GPT, GPTConfig
from torch.serialization import add_safe_globals

# Add GPTConfig to safe globals for loading
add_safe_globals([GPTConfig])

def load_encoder_decoder():
    """Load the encoder and decoder for text processing"""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc

def load_model(model_path, device):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with the same config
    config = checkpoint['config']
    model = GPT(config)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    return model

def generate_text(model, encoder, decoder, prompt, max_tokens=100, temperature=0.8, top_k=40):
    """Generate text from the model"""
    # Get device from model parameters
    device = next(model.parameters()).device
    
    # Encode the prompt
    context = torch.tensor(encoder.encode(prompt), dtype=torch.long, device=device)
    context = context.unsqueeze(0)  # Add batch dimension
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get the predictions
            logits, _ = model(context)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the context
            context = torch.cat((context, next_token), dim=1)
            
            # Stop if we generate an end token
            if next_token.item() == encoder.eot_token:
                break
    
    # Decode the generated tokens
    generated_text = decoder.decode(context[0].tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Generate text using the trained model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                      help='Path to the trained model')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                      help='Prompt to start the generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40,
                      help='Top-k sampling parameter')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load encoder/decoder
    encoder, decoder = load_encoder_decoder()
    
    # Generate text
    print("\nGenerating text...")
    print(f"Prompt: {args.prompt}")
    generated_text = generate_text(
        model, 
        encoder, 
        decoder,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == '__main__':
    main() 