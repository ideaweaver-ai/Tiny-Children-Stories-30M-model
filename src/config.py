from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    max_iters: int = 20000
    warmup_steps: int = 1000
    min_lr: float = 5e-4
    eval_iters: int = 500
    batch_size: int = 32
    block_size: int = 128
    gradient_accumulation_steps: int = 32
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-9
    max_norm: float = 0.5 