from dataclasses import dataclass

@dataclass
class LLMConfig:
    # LLM Model architecture : Qwen 7B
    batch_size: int = 4               
    gradient_accumulation_steps: int = 4   # Effective batch size = 16
    lr: float = 2e-4
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 3             
    logging_steps: int = 50
    max_seq_length: int = 4096