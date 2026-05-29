from dataclasses import dataclass

@dataclass
class LLMConfig:
    # LLM Model architecture : Qwen 7B
    batch_size:int = 8
    gradient_accumulation_steps:int = 2
    lr:float = 2e-4
    lr_scheduler_type:str = "cosine"
    num_train_epochs:int = 10
    logging_steps:int = 10
    max_seq_length:int = 4096