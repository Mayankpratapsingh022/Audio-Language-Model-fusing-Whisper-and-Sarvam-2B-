import dataclasses
from typing import Optional, Tuple

@dataclasses.dataclass
class ModelConfig:
    audio_model_id: str = "openai/whisper-small"
    text_model_id: str = "sarvamai/sarvam-2b-v0.5"
    hidden_size: int = 2048
    projector_act: str = "gelu"
    stack_factor: int = 8

@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 4
    accum_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int = 1000 # Use either epochs or steps
    
    # Paths
    output_dir: str = "./checkpoints"
    # data_path: str = "./data/train.jsonl" # REMOVED
    dataset_name: str = "fixie-ai/common_voice_17_0"
    dataset_subset: str = "hi" # Hindi
    dataset_split: str = "train"
    val_dataset_split: str = "validation"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Misc
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
