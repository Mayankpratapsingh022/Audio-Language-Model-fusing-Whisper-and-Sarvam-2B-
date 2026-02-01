
import os
import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from .config import TrainConfig, ModelConfig
from .model import MultiModalModel
from .data import AudioTextDataset, DataCollator

def train():
    # Load Configs
    train_config = TrainConfig()
    model_config = ModelConfig()
    
    # Initialize Tokenizer & Processor
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.text_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = transformers.AutoProcessor.from_pretrained(model_config.audio_model_id)
    
    # Initialize Model
    model = MultiModalModel(model_config)
    
    # Apply LoRA if requested
    if train_config.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=train_config.lora_r, 
            lora_alpha=train_config.lora_alpha, 
            lora_dropout=train_config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
        
    # Dataset
    train_dataset = AudioTextDataset(train_config, processor, model_config, tokenizer)
    data_collator = DataCollator(processor, tokenizer)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.accum_steps,
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.num_epochs,
        max_steps=train_config.max_steps,
        logging_steps=train_config.log_steps,
        save_steps=train_config.save_steps,
        eval_strategy="no", # change if val set provided
        remove_unused_columns=False, # Important because we have custom forward signature
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)
    processor.save_pretrained(train_config.output_dir)

if __name__ == "__main__":
    train()
