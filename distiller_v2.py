import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        teacher_tokenizer,
        student_tokenizer,
        device,
        temperature=2.0,
        alpha=0.5  # Balance between soft and hard targets
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
    
    def compute_loss(self, student_logits, teacher_logits, labels):
        """
        Compute the distillation loss:
        - soft_loss: KL divergence between softened teacher and student logits
        - hard_loss: Cross-entropy between student logits and true labels
        """
        # Reshape logits and labels for loss computation
        # Move channel dim to end for KL-div
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        # Flatten labels for cross-entropy
        labels = labels.view(-1)
        
        # Soften the logits by dividing by temperature
        soft_teacher_logits = teacher_logits / self.temperature
        soft_student_logits = student_logits / self.temperature
        
        # Compute soft targets loss (KL divergence)
        soft_targets = F.softmax(soft_teacher_logits, dim=-1)
        soft_loss = F.kl_div(
            F.log_softmax(soft_student_logits, dim=-1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Compute hard targets loss
        # Ignore padding tokens (usually -100)
        mask = labels != -100
        hard_loss = F.cross_entropy(
            student_logits[mask],
            labels[mask],
            reduction='mean'
        )
        
        # Combine losses
        loss = (self.alpha * soft_loss) + ((1 - self.alpha) * hard_loss)
        return loss

    def train(self, train_dataloader, num_epochs, optimizer, scheduler=None):
        """Training loop with knowledge distillation"""
        self.student_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits
                
                # Get student logits
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits
                
                # Compute distillation loss
                loss = self.compute_loss(student_logits, teacher_logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1} average loss: {avg_loss:.4f}')

def load_and_prepare_data(file_path, tokenizer, max_length=512):
    """Load and process JSONL data"""
    questions, answers = [], []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            messages = data['messages']
            for i in range(len(messages)-1):
                if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                    questions.append(messages[i]['content'])
                    answers.append(messages[i+1]['content'])
    
    # Combine questions and answers
    combined_texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
    
    # Tokenize
    encodings = tokenizer(
        combined_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create a proper dataset with tensor values
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'].tolist(),
        'attention_mask': encodings['attention_mask'].tolist(),
        'labels': encodings['input_ids'].tolist()
    })
    
    # Add the tensor conversion function
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return dataset

def save_model(student_model, student_tokenizer, output_dir):
    """Save the model and tokenizer with proper configuration"""
    print(f"Saving model and tokenizer to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    student_model.save_pretrained(
        output_dir,
        save_function=torch.save,
        safe_serialization=True
    )
    
    # Save tokenizer with Llama-specific configuration
    student_tokenizer.save_pretrained(
        output_dir,
        legacy_format=False
    )
    
    # Save additional tokenizer configuration
    tokenizer_config = {
        "model_max_length": 512,
        "padding_side": "right",
        "truncation_side": "right",
        "pad_token": student_tokenizer.pad_token,
        "eos_token": student_tokenizer.eos_token,
        "bos_token": student_tokenizer.bos_token,
        "model_type": "llama",  # Updated to Llama
        "tokenizer_type": "meta-llama/Llama-2-7b-chat"  # Reference the base tokenizer
    }
    
    with open(f"{output_dir}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    
    # Save Llama-specific generation config
    generation_config = {
        "max_length": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.0,
        "repetition_penalty": 1.1
    }
    
    with open(f"{output_dir}/generation_config.json", "w") as f:
        json.dump(generation_config, f, ensure_ascii=False, indent=2)
    
    print("Model and configurations saved successfully!")

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
    
    try:
        # Model names - updated to Llama 2
        teacher_model_name = "meta-llama/Llama-2-7b"
        student_model_name = "meta-llama/Llama-2-7b-chat"  # or you could use a smaller version
        print(f"Loading teacher model: {teacher_model_name}")
        
        # Load models and tokenizers with Llama-specific configurations
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name,
            use_fast=False  # Llama uses slow tokenizer
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"Loading student model: {student_model_name}")
        student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name,
            use_fast=False
        )
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Ensure proper tokenizer settings for Llama
        for tokenizer in [teacher_tokenizer, student_tokenizer]:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        
        # Prepare dataset
        print("Loading and preparing dataset...")
        dataset = load_and_prepare_data(
            "fine_tune_data.jsonl",
            student_tokenizer
        )
        
        # Create dataloader with proper collation
        train_dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            pin_memory=True
        )
        
        print("Initializing trainer...")
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            device=device,
            temperature=2.0,
            alpha=0.5
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        
        print("Starting training...")
        trainer.train(
            train_dataloader=train_dataloader,
            num_epochs=3,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        print("Training completed. Saving model...")
        save_model(student_model, student_tokenizer, "./final_distilled_model")
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()