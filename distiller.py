import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# ---------------- SETUP LOGGING TO HIDE WARNINGS ----------------
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ---------------- STEP 1: LOAD TEACHER MODEL (GPTQ-QUANTIZED) ----------------
teacher_model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

# Fix Exllama warning & use proper quantization config
quant_config = GPTQConfig(bits=4, use_exllama=False)

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Fix padding issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    quantization_config=quant_config,  # Correct GPTQ loading
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# Compile model for faster inference
teacher_model = torch.compile(teacher_model)

print("✅ Teacher Model Loaded Successfully.")

# ---------------- CHECK CUDA AVAILABILITY ----------------
if not torch.cuda.is_available():
    raise RuntimeError("🚨 CUDA is not available! Check your PyTorch installation.")

print(f"✅ CUDA Available: {torch.cuda.is_available()}")

# ---------------- STEP 2: LOAD JSONL DATA ----------------
jsonl_file = "fine_tune_data.jsonl"
data = []

with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"✅ Loaded {len(data)} Q&A pairs.")

# ---------------- STEP 3: GENERATE TEACHER RESPONSES (WITH DEBUG PRINTS) ----------------
teacher_outputs = []
batch_size = 8  # Process 8 questions at a time

for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]  # Get a batch of questions
    questions = [item["messages"][1]["content"] for item in batch]
    
    # Tokenize the batch
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        output_tokens = teacher_model.generate(**inputs, max_length=100, do_sample=True)

    # Decode batch responses
    batch_answers = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    # Save responses
    for j, item in enumerate(batch):
        teacher_outputs.append({
            "question": item["messages"][1]["content"],
            "original_answer": item["messages"][2]["content"],
            "teacher_answer": batch_answers[j]
        })

    # Print progress every 10 batches
    if i % (10 * batch_size) == 0:
        print(f"✅ Processed {i}/{len(data)} questions...")

print("✅ Teacher Responses Collected.")

# ---------------- STEP 4: SAVE TEACHER OUTPUTS AS JSONL ----------------
teacher_output_file = "distilled_dataset.jsonl"
with open(teacher_output_file, "w", encoding="utf-8") as f:
    for item in teacher_outputs:
        f.write(json.dumps(item) + "\n")

print(f"✅ Teacher-generated dataset saved as: {teacher_output_file}")

# ---------------- STEP 5: LOAD STUDENT MODEL (TINYLLAMA) ----------------
student_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# Fix padding for student model tokenizer
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).train()

print("✅ Student Model Loaded.")

# ---------------- STEP 6: DEFINE KNOWLEDGE DISTILLATION LOSS ----------------
class DistillationLoss(nn.Module):
    """KL Divergence Loss for Knowledge Distillation"""
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, original_logits):
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        original_probs = torch.nn.functional.softmax(original_logits / self.temperature, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

        loss_teacher = self.kl_loss(student_log_probs, teacher_probs)
        loss_original = self.kl_loss(student_log_probs, original_probs)

        return (loss_teacher + loss_original) * (self.temperature ** 2)

loss_fn = DistillationLoss(temperature=2.0)
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)

print("✅ Custom Distillation Loss Initialized.")

# ---------------- STEP 7: TRAINING LOOP (WITH DEBUG PRINTS) ----------------
def train_distillation(epochs=3):
    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        total_loss = 0

        for i in range(0, len(teacher_outputs), batch_size):
            batch = teacher_outputs[i:i + batch_size]

            # Tokenize batch
            questions = [item["question"] for item in batch]
            original_answers = [item["original_answer"] for item in batch]
            teacher_answers = [item["teacher_answer"] for item in batch]

            inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to("cuda")
            original_inputs = tokenizer(original_answers, return_tensors="pt", padding=True, truncation=True).to("cuda")
            teacher_inputs = tokenizer(teacher_answers, return_tensors="pt", padding=True, truncation=True).to("cuda")

            # Get logits
            with torch.no_grad():
                teacher_logits = teacher_model(**teacher_inputs).logits
                original_logits = teacher_model(**original_inputs).logits  

            student_logits = student_model(**inputs).logits

            # Compute loss
            loss = loss_fn(student_logits, teacher_logits, original_logits)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 10 batches
            if i % (10 * batch_size) == 0:
                print(f"🔄 Epoch {epoch+1}, Processed {i}/{len(teacher_outputs)} questions - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(teacher_outputs)
        print(f"✅ Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")

train_distillation(epochs=3)

# ---------------- STEP 8: SAVE TRAINED STUDENT MODEL ----------------
student_model.save_pretrained("./distilled_model")
student_tokenizer.save_pretrained("./distilled_model")

print("✅ Distilled Model Saved Successfully!")
