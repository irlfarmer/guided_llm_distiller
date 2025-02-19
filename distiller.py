import torch._dynamo
# Suppress errors from TorchDynamo (such as missing Triton)
torch._dynamo.config.suppress_errors = True

import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Set high precision for float32 matmul (to silence TF32 warnings)
torch.set_float32_matmul_precision("high")

# ---------------- SETUP LOGGING ----------------
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ---------------- STEP 1: LOAD TEACHER MODEL ----------------
teacher_model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# Use GPTQ configuration; use_exllama=False (replacing the deprecated disable_exllama)
quant_config = GPTQConfig(bits=4, use_exllama=False)

tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# Optionally try compiling the model for faster inference;
# if compilation fails, fall back to eager mode.
try:
    teacher_model = torch.compile(teacher_model)
except Exception as e:
    print("torch.compile() failed; proceeding without compilation. Error:", e)

print("✅ Teacher Model Loaded Successfully.")
if not torch.cuda.is_available():
    raise RuntimeError("🚨 CUDA is not available! Check your PyTorch installation.")
print(f"✅ CUDA Available: {torch.cuda.is_available()} (CUDA Version: {torch.version.cuda})")

# ---------------- STEP 2: LOAD JSONL DATA ----------------
jsonl_file = "fine_tune_data.jsonl"
data = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))
print(f"✅ Loaded {len(data)} Q&A pairs.")

# ---------------- STEP 3: GENERATE TEACHER RESPONSES ----------------
teacher_outputs = []
batch_size = 8  # Adjust based on available VRAM
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    questions = [item["messages"][1]["content"] for item in batch]
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        output_tokens = teacher_model.generate(**inputs, max_length=100, do_sample=True)
    batch_teacher_answers = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    for j, item in enumerate(batch):
        teacher_outputs.append({
            "question": item["messages"][1]["content"],
            "original_answer": item["messages"][2]["content"],
            "teacher_answer": batch_teacher_answers[j]
        })
    if i % (10 * batch_size) == 0:
        print(f"✅ Processed {i}/{len(data)} questions for teacher generation...")
print("✅ Teacher Responses Collected.")

teacher_output_file = "distilled_dataset.jsonl"
with open(teacher_output_file, "w", encoding="utf-8") as f:
    for sample in teacher_outputs:
        f.write(json.dumps(sample) + "\n")
print(f"✅ Teacher-generated dataset saved as: {teacher_output_file}")

# ---------------- STEP 4: LOAD STUDENT MODEL ----------------
student_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).train()
print("✅ Student Model Loaded.")

# ---------------- STEP 5: DEFINE ROBUST DISTILLATION LOSS ----------------
class RobustDistillationLoss(nn.Module):
    """
    Computes KL divergence over the answer portion of the sequence.
    Uses temperature scaling and adds a small epsilon to avoid division by zero.
    """
    def __init__(self, temperature=2.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.kl_loss = nn.KLDivLoss(reduction="sum")

    def forward(self, student_logits, teacher_logits, mask):
        student_log_probs = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        teacher_probs = teacher_probs.clamp(min=self.eps)
        kl = self.kl_loss(student_log_probs, teacher_probs)
        denom = mask.sum() + self.eps
        return kl / denom

loss_fn = RobustDistillationLoss(temperature=2.0)
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
print("✅ Robust Distillation Loss Initialized.")

# ---------------- STEP 6: TRAINING LOOP ----------------
def train_distillation(epochs=3):
    student_model.train()
    teacher_model.eval()
    total_tokens = 0
    total_loss = 0.0
    for epoch in range(epochs):
        for i in range(0, len(teacher_outputs), batch_size):
            batch = teacher_outputs[i:i + batch_size]
            combined_texts = []
            prompt_texts = []
            for sample in batch:
                q = sample["question"]
                t_ans = sample["teacher_answer"]
                combined_texts.append(f"Question: {q}\nAnswer: {t_ans}")
                prompt_texts.append(f"Question: {q}\nAnswer:")
            combined_enc = student_tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            prompt_enc = student_tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            prompt_lengths = prompt_enc.input_ids.ne(student_tokenizer.pad_token_id).sum(dim=1)
            with torch.no_grad():
                teacher_logits_full = teacher_model(**combined_enc).logits
            student_logits_full = student_model(**combined_enc).logits

            batch_loss = 0.0
            batch_tokens = 0
            for j in range(len(batch)):
                p_len = prompt_lengths[j].item()
                if p_len >= student_logits_full[j].shape[0]:
                    continue
                teacher_slice = teacher_logits_full[j, p_len:, :]
                student_slice = student_logits_full[j, p_len:, :]
                mask = torch.ones(student_slice.shape[0], device=student_slice.device)
                sample_loss = loss_fn(student_slice, teacher_slice, mask)
                batch_loss += sample_loss
                batch_tokens += student_slice.shape[0]
            if batch_tokens == 0:
                continue
            loss = batch_loss / (batch_tokens + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            if i % (10 * batch_size) == 0:
                print(f"Epoch {epoch+1}, batch {i//batch_size}: loss per token: {loss.item():.4f}")
        avg_loss = total_loss / (total_tokens + 1e-8)
        print(f"Epoch {epoch+1} complete: average loss per token: {avg_loss:.4f}")

train_distillation(epochs=3)
student_model.save_pretrained("./distilled_model")
student_tokenizer.save_pretrained("./distilled_model")
print("✅ Distilled Model Saved Successfully!")
