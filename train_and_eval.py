import os
import re
import sqlite3
import torch
import evaluate
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qwen-sql-lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE.upper()}")

# 2. Load & Format Dataset
print("Loading dataset...")
raw_dataset = load_dataset("b-mc2/sql-create-context", split="train")
raw_dataset = raw_dataset.shuffle(seed=42)

# Use 600 train / 30 test samples so it runs fast locally
train_raw = raw_dataset.select(range(600))
test_raw = raw_dataset.select(range(600, 630))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_prompt(schema, question):
    return (
        "<|im_start|>system\n"
        "You are an expert SQL generator. Convert the question into a valid SQL query using the provided table schema. Output ONLY the SQL query.<|im_end|>\n"
        f"<|im_start|>user\nSchema:\n{schema}\n\nQuestion:\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def format_for_training(example):
    prompt = build_prompt(example["context"], example["question"])
    full_text = f"{prompt}{example['answer']}<|im_end|>"
    return {"text": full_text}

train_dataset = train_raw.map(format_for_training)

# 3. Load Base Model
print("Loading base model...")
dtype = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(DEVICE)

def generate_sql(active_model, schema, question):
    prompt = build_prompt(schema, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = active_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    # Clean markdown code fences if the model adds them
    response = re.sub(r"```sql|```", "", response).strip()
    return response

# 4. Baseline Evaluation (Before Fine-Tuning)
print("Running baseline evaluation on 30 test queries...")
base_preds = []
references = []
for row in test_raw:
    pred = generate_sql(model, row["context"], row["question"])
    base_preds.append(pred)
    references.append(row["answer"].strip())

# 5. Configure LoRA Adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Fine-Tune with SFTTrainer
print("Starting LoRA fine-tuning...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=512,
    per_device_train_batch_size=4 if DEVICE == "cuda" else 2,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=(DEVICE == "cuda"),
    save_strategy="no",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=training_args,
)
trainer.train()

# Save LoRA adapter weights
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved fine-tuned adapter to {OUTPUT_DIR}")

# 7. Plot Training Loss Curve (For your PPT slides)
steps = [log["step"] for log in trainer.state.log_history if "loss" in log]
losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o", linewidth=2, color="#2563eb")
plt.title("QLoRA/LoRA Training Loss Curve (Qwen2.5-0.5B Text-to-SQL)")
plt.xlabel("Training Steps")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("training_loss.png", dpi=300)
print("Saved training_loss.png")

# 8. Fine-Tuned Evaluation & Comparison
print("Running fine-tuned evaluation...")
model.eval()
ft_preds = []
for row in test_raw:
    pred = generate_sql(model, row["context"], row["question"])
    ft_preds.append(pred)

rouge = evaluate.load("rouge")

def compute_metrics(preds, refs, schemas):
    rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"] * 100
    exact_matches = sum(1 for p, r in zip(preds, refs) if p.lower().rstrip(";") == r.lower().rstrip(";"))
    exact_acc = (exact_matches / len(refs)) * 100

    valid_sql = 0
    for p, schema in zip(preds, schemas):
        try:
            conn = sqlite3.connect(":memory:")
            conn.executescript(schema)
            conn.execute(f"EXPLAIN {p}")
            valid_sql += 1
            conn.close()
        except Exception:
            pass
    syntax_rate = (valid_sql / len(refs)) * 100
    return rouge_score, syntax_rate, exact_acc

schemas = [row["context"] for row in test_raw]
base_rouge, base_syn, base_em = compute_metrics(base_preds, references, schemas)
ft_rouge, ft_syn, ft_em = compute_metrics(ft_preds, references, schemas)

print("\n" + "=" * 60)
print(f"{'METRIC':<25} | {'BASE MODEL':<14} | {'FINE-TUNED':<14}")
print("-" * 60)
print(f"{'ROUGE-L Score':<25} | {base_rouge:>13.2f}% | {ft_rouge:>13.2f}%")
print(f"{'SQLite Syntax Validity':<25} | {base_syn:>13.2f}% | {ft_syn:>13.2f}%")
print(f"{'Exact Match Accuracy':<25} | {base_em:>13.2f}% | {ft_em:>13.2f}%")
print("=" * 60)

print("\n--- SAMPLE QUALITATIVE COMPARISONS ---")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(f"Question    : {test_raw[i]['question']}")
    print(f"Ground Truth: {references[i]}")
    print(f"Base Model  : {base_preds[i]}")
    print(f"Fine-Tuned  : {ft_preds[i]}")