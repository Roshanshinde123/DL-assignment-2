import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import sqlite3
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "./qwen-sql-lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading QueryCraft AI on {DEVICE.upper()}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)
ft_model.eval()

def run_inference(model_obj, schema, question):
    prompt = (
        "<|im_start|>system\n"
        "You are an expert SQL generator. Convert the question into a valid SQL query using the provided table schema. Output ONLY the SQL query.<|im_end|>\n"
        f"<|im_start|>user\nSchema:\n{schema}\n\nQuestion:\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model_obj.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r"```sql|```", "", text).strip()

def check_sqlite_syntax(schema, sql_query):
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.execute(f"EXPLAIN {sql_query}")
        conn.close()
        return "✅ Valid SQLite Syntax & Schema Match"
    except Exception as e:
        return f"❌ Syntax / Schema Error: {str(e)}"

def compare_models(schema, question):
    if not schema.strip() or not question.strip():
        return "Please enter a valid CREATE TABLE schema.", "Please enter a question.", "⚠️ Missing Input"
    
    # Toggle adapter OFF for raw Base Model output
    with ft_model.disable_adapter():
        base_sql = run_inference(ft_model, schema, question)
    
    # Toggle adapter ON for Fine-Tuned LoRA output
    ft_sql = run_inference(ft_model, schema, question)
    
    # Validate Fine-Tuned SQL against schema in SQLite
    validation_status = check_sqlite_syntax(schema, ft_sql)
    return base_sql, ft_sql, validation_status

custom_css = """
body, .gradio-container { background-color: #0b0f19 !important; color: #f8fafc !important; }
.header-title { color: #38bdf8; font-size: 28px; font-weight: 800; margin-bottom: 2px; }
.header-sub { color: #94a3b8; font-size: 14px; margin-bottom: 16px; }
.sidebar-card { background: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
.sidebar-card-active { background: #0f172a; border: 1px solid #38bdf8; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
"""

with gr.Blocks(css=custom_css, title="QueryCraft AI - Text-to-SQL") as demo:
    gr.HTML("""
        <div>
            <div class="header-title">⚡ QueryCraft <span style="color:#ffffff;">AI</span></div>
            <div class="header-sub">Fine-Tuned Qwen2.5-0.5B LoRA Text-to-SQL Assistant • Schema-Grounded Query Generation & SQLite Validation</div>
        </div>
    """)
    
    with gr.Row():
        # Left Workspace Sidebar
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="color:#38bdf8; font-weight:700; font-size:13px; letter-spacing:1px; margin-bottom:10px;">✦ SQL WORKSPACE</div>
                <div class="sidebar-card-active">
                    <div style="color:#38bdf8; font-weight:700;">01 Schema Input</div>
                    <div style="color:#94a3b8; font-size:12px;">Define your CREATE TABLE database schema.</div>
                </div>
                <div class="sidebar-card">
                    <div style="color:#f8fafc; font-weight:700;">02 Query Generator</div>
                    <div style="color:#94a3b8; font-size:12px;">Translate natural language questions into SQL.</div>
                </div>
                <div class="sidebar-card">
                    <div style="color:#f8fafc; font-weight:700;">03 Model Comparison</div>
                    <div style="color:#94a3b8; font-size:12px;">Compare Base LLM vs. LoRA Fine-Tuned output.</div>
                </div>
                <div class="sidebar-card">
                    <div style="color:#f8fafc; font-weight:700;">04 SQLite Verifier</div>
                    <div style="color:#94a3b8; font-size:12px;">In-memory EXPLAIN syntax & column verification.</div>
                </div>
            """)
        
        # Main Interactive Workspace
        with gr.Column(scale=3):
            schema_box = gr.Textbox(
                lines=4,
                label="01 Database Table Schema (CREATE TABLE DDL)",
                value="CREATE TABLE students (roll_no INT, name VARCHAR, branch VARCHAR, cgpa FLOAT);"
            )
            question_box = gr.Textbox(
                lines=2,
                label="02 Natural Language Question",
                value="List the names and CGPA of students in Computer Engineering with CGPA above 8.5."
            )
            
            with gr.Row():
                generate_btn = gr.Button("▶ Generate & Compare SQL", variant="primary")
                clear_btn = gr.ClearButton([schema_box, question_box], value="↺ Clear")
            
            with gr.Row():
                base_output = gr.Textbox(lines=4, label="Base Model Output (Qwen2.5-0.5B Before Fine-Tuning)")
                ft_output = gr.Textbox(lines=4, label="Fine-Tuned LoRA Output (QueryCraft AI)")
            
            status_box = gr.Textbox(lines=1, label="04 Automated SQLite Syntax & Schema Validation")
            
            gr.Examples(
                examples=[
                    [
                        "CREATE TABLE students (roll_no INT, name VARCHAR, branch VARCHAR, cgpa FLOAT);",
                        "List the names and CGPA of students in Computer Engineering with CGPA above 8.5."
                    ],
                    [
                        "CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT, status VARCHAR);",
                        "What is the total order amount for completed orders?"
                    ],
                    [
                        "CREATE TABLE festival_detail (Festival_Name VARCHAR, YEAR INT, Location VARCHAR);",
                        "What are the names of festivals held in year 2007?"
                    ]
                ],
                inputs=[schema_box, question_box],
                label="Quick Start • Click an Example"
            )

    generate_btn.click(
        fn=compare_models,
        inputs=[schema_box, question_box],
        outputs=[base_output, ft_output, status_box]
    )

if __name__ == "__main__":
    demo.launch()