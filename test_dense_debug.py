"""
Debug Dense Baseline Generation
================================

Inspect actual model outputs to diagnose why F1=0.0191 (should be ~0.20-0.30).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

print("="*60)
print("Dense Baseline Debug")
print("="*60)

# Load model
print("\n1. Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
model.eval()

print(f"   Model loaded on: {model.device}")

# Load one sample from narrativeqa
print("\n2. Loading NarrativeQA sample...")
try:
    with open('/root/FRC/data/narrativeqa.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    print(f"   ✓ Sample loaded")
except Exception as e:
    print(f"   Creating synthetic sample (dataset not found)")
    sample = {
        'context': "The story of Alice in Wonderland begins when Alice follows a white rabbit down a rabbit hole. She finds herself in a strange world full of peculiar creatures. She meets the Cheshire Cat, who can disappear and reappear at will. Later, she attends a mad tea party hosted by the Mad Hatter and the March Hare.",
        'question': "Who hosted the mad tea party?",
        'answers': ["Mad Hatter and March Hare", "the Mad Hatter", "Mad Hatter"]
    }

context = sample['context'][:1000]  # Truncate for testing
question = sample['question']
answers = sample.get('answers', sample.get('answer', ['unknown']))

print(f"   Context length: {len(context)} chars")
print(f"   Question: {question}")
print(f"   Expected answers: {answers}")

# Test different prompt formats
print("\n3. Testing prompt formats...")

# Format 1: Qwen chat template
print("\n--- Format 1: Qwen Chat Template ---")
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer the question based on the context provided. Output ONLY the answer, nothing else."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
]
prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Prompt preview: {prompt1[:300]}...")

inputs1 = tokenizer(prompt1, return_tensors="pt", truncation=True, max_length=3000).to("cuda")
print(f"Input tokens: {inputs1['input_ids'].shape[1]}")

with torch.no_grad():
    outputs1 = model.generate(
        **inputs1,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only the new tokens
generated_ids = outputs1[0][inputs1['input_ids'].shape[1]:]
generated1 = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated: '{generated1}'")

# Format 2: Simple instruct format
print("\n--- Format 2: Simple Instruct Format ---")
prompt2 = f"""### Instruction:
Answer the question based on the context. Output ONLY the answer.

### Context:
{context}

### Question:
{question}

### Answer:"""

inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=3000).to("cuda")
print(f"Input tokens: {inputs2['input_ids'].shape[1]}")

with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs2[0][inputs2['input_ids'].shape[1]:]
generated2 = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated: '{generated2}'")

# Format 3: Minimal format (what runner.py uses)
print("\n--- Format 3: Minimal Format (Current Runner) ---")
prompt3 = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

inputs3 = tokenizer(prompt3, return_tensors="pt", truncation=True, max_length=3000).to("cuda")
print(f"Input tokens: {inputs3['input_ids'].shape[1]}")

with torch.no_grad():
    outputs3 = model.generate(
        **inputs3,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs3[0][inputs3['input_ids'].shape[1]:]
generated3 = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Generated: '{generated3}'")

# Check if any format produced reasonable output
print("\n4. Analysis:")
print(f"   Expected (any of): {answers}")
print(f"   Format 1: {generated1[:100]}")
print(f"   Format 2: {generated2[:100]}")
print(f"   Format 3: {generated3[:100]}")

# Simple overlap check
for i, gen in enumerate([generated1, generated2, generated3], 1):
    gen_lower = gen.lower()
    match_found = any(ans.lower() in gen_lower for ans in answers)
    print(f"   Format {i} contains answer: {match_found}")

print("\n" + "="*60)
print("Debug complete. Check which format works best.")
print("="*60)
