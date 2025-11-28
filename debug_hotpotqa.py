"""
Debug HotPotQA Performance Issues
==================================

Test with one sample to diagnose:
1. Context truncation
2. Prompt format
3. Generation quality
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

print("="*60)
print("HotPotQA Debug")
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

# Load HotPotQA from LongBench
print("\n2. Loading HotPotQA data...")
try:
    # Try local file first
    with open('/root/FRC/experiments/longbench_qa/data/hotpotqa/longbench_raw/hotpotqa.jsonl', 'r') as f:
        sample = json.loads(f.readline())
except Exception as e:
    print(f"ERROR: Could not load HotPotQA data: {e}")
    sys.exit(1)

context = sample['context']
question = sample['input']  # LongBench uses 'input' for question
answer = sample['answers'][0] if isinstance(sample['answers'], list) else sample['answers']

print(f"   Context length: {len(context)} chars")
print(f"   Question: {question}")
print(f"   Expected answer: {answer}")
print(f"   Context preview (first 200 chars): {context[:200]}...")
print(f"   Context preview (last 200 chars): ...{context[-200:]}")

# Test prompt construction
print("\n3. Testing prompt construction...")
max_length = 32768
max_new_tokens = 256

# Construct Qwen prompt
prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions concisely based on the given context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Answer with only the answer, nothing else.<|im_end|>
<|im_start|>assistant
"""

print(f"   Full prompt length: {len(prompt)} chars")

# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors='pt',
    truncation=True,
    max_length=max_length - max_new_tokens,
)
print(f"   Tokenized length: {inputs['input_ids'].shape[1]} tokens")

# Decode to see what model actually sees
truncated_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
print(f"\n4. Checking truncated prompt...")
print(f"   Truncated prompt length: {len(truncated_prompt)} chars")

# Check if question is still there
if question in truncated_prompt:
    print(f"   ✓ Question is present")
else:
    print(f"   ✗ WARNING: Question was truncated!")
    print(f"   Last 300 chars of truncated prompt:")
    print(f"   ...{truncated_prompt[-300:]}")

# Check if assistant tag is there
if "<|im_start|>assistant" in truncated_prompt:
    print(f"   ✓ Assistant tag present")
else:
    print(f"   ✗ WARNING: Assistant tag missing!")

# Generate
print(f"\n5. Generating answer...")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

import time
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
gen_time = time.time() - start

generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"   Generated: '{generated}'")
print(f"   Generated tokens: {len(generated_ids)}")
print(f"   Generation time: {gen_time:.2f}s")

# Analysis
print(f"\n6. Analysis:")
print(f"   Expected: '{answer}'")
print(f"   Got: '{generated}'")

if len(generated) < 10:
    print(f"   ⚠️ Very short prediction")
if answer.lower() in generated.lower():
    print(f"   ✓ Contains expected answer")
else:
    print(f"   ✗ Does not contain expected answer")

# Check for common issues
if "the the" in generated:
    print(f"   ⚠️ Repetitive text detected (prompt corruption?)")
if generated.startswith((",", ".", " ", ")", "]")):
    print(f"   ⚠️ Starts with punctuation (context fragment?)")

print("\n" + "="*60)
