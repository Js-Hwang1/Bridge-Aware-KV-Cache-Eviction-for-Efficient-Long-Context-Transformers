"""
Test Long Context Truncation
==============================

Reproduce the dense baseline issue with actual long contexts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

print("="*60)
print("Long Context Truncation Test")
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

# Load actual NarrativeQA sample
print("\n2. Loading NarrativeQA sample...")
try:
    with open('/root/FRC/data/narrativeqa.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    print(f"   ✓ Loaded real sample")
except Exception as e:
    print(f"   ✗ Failed to load: {e}")
    exit(1)

context = sample['context']
question = sample['question']
answers = sample.get('answers', sample.get('answer', []))

print(f"   Context length: {len(context)} chars")
print(f"   Question: {question}")
print(f"   Expected answers: {answers}")

# Test with same parameters as runner.py
max_length = 4096
max_new_tokens = 256
model_name = "Qwen/Qwen2.5-7B-Instruct"

# Construct prompt exactly like runner.py
prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions concisely based on the given context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Answer with only the answer, nothing else.<|im_end|>
<|im_start|>assistant
"""

print(f"\n3. Prompt construction:")
print(f"   Full prompt length: {len(prompt)} chars")
print(f"   Truncation limit: {max_length - max_new_tokens} tokens = {3840} tokens")

# Tokenize with truncation (like runner.py)
inputs = tokenizer(
    prompt,
    return_tensors='pt',
    truncation=True,
    max_length=max_length - max_new_tokens,
)
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print(f"   Actual tokenized length: {inputs['input_ids'].shape[1]} tokens")

# Decode the truncated prompt to see what the model actually sees
truncated_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
print(f"\n4. Truncated prompt preview:")
print(f"   Length: {len(truncated_prompt)} chars")
print(f"   Last 500 chars:")
print("   " + "-"*56)
print(f"   ...{truncated_prompt[-500:]}")
print("   " + "-"*56)

# Check if question is still in the truncated prompt
if question in truncated_prompt:
    print(f"   ✓ Question is present in truncated prompt")
else:
    print(f"   ✗ WARNING: Question was truncated away!")

# Check if the assistant prompt is there
if "<|im_start|>assistant" in truncated_prompt:
    print(f"   ✓ Assistant tag is present")
else:
    print(f"   ✗ WARNING: Assistant tag was truncated!")

# Generate
print(f"\n5. Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"   Generated: '{generated}'")
print(f"   Generated tokens: {len(generated_ids)}")

# Check if answer makes sense
print(f"\n6. Analysis:")
print(f"   Expected: {answers}")
print(f"   Generated: {generated[:100]}")

# Check if generated text is from context (exact match)
if generated.strip() in context:
    print(f"   ⚠️ Generated text is COPIED FROM CONTEXT (exact match)")
    # Find where in context
    pos = context.find(generated.strip())
    if pos != -1:
        print(f"   Found at position {pos} in context")
        print(f"   Context around that position: ...{context[max(0,pos-50):pos+len(generated)+50]}...")
else:
    print(f"   ✓ Generated text is not direct copy from context")

print("\n" + "="*60)
print("Test complete")
print("="*60)
