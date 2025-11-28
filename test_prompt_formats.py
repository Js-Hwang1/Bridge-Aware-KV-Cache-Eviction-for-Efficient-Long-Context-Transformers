"""
Compare Prompt Formats for Qwen2.5
====================================

Test if manual chat template vs tokenizer.apply_chat_template produce different outputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*60)
print("Prompt Format Comparison")
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

# Sample data
context = "The story of Alice in Wonderland begins when Alice follows a white rabbit down a rabbit hole. She finds herself in a strange world full of peculiar creatures. She meets the Cheshire Cat, who can disappear and reappear at will. Later, she attends a mad tea party hosted by the Mad Hatter and the March Hare."
question = "Who hosted the mad tea party?"
expected = "Mad Hatter and March Hare"

print(f"\n2. Context: {context[:100]}...")
print(f"   Question: {question}")
print(f"   Expected: {expected}")

# Format 1: Manual construction (current runner.py)
print("\n3. Testing Format 1: Manual Construction (runner.py)")
prompt1 = f"""<|im_start|>system
You are a helpful assistant. Answer questions concisely based on the given context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Answer with only the answer, nothing else.<|im_end|>
<|im_start|>assistant
"""

print(f"   Prompt length: {len(prompt1)} chars")
print(f"   First 200 chars: {repr(prompt1[:200])}")

inputs1 = tokenizer(prompt1, return_tensors="pt", truncation=True, max_length=3840).to("cuda")
print(f"   Tokenized length: {inputs1['input_ids'].shape[1]} tokens")

with torch.no_grad():
    outputs1 = model.generate(
        **inputs1,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids1 = outputs1[0][inputs1['input_ids'].shape[1]:]
generated1 = tokenizer.decode(generated_ids1, skip_special_tokens=True)
print(f"   Generated: '{generated1}'")
print(f"   Generated tokens: {len(generated_ids1)}")

# Format 2: Using tokenizer.apply_chat_template (test script)
print("\n4. Testing Format 2: apply_chat_template (test script)")
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer questions concisely based on the given context."},
    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer with only the answer, nothing else."}
]
prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"   Prompt length: {len(prompt2)} chars")
print(f"   First 200 chars: {repr(prompt2[:200])}")

inputs2 = tokenizer(prompt2, return_tensors="pt", truncation=True, max_length=3840).to("cuda")
print(f"   Tokenized length: {inputs2['input_ids'].shape[1]} tokens")

with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids2 = outputs2[0][inputs2['input_ids'].shape[1]:]
generated2 = tokenizer.decode(generated_ids2, skip_special_tokens=True)
print(f"   Generated: '{generated2}'")
print(f"   Generated tokens: {len(generated_ids2)}")

# Compare
print("\n5. Comparison:")
print(f"   Manual format output:        '{generated1[:100]}'")
print(f"   Chat template output:        '{generated2[:100]}'")
print(f"   Manual tokens generated:     {len(generated_ids1)}")
print(f"   Chat template tokens:        {len(generated_ids2)}")

# Check which contains expected answer
gen1_match = expected.lower() in generated1.lower() or any(word in generated1.lower() for word in expected.lower().split())
gen2_match = expected.lower() in generated2.lower() or any(word in generated2.lower() for word in expected.lower().split())

print(f"   Manual format has answer:    {gen1_match}")
print(f"   Chat template has answer:    {gen2_match}")

# Show raw generated token IDs
print("\n6. Generated Token IDs:")
print(f"   Manual format: {generated_ids1[:10].tolist()}")
print(f"   Chat template: {generated_ids2[:10].tolist()}")

# Check if EOS was generated
if tokenizer.eos_token_id in generated_ids1[:5]:
    print(f"   ⚠️ Manual format generated EOS at position {(generated_ids1 == tokenizer.eos_token_id).nonzero()[0].item()}")
if tokenizer.eos_token_id in generated_ids2[:5]:
    print(f"   ⚠️ Chat template generated EOS at position {(generated_ids2 == tokenizer.eos_token_id).nonzero()[0].item()}")

print("\n" + "="*60)
print("Analysis complete")
print("="*60)
