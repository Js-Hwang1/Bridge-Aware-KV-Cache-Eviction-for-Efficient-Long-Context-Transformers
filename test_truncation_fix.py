"""
Validate Context Truncation Fix
================================

Quick test to verify question/instructions are preserved after truncation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

print("="*60)
print("Context Truncation Fix Validation")
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

# Load runner
print("\n2. Loading runner...")
from experiments.longbench_qa.runner import LongBenchQARunner
from experiments.longbench_qa.config import ExperimentConfig

config = ExperimentConfig(
    name="Qwen/Qwen2.5-7B-Instruct",
    max_length=4096,
    max_new_tokens=256,
)
runner = LongBenchQARunner(config)

# Load one NarrativeQA sample
print("\n3. Loading NarrativeQA sample...")
try:
    with open('/root/FRC/data/narrativeqa.jsonl', 'r') as f:
        sample = json.loads(f.readline())
    print(f"   ✓ Loaded sample")
except:
    print(f"   Creating synthetic long sample...")
    sample = {
        'context': "The story begins. " * 10000 + " The Mad Hatter and the March Hare hosted a tea party.",
        'question': "Who hosted the tea party?",
        'answers': ["Mad Hatter and March Hare"]
    }

context = sample['context']
question = sample['question']
answers = sample.get('answers', sample.get('answer', []))

print(f"   Context length: {len(context)} chars")
print(f"   Question: {question}")
print(f"   Expected: {answers}")

# Test the truncation logic
print("\n4. Testing _format_prompt with truncation...")
max_context_tokens = 3700  # Conservative limit
prompt = runner._format_prompt(context, question, max_context_tokens=max_context_tokens)

# Check if question is in prompt
if question in prompt:
    print(f"   ✓ Question PRESERVED in truncated prompt")
else:
    print(f"   ✗ ERROR: Question was removed!")
    exit(1)

# Check if assistant tag is there (for Qwen)
if "<|im_start|>assistant" in prompt:
    print(f"   ✓ Assistant tag present")
else:
    print(f"   ✗ WARNING: Assistant tag missing")

# Tokenize and check length
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
print(f"   Prompt tokens: {len(prompt_tokens)}")

if len(prompt_tokens) <= 3840:
    print(f"   ✓ Within limit (3840 tokens)")
else:
    print(f"   ⚠️  Slightly over limit: {len(prompt_tokens) - 3840} tokens")

# Generate with the runner
print("\n5. Generating response...")
prediction, diagnostics = runner.generate_response(
    context=context,
    question=question,
    max_new_tokens=256,
    sparse_method="dense",
)

print(f"   Generated: '{prediction}'")
print(f"   Generation time: {diagnostics['generation_time_ms']:.1f} ms")

# Check if prediction makes sense
print("\n6. Validation:")
print(f"   Expected: {answers}")
print(f"   Got: '{prediction}'")

# Check for common failure modes
if len(prediction) < 5:
    print(f"   ⚠️  Prediction too short ({len(prediction)} chars)")
elif prediction.startswith((",", ".", " ")):
    print(f"   ⚠️  Prediction starts with punctuation (context fragment?)")
elif any(ans_word.lower() in prediction.lower() for ans in answers for ans_word in ans.split()):
    print(f"   ✓ Prediction contains expected answer words")
else:
    print(f"   ⚠️  Prediction may not contain expected answer")

# More detailed check
if context.startswith(prediction[:20]):
    print(f"   ✗ ERROR: Prediction is copying context verbatim!")
else:
    print(f"   ✓ Prediction is not direct context copy")

print("\n" + "="*60)
print("Validation complete!")
print("="*60)
