"""Minimal debug script to understand attention extraction."""

import torch
from transformers import GPT2Model, GPT2Tokenizer

print("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Loading model...")
# Use 'eager' attention to enable attention output
model = GPT2Model.from_pretrained("gpt2", attn_implementation='eager').cuda()
model.eval()

print(f"Model config:")
print(f"  - output_attentions: {model.config.output_attentions}")
print(f"  - output_hidden_states: {model.config.output_hidden_states}")

text = "Hello world"
tokens = tokenizer.encode(text, return_tensors='pt').cuda()

print(f"\nRunning forward pass...")
with torch.no_grad():
    outputs = model(tokens, output_attentions=True)

print(f"\nOutputs type: {type(outputs)}")
print(f"Outputs keys: {outputs.keys() if hasattr(outputs, 'keys') else 'N/A'}")
print(f"Has attentions attr: {hasattr(outputs, 'attentions')}")

if hasattr(outputs, 'attentions'):
    print(f"Attentions: {outputs.attentions}")
    if outputs.attentions is not None:
        print(f"Attentions length: {len(outputs.attentions)}")
        print(f"First attention: {outputs.attentions[0]}")
