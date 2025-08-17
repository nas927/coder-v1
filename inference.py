import torch
from transformers import AutoTokenizer
import argparse

model = torch.load("checkpoint.pt", weights_only=False)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("hey salut", return_tensors="pt")

print(model.generate(inputs['input_ids'])[0])
print(tokenizer.decode(model.generate(inputs['input_ids'])[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
