import torch
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default="text à prédire", help="Texte à prédire")
parser.add_argument("--top_k", type=int, default=10, help="Le nombre de tokens les plus probables à garder")
parser.add_argument("--top_p", type=float, default=0.2, help="Compris entre 0.0 et 1.0 On prend les tokens les plus probable")
parser.add_argument("--temperature", type=float, default=1.0, help="Si c'est < 1.0 les tokens seront plus prédictif sinon ça sera plus créatif")
parser.add_argument("--max_tokens", type=int, default=10, help="Le nombre maximum de tokens à générer")

args = parser.parse_args()
model = torch.load("checkpoint.pt", weights_only=False)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer(args.text, return_tensors="pt")

print("Prédiction pour : ", args.text)
print(model.generate(inputs['input_ids'])[0])
print(tokenizer.decode(model.generate(inputs['input_ids'],
                                      top_k=args.top_k,
                                      top_p=args.top_p,
                                      temperature=args.temperature,
                                      max_new_tokens=args.max_tokens
                                      )[0], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True))
