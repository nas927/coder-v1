import torch
from tokenizers import Tokenizer
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default="Aujourd’hui, les chercheurs en intelligence artificielle ", help="Texte à prédire")
parser.add_argument("--top_k", type=int, default=10, help="Le nombre de tokens les plus probables à garder")
parser.add_argument("--top_p", type=float, default=0.9, help="Compris entre 0.0 et 1.0 On prend les tokens les plus probable jusqu'à ce que leur sommes de probabilité arrive à top_p")
parser.add_argument("--temperature", type=float, default=1.0, help="Si c'est < 1.0 les tokens seront plus prédictif sinon ça sera plus créatif")
parser.add_argument("--max_tokens", type=int, default=1, help="Le nombre maximum de tokens à générer")
args = parser.parse_args()

model = torch.load("checkpoint.pt", weights_only=False)
model.to(device)

tokenizer = Tokenizer.from_file("./coder-v1.json")
inputs = tokenizer.encode(args.text)
inputs = torch.tensor(inputs.ids).unsqueeze(0)

print("Prédiction pour : ", args.text)
tokens_generated = model.generate(inputs,
                                      top_k=args.top_k,
                                      top_p=args.top_p,
                                      temperature=args.temperature,
                                      max_new_tokens=args.max_tokens
                                )
tokenized_text = tokenizer.decode(tokens_generated[0].tolist())
print(tokenized_text)
