from transformers import AutoTokenizer
from convert_to_hf import CustomCoderModel 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default="Aujourd’hui, les chercheurs en intelligence artificielle ", help="Texte à prédire")
parser.add_argument("--top_k", type=int, default=10, help="Le nombre de tokens les plus probables à garder")
parser.add_argument("--top_p", type=float, default=0.9, help="Compris entre 0.0 et 1.0 On prend les tokens les plus probable jusqu'à ce que leur sommes de probabilité arrive à top_p")
parser.add_argument("--temperature", type=float, default=1.0, help="Si c'est < 1.0 les tokens seront plus prédictif sinon ça sera plus créatif")
parser.add_argument("--max_tokens", type=int, default=2, help="Le nombre maximum de tokens à générer")
args = parser.parse_args()

Huggin_face_MODEL = "./huggingf/"
# chemin vers ton tokenizer.json

model = CustomCoderModel.from_pretrained(Huggin_face_MODEL)
tokenizer = AutoTokenizer.from_pretrained(Huggin_face_MODEL)
encoded = tokenizer(args.text, return_tensors="pt")
output = model.generate(encoded['input_ids'],
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        max_new_tokens=args.max_tokens
)

decoded = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(decoded)