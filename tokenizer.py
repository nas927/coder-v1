import os
from transformers import AutoTokenizer

# Créer le tokenizer BPE
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token="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")

# Préparer les fichiers
folder = "./datasets/"
files = [os.path.join(folder, f) for f in os.listdir(folder)]

print("Vocab size : ", len(tokenizer))

# Sauvegarder
tokenizer.save_pretrained("huggingf_compatible")