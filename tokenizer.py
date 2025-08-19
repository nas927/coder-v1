import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Créer le tokenizer BPE
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Entraîner
trainer = BpeTrainer(vocab_size=32000, 
                    special_tokens=["<fim_start>", "<fim_hole>", "<fim_end>", "<eos_token>", "<PAD>"])

# Sur tes fichiers texte
folder = "./datasets/"
files = os.listdir(folder)
for file in files:
    files[files.index(file)] = folder + file
tokenizer.train(files, trainer)

# Sauvegarder
tokenizer.save("coder-v1.json")