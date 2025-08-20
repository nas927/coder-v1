import os
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# Créer le tokenizer BPE
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token="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")

# Préparer les fichiers
folder = "./datasets/"
files = [os.path.join(folder, f) for f in os.listdir(folder)]

# Ajouter les tokens spéciaux FIM
special_tokens = {
    'additional_special_tokens': [
        '<fim_start>',  # Début de la séquence FIM
        '<fim_hole>',   # Trou à remplir
        '<fim_end>',    # Fin de la séquence FIM
    ]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = "<unk>"
tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="<s> <fim_start> $A </s>",
    pair="<s> <fim_start> $A </s> <s> <fim_start> $B </s>",
    special_tokens=[
        ("<fim_start>", tokenizer.convert_tokens_to_ids("<fim_start>")),
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
        ("</s>", tokenizer.convert_tokens_to_ids("</s>"))
    ]
)

print("Vocab size : ", len(tokenizer))

# Sauvegarder
tokenizer.save_pretrained("huggingf_compatible")