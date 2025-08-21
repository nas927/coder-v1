import os
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing
from colorama import init, Fore, Style, Back

init(autoreset=True)

OUTPUT_FOLER = "huggingf_compatible"

def tokenize():
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
            '<table_tokens>',
            '</table_tokens>'
        ]
    }

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = "<unk>"

    # Maintenant récupérer tous les IDs
    special_tokens_list = [
        ("<s>", tokenizer.convert_tokens_to_ids("<s>")),
        ("</s>", tokenizer.convert_tokens_to_ids("</s>"))
    ]

    print(special_tokens_list)
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=special_tokens_list
    )

    len_tokenizer = len(tokenizer)
    print(Fore.GREEN + "Vocab size : ", len_tokenizer)

    # Sauvegarder
    print(Back.WHITE + Fore.BLACK + f"Sauvagarde tokenizer" + Style.RESET_ALL)
    if not os.path.exists(OUTPUT_FOLER):
        os.makedirs(OUTPUT_FOLER)
    tokenizer.save_pretrained(OUTPUT_FOLER)

    return len_tokenizer

if __name__ == "__main__":
    tokenize()