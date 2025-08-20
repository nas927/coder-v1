from transformers import AutoTokenizer
from datasets import load_dataset
import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFKC", text)

def split_data(batch):
    return {"lines": [doc.split("\n") for doc in batch["text"]]}

def split_input_prediction(dataset):
    fim_end_token = "<fim_end>"
    eos_token = "<eos_token>"
    input_output = []

    for line in dataset:
        idx = line.index(fim_end_token) + len(fim_end_token)  # inclure fim_end
        inp = line[:idx]   # tout avant fim_end inclus
        out = line[idx:]  # tout après fim_end

        input_output.append({
            "input": inp,
            "output": out
        })

    return input_output

def check_data(dataset):
    for data in dataset:
        if not data:
            raise("error")
        if not isinstance(data, str):
            raise("error")

def load_data():
    dataset = load_dataset(            
        "text",
        data_files={"all-in-one.txt"},
        sample_by="document"
    )

    dataset = dataset.map(split_data, batched=True)
    dataset = dataset["train"]["lines"]
    dataset = [normalize_text(line.strip()) for line in dataset[0] if line.strip() != ""]
    return dataset

def tokenize():
    # mistral qui utilise BPE sur notre vocabulaire
    tokenizer = AutoTokenizer.from_pretrained("huggingf_compatible")

    return tokenizer

def encode_data(tokenizer, dataset):
    input_output = split_input_prediction(dataset)
    input_output_tokenized = {}
    inp = []
    out = []

    for in_out in input_output:
        inp.append(str(in_out["input"]))
        out.append(str(in_out["output"]))

    check_data(inp)
    check_data(out)

    if len(inp) != len(out):
        raise("Erreur les deux listes ne sont pas de même taille !") 
    all_strings = inp + out
    max_length = len(max(all_strings, key=len))
    # Configurer le padding
    input_output_tokenized["input"] = tokenizer(inp, padding='max_length', max_length=max_length, return_tensors="pt")
    input_output_tokenized["output"] = tokenizer(out, padding='max_length', max_length=max_length, return_tensors="pt")

    return input_output_tokenized

def decode_data(tokenizer, tokens_ids):
    decode = tokenizer.decode(tokens_ids)

    return decode

def ret_batch(input_output_tokenized, batch_size=3):
    batches = []
    inputs = input_output_tokenized["input"]
    outputs = input_output_tokenized["output"]
    
    for i in range(0, len(inputs), batch_size):
        batch = {
            "input": inputs[i:i + batch_size],
            "output": outputs[i:i + batch_size]
        }
        batches.append(batch)
    
    return batches