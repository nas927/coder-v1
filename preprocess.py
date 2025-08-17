from transformers import AutoTokenizer
from datasets import load_dataset
import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFKC", text)

def split_data(batch):
    return {"lines": [doc.split("\n") for doc in batch["text"]]}

def check_data(dataset):
    for data in dataset:
        print(type(data))
        print(data)

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
    # GPT-2 qui utilise BPE
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = ["<fim_start>", "<fim_hole>", "<fim_end>", "<eos_token>", "<PAD>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.pad_token = "<PAD>"

    return tokenizer

def encode_data(tokenizer, dataset):
    input_output = split_input_prediction(dataset)
    input_output_tokenized = {}
    inp = []
    out = []

    for in_out in input_output:
        inp.append(str(in_out["input"]))
        out.append(str(in_out["output"]))
    
    all_strings = inp + out
    max_length = len(max(all_strings, key=len))
    input_output_tokenized["input"] = tokenizer(out, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)
    input_output_tokenized["output"] = tokenizer(inp, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)

    return input_output_tokenized

def decode_data(tokenizer, tokens_ids):
    decode = tokenizer.decode(tokens_ids)
    return decode

def ret_batch(input_ids, batch_size=3):
    batches = []

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size]  # récupère 3 séquences
        batches.append(batch)

    return batches
