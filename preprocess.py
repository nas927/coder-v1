from typing import Any
from transformers import AutoTokenizer
from datasets import load_dataset
import unicodedata

def normalize_text(text) -> str:
    return unicodedata.normalize("NFKC", text)

def split_data(batch) -> dict:
    return {"lines": [doc.split("\n") for doc in batch["text"]]}

def check_data(dataset) -> None:
    for data in dataset:
        if not data:
            raise("error")
        if not isinstance(data, str):
            raise("error")

def load_data(file: str = "all-in-one.txt") -> list[str]:
    dataset = load_dataset(            
        "text",
        data_files={file},
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

def encode_data(tokenizer, dataset: list[any], max_length: int = 0) -> dict:
    if max_length > 0:
        return tokenizer(dataset, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt", padding_side="right")
    else:
        return tokenizer(dataset, padding=True, return_tensors="pt", padding_side="right")

def decode_data(tokenizer, tokens_ids) -> str:
    decode = tokenizer.decode(tokens_ids)

    return decode

def ret_batch(tokenizer, dataset: dict, batch_size: int=3, max_length: int=0) -> list[list[Any]]:
    batches: list[list[Any]] = []
    datasetLength: int = len(dataset)

    print("La taille du dataset est de : ", datasetLength)
    
    for i in range(0, len(dataset), batch_size):
        batches.append(encode_data(tokenizer, dataset[i:i + batch_size], max_length=max_length))

    print(f"La taille du batch est de : {datasetLength}/{batch_size} = ", len(batches))
    
    return batches




# Pour tester
# encoded = encode_data(tokenize(), load_data())
# batches = ret_batch(encoded)
# print(batches[0][0])
# decoded = decode_data(tokenize(), batches[0][0])
# print(decoded)
# tokenizer = tokenize()
# dataset = load_data()
# batches = ret_batch(tokenizer, dataset)
# print(batches[0][2].ids)