from datasets import load_dataset
import random
import os

def split_data(batch):
    # batch['text'] est une liste de documents
    return {"lines": [doc.split("\n") for doc in batch["text"]]}

# 2. Transformer en FIM
def make_fim_examples(lines):
    for index, line in enumerate(lines):
        line = "<fim_start>" + line
        line = line.split(' ')
        lenLine = len(line)
        if lenLine >= 4:
            middle_start = random.randint(1, lenLine-4)
            middle_end = random.randint(middle_start+2, lenLine-2)
            fim_hole = line[middle_start:middle_end]
            fim_hole = ' '.join(fim_hole)
        else:
            fim_hole = ""
        lenLine = len(line)
        line[lenLine - 1] = line[lenLine - 1] + "<fim_end>" + fim_hole + "<eos_token>"
        line = ' '.join(line)
        line = line.replace(fim_hole, "<fim_hole>", 1)
        lines[index] = line

def transform_dataset():
    folderDatasets: str = './datasets/'
    listDirDataset = os.listdir(folderDatasets)
    for file in listDirDataset:
        dataset = load_dataset(
            "text",
            data_files={"train": folderDatasets + file},
            sample_by="document"
        )
        dataset = dataset.map(split_data, batched=True)
        dataset = dataset["train"]["lines"]
        dataset = [line.strip() for line in dataset[0] if line.strip() != ""]
        make_fim_examples(dataset)
        with open('all-in-one.txt', 'w', encoding='utf-8') as f:
            for row in dataset:
                f.write(row + '\n')
