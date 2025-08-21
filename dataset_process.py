from datasets import load_dataset
import random
import os

DATASET_DIR: str = './datasets/'
OTHER_DATASETS_DIR: str = './other_datasets/'

def split_data(batch):
    # batch['text'] est une liste de documents
    return {"lines": [doc.split("\n") for doc in batch["text"]]}

# 2. Transformer en FIM
def make_fim_examples(lines):
    for index, line in enumerate(lines):
        line: str = "<fim_start>" + line
        line: list[str] = line.split(' ')
        lenLine: int = len(line)
        if lenLine >= 4:
            middle_start: int = random.randint(1, lenLine-4)
            middle_end: int = random.randint(middle_start+2, lenLine-2)
            fim_hole: list[str] = line[middle_start:middle_end]
            fim_hole: str = ' '.join(fim_hole)
        else:
            fim_hole = ""
        lenLine = len(line)
        line[lenLine - 1] = line[lenLine - 1] + "<fim_end>" + fim_hole
        line = ' '.join(line)
        line = line.replace(fim_hole, "<fim_hole>", 1)
        lines[index] = line

# Take your files extension in other_datasets transform to txt 
# Columns is a list of column in json, jsonl or csv file specified
# each column while be in a row for each line of dataset
def convert_to_txt(columns: list[str], extension: str):
    folder: str = OTHER_DATASETS_DIR
    files: list[str] = os.listdir(folder)
    jsonl_files: list[str] = []
    lines_to_write: list[str] = []

    for file in files:
        if file.endswith(extension):
            jsonl_files.append(folder + file)

    dataset: dict = load_dataset(extension.replace('jsonl', 'json'), data_files=jsonl_files)

    for data in dataset["train"]:
        for column in columns:
            temp: str = ''
            if column in data:
                temp += data[column].strip().replace('\n', ' ')

        if temp:
            lines_to_write.append(temp.strip())
        temp = ''

    with open(DATASET_DIR + str(random.randint(10000, 99999)) + extension + ".txt", 'w', encoding='utf-8') as f:
        for row in lines_to_write:
            f.write(row + '\n')

    return lines_to_write

# Take your file name in other_datasets transform to txt 
# Columns is a list of column in json, jsonl or csv file specified
# each column while be in a row for each line of dataset
def convert_each_file(file, columns) -> None:
    extension = os.path.splitext(file)[1].replace('.', '')
    lines_to_write = []

    dataset = load_dataset(extension.replace("jsonl", "json"), data_files=(OTHER_DATASETS_DIR + file))
    for data in dataset["train"]:
        for column in columns:
            temp = ''
            if column in data:
                temp += data[column].strip().replace('\n', ' ')

        if temp:
            lines_to_write.append(temp.strip())
        temp = ''

        with open(DATASET_DIR + file + str(random.randint(10000, 99999)) + extension + ".txt", 'w', encoding='utf-8') as f:
            for row in lines_to_write:
                f.write(row + '\n')

def transform_dataset() -> None:
    folderDatasets: str = './datasets/'
    listDirDataset: list[str] = os.listdir(folderDatasets)
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
        with open('all-in-one.txt', 'a', encoding='utf-8') as f:
            for row in dataset:
                f.write(row + '\n')

# convert_to_txt(["French"], "csv")
# convert_to_txt(["English"], "csv")
# convert_to_txt(["prompt"], "csv")
transform_dataset()
#convert_each_file("humaneval-js.jsonl", ["prompt"])