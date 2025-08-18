from datasets import load_dataset, DatasetDict
import random
import os

FOLDER_DATASETS = "./datasets/"
OUTPUT_FILE = "all-in-one.txt"


def split_data(batch: DatasetDict) -> dict[str, list[str]]:
    # batch['text'] est une liste de documents
    return {'lines': [doc.split('\n') for doc in batch['text']]}


# 2. Transformer en FIM
def make_fim_examples(lines: list[str]) -> None:
    for index, line in enumerate(lines):
        splitted_line = line.split(' ')
        len_splitted_Line = len(splitted_line)
        if len_splitted_Line >= 4:
            middle_start = random.randint(1, len_splitted_Line-4)
            middle_end = random.randint(middle_start+2, len_splitted_Line-2)
            fim_hole = splitted_line[middle_start:middle_end]
            fim_hole = ' '.join(fim_hole)
            # remplace le text a remplace par <fim_hole>
            splitted_line[middle_start:middle_end] = ["<fim_hole>"]
        else:
            fim_hole = ""
        formatted_line = "<fim_start>" + ' '.join(splitted_line) + "<fim_end>" + fim_hole + "<eos_token>"
        lines[index] = formatted_line


def transform_dataset() -> None:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        listDirDataset: list[str] = os.listdir(FOLDER_DATASETS)
        for file in listDirDataset:
            dataset: DatasetDict = load_dataset(
                "text",
                data_files={'train': os.path.join(FOLDER_DATASETS, file)},
                sample_by='document'
            )
            dataset = dataset.map(split_data, batched=True)

            dataset_lines: list[str] = dataset['train']['lines']
            dataset_lines = [line.strip() for line in dataset_lines[0] if line.strip() != ""]
            make_fim_examples(dataset_lines)

            for row in dataset_lines:
                f.write(row + '\n')


if __name__ == "__main__":
    random.seed(42)
    transform_dataset()
