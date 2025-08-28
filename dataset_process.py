from datasets import load_dataset, DatasetDict
import random
import os
from colorama import init, Style, Back, Fore
import dataset_downloader

init(autoreset=True)

DATASETS_DIR = "./datasets/"
OTHER_DATASETS_DIR = "./other_datasets/"
OUTPUT_FILE = "all-in-one.txt"
if not os.path.exists(DATASETS_DIR):
    os.mkdir(DATASETS_DIR)
if not os.path.exists(OTHER_DATASETS_DIR):
    os.mkdir(OTHER_DATASETS_DIR)


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
        formatted_line = "<fim_start>" + ' '.join(splitted_line) + "<fim_end>" + fim_hole
        lines[index] = formatted_line

    return lines

def convert_to_txt(columns: list[str], extension: str) -> None:
    """
    Take your files extension in other_datasets transform to txt
    Columns is a list of column in json, jsonl or csv file specified
    each column while be in a row for each line of dataset
    """
    if extension.count('.') <= 0:
        extension = '.' + extension

    try:
        files: list[str] = os.listdir(OTHER_DATASETS_DIR)
        jsonl_files: list[str] = []

        for file in files:
            if file.endswith(extension):
                jsonl_files.append(OTHER_DATASETS_DIR + file)

        if len(jsonl_files) <= 0:
            print(f"No files with extension '{extension}' found")
            return

        dataset = load_dataset(extension.replace("jsonl", "json"), data_files=jsonl_files)
        lines_to_write: list[str] = []

        for data in dataset["train"]:
            for column in columns:
                if column in data:
                    text = data[column].strip().replace('\n', "\\n")
                    if text:
                        lines_to_write.append(text.strip())

        if not os.path.exists(DATASETS_DIR):
            os.mkdir(DATASETS_DIR)

        counter = 1
        base_filepath = DATASETS_DIR + os.path.splitext(extension)[0].replace('.', '')
        output_filename = f"{base_filepath}_{counter:04}.txt"
        while os.path.exists(output_filename):
            output_filename = f"{base_filepath}_{counter:04}.txt"
            counter += 1

        with open(output_filename, 'w', encoding='utf-8') as f:
            for row in lines_to_write:
                f.write(row + '\n')

    except Exception as e:
        print(Fore.RED + f"convert_to_txt() : {type(e).__name__} : {e}")

def convert_each_file(file: str, columns: list[str] | bool, separator: str, with_fim: bool=False) -> None:
    """
    Take your file name in other_datasets transform to txt
    Columns is a list of column in json, jsonl or csv file specified
    each column while be in a row for each line of dataset
    """
    try:
        extension = os.path.splitext(file)[1].replace('.', '')
        lines_to_write: list[str] = []

        dataset = load_dataset(extension.replace("jsonl", "json"), data_files=(OTHER_DATASETS_DIR + file))
        if not columns:
            columns = dataset.column_names["train"]

        for data in dataset["train"]:
            text: str = ""
            for column in columns:
                if column in data:
                    sep: str = separator if text != "" else ""
                    text += sep + str(data[column]).strip().replace('\n', " ")

            if text != "":
                lines_to_write.append(text.strip())
            text = ""

        if os.path.exists(DATASETS_DIR + os.path.basename(file) + ".txt"):
            file = str(random.randint(1000, 9999)) + "_" + os.path.basename(file)

        if with_fim:
            lines_to_write = make_fim_examples(lines_to_write)
        with open(DATASETS_DIR + os.path.basename(file) + ".txt", 'w', encoding='utf-8') as f:
            for row in lines_to_write:
                f.write(row + '\n')

    except Exception as e:
        print(Fore.RED + f"convert_each_file() : {type(e).__name__} : {e}")

def transform_dataset(file: str) -> None:
    dataset_lines: list[str] = []
    output_file = OUTPUT_FILE if not file else file
    if os.path.exists(DATASETS_DIR + output_file):
        output_file = str(random.randint(1000, 9999)) + "_" + output_file
    print(Fore.GREEN + f"Tout mettre dans un seul fichier : {DATASETS_DIR + output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if not file:
                listDirDataset: list[str] = os.listdir(DATASETS_DIR)
                for file in listDirDataset:
                    dataset: DatasetDict = load_dataset(
                        "text",
                        data_files={'train': os.path.join(DATASETS_DIR, file)},
                        sample_by='document'
                    )
                    dataset = dataset.map(split_data, batched=True)
                    dataset_lines: list[str] = dataset['train']['lines']
                    dataset_lines = [line.strip() for line in dataset_lines[0] if line.strip() != ""]

            else:
                with(open(OTHER_DATASETS_DIR + file, 'r', encoding='utf-8') as file):
                    dataset_lines= [line.strip() for line in file if line.strip() != ""] 

            make_fim_examples(dataset_lines)
            for row in dataset_lines:
                f.write(row + '\n')

    except Exception as e:
        print(f"transform_dataset() : {type(e).__name__} : {e}")

def all_in_one() -> None:
    print(Fore.GREEN + f"Tout mettre dans un seul fichier : {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            listDirDataset: list[str] = os.listdir(DATASETS_DIR)
            for file in listDirDataset:
                f.write(open(DATASETS_DIR + file, 'r', encoding='utf-8').read())

    except Exception as e:
        print(f"all_in_one() : {type(e).__name__} : {e}")

if __name__ == "__main__":
    # convert_to_txt(["French"], "csv")
    # convert_to_txt(["English"], "csv")
    os.system("python AppScript_dataset_converter.py")
    convert_each_file("prog_language/humaneval-js.jsonl", ["task_id", "prompt"], ' : ')
    math_dataset = "maths/PAL-Math/datasets/"
    for dir in os.listdir(OTHER_DATASETS_DIR + math_dataset):
        for file in os.listdir(OTHER_DATASETS_DIR + math_dataset + dir):
            print(OTHER_DATASETS_DIR + math_dataset + dir + "/" +file)
            convert_each_file(math_dataset + dir + "/" + file , False, " : ")
    sib_french = dataset_downloader.download_dataset("sib200_french", "mteb/sib200", "fra_Latn")
    print(sib_french)
    transform_dataset(sib_french)
    all_in_one()
