import json
import os
import random
import traceback
import pandas as pd
from typing import TypedDict, List, Dict, Optional

INDIR = "other_datasets"
TABLE_DATA_CSV_DIR = os.path.join(INDIR, "AppScript_dataset_csv")
OUTDIR = "datasets"


class ExampleDict(TypedDict):
    input_data: Optional[Dict[str, str]]
    prompts: List[str]
    table_data: Optional[list[str]]


class FunctionBlock(TypedDict):
    description: Optional[str]
    function_code: List[str]
    examples: List[ExampleDict]


AppScriptDataset = Dict[str, FunctionBlock]

######## Dataset format ########
{
  "FUNCTION_NAME": {
    "description": "description de FUNCTION_NAME",
    "function_code": [
      "function FUNCTION_NAME() {",
      "  console.log(\"{data1}\");",
      "  console.log(\"{data2}\");",
      "}"
    ],
    "examples": [
      {
        "input_data": {"data1": "valeur de data1", "data2": "valeur de data2"},  # Necessaire seulement si il y a des parametres specifique a ces prompts
        "table_data": ["people1"],  # Optional, names of the .csv file inside TABLE_DATA_CSV_DIR
        "prompts": [
          "Creer moi une function qui affiche \"valeur de data1\" et \"valeur de data2\".",
          "Affiche 'valeur de data1' et 'valeur de data2' dans la console."
        ]
      },
      {
        "input_data": {"data1": "some data", "data2": "other some data"},
        "prompts": [
          "Fais un console.log de 'some data' et 'other some data'."
        ]
      },
    ]
  }
}
################################


def validate_dataset_format(dataset: dict) -> None:
    if not isinstance(dataset, dict):
        raise ValueError("Le dataset doit être un dictionnaire racine.")

    for func_name, func_block in dataset.items():
        if not isinstance(func_block, dict):
            raise ValueError(f"Le bloc de fonction '{func_name}' doit être un dictionnaire.")

        if "description" in func_block and not isinstance(func_block["description"], str):
            raise ValueError(f"'description' de '{func_name}' doit être une string")

        if "function_code" not in func_block or not isinstance(func_block["function_code"], list):
            raise ValueError(f"'{func_name}' doit contenir une liste 'function_code'.")

        if not all(isinstance(line, str) for line in func_block["function_code"]):
            raise ValueError(f"'function_code' de '{func_name}' doit être une liste de string représentant la function.")

        if "examples" not in func_block or not isinstance(func_block["examples"], list):
            raise ValueError(f"'{func_name}' doit contenir une liste 'examples'.")

        for i, example in enumerate(func_block["examples"]):
            if not isinstance(example, dict):
                raise ValueError(f"Example #{i} de '{func_name}' doit être un dictionnaire.")

            if "input_data" in example and not isinstance(example["input_data"], dict):
                raise ValueError(f"'input_data' de example #{i} de '{func_name}' doit être un dictionnaire")

            if "table_data" in example and (
                    not isinstance(example["table_data"], list) or
                    not all(isinstance(line, str) for line in example["table_data"])
                    ):
                raise ValueError(f"'table_data' de example #{i} de '{func_name}' doit être une liste de string")

            if "prompts" not in example or (
                    not isinstance(example["prompts"], list) or
                    not all(isinstance(line, str) for line in example["prompts"])
                    ):
                raise ValueError(f"Example #{i} de '{func_name}' doit contenir une liste 'prompts'.")


def load_dataset() -> AppScriptDataset:
    with open("other_datasets/AppScript_dataset.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    validate_dataset_format(dataset)
    return dataset


def escape_curly_except_placeholders(code_lines: list[str], placeholders: list[str]) -> list[str]:
    escaped_lines: list[str] = []
    for line in code_lines:
        # Échappe tous les { et } en {{ et }} d'abord
        line_escaped = line.replace("{", "{{").replace("}", "}}")

        # Ensuite, remplace les {{placeholder}} par {placeholder}
        for ph in placeholders:
            line_escaped = line_escaped.replace(f"{{{{{ph}}}}}", f"{{{ph}}}")

        escaped_lines.append(line_escaped)
    return escaped_lines


def convert_to_fim(line: str) -> str:
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
    formatted_line = ' '.join(splitted_line) + "<fim_end>" + fim_hole
    return formatted_line


def get_a1_notation(row_idx: int, col_idx: int) -> str:
    """Convertit les indices en notation A1 (ex: A1, B2, etc.)"""
    col_letter = ''
    while col_idx >= 0:
        col_letter = chr(col_idx % 26 + ord('A')) + col_letter
        col_idx = col_idx // 26 - 1
    return f"{col_letter}{row_idx + 2}"


def load_table_data(name: str) -> str:
    df = pd.read_csv(os.path.join(TABLE_DATA_CSV_DIR, name + ".csv"))
    output: list[str] = []

    # Add header row
    header = ", ".join([f"{get_a1_notation(-1, i)}: {col}" for i, col in enumerate(df.columns)])
    output.append(header)

    # Add data rows
    for row_idx in range(len(df)):
        row_output: list[str] = []
        for col_idx in range(len(df.columns)):
            cell_value = df.iat[row_idx, col_idx]
            if pd.notna(cell_value) and str(cell_value).strip() != "":
                cell = get_a1_notation(row_idx, col_idx)
                row_output.append(f"{cell}: {cell_value}")
        if row_output:
            output.append(", ".join(row_output))

    data = "\\n".join(output)
    # print(data.replace("\\n", "\n"))
    return data


def AppScript_dataset_converter() -> None:
    try:
        dataset: AppScriptDataset = load_dataset()

        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

        cached_table_data: dict[str, str] = {}

        with open(os.path.join(OUTDIR, "AppScript_dataset.txt"), 'w', encoding='utf-8') as f:
            for i, entry_name in enumerate(dataset, start=1):
                print(f"{i} : {entry_name}")
                entry = dataset[entry_name]

                placeholders = entry['examples'][0].get('input_data', {}).keys()
                escaped_code_lines = escape_curly_except_placeholders(entry['function_code'], placeholders)
                raw_function_code = "\\n".join(escaped_code_lines)

                for example in entry['examples']:
                    function_code = raw_function_code.format(**example.get('input_data', {}))

                    table_datas: list[str] = []
                    for table_data_name in example.get('table_data', []):
                        if table_data_name not in cached_table_data:
                            cached_table_data[table_data_name] = load_table_data(table_data_name)
                        table_data = "<table_data>" + cached_table_data[table_data_name] + "</table_data>"
                        table_datas.append(table_data)

                    if len(table_datas) == 0:
                        table_datas.append("")

                    for table_data in table_datas:
                        for prompt in example['prompts']:
                            output = "<fim_start>" + prompt + table_data + convert_to_fim(function_code)
                            f.write(output + '\n')

    except Exception as e:
        print(f"AppScript_dataset_converter : {type(e).__name__} : {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    AppScript_dataset_converter()
