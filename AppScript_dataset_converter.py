import json
import os
import random
import traceback
from typing import TypedDict, List, Dict, Optional

OUTDIR = "datasets"


class ExampleDict(TypedDict):
    input_data: Dict[str, str]
    prompts: List[str]
    table_data: Optional[str]


class FunctionBlock(TypedDict):
    description: str
    function_code: List[str]
    examples: List[ExampleDict]


AppScriptDataset = Dict[str, FunctionBlock]


def validate_dataset_format(dataset: dict) -> None:
    if not isinstance(dataset, dict):
        raise ValueError("Le dataset doit être un dictionnaire racine.")

    for func_name, func_block in dataset.items():
        if not isinstance(func_block, dict):
            raise ValueError(f"Le bloc de fonction '{func_name}' doit être un dictionnaire.")

        if "description" in func_block and not isinstance(func_block["description"], str):
            raise ValueError(f"'description' de '{func_name}' doit être une string s’il est présent.")

        if "function_code" not in func_block or not isinstance(func_block["function_code"], list):
            raise ValueError(f"'{func_name}' doit contenir une liste 'function_code'.")

        if not all(isinstance(line, str) for line in func_block["function_code"]):
            raise ValueError(f"'function_code' de '{func_name}' doit être une liste de string représentant la function.")

        if "examples" not in func_block or not isinstance(func_block["examples"], list):
            raise ValueError(f"'{func_name}' doit contenir une liste 'examples'.")

        for i, example in enumerate(func_block["examples"]):
            if not isinstance(example, dict):
                raise ValueError(f"Example #{i} de '{func_name}' doit être un dictionnaire.")

            if "input_data" not in example or not isinstance(example["input_data"], dict):
                raise ValueError(f"Example #{i} de '{func_name}' doit contenir 'input_data'.")

            if "prompts" not in example or not isinstance(example["prompts"], list):
                raise ValueError(f"Example #{i} de '{func_name}' doit contenir une liste 'prompts'.")

            if "table_data" in example and not isinstance(example["table_data"], str):
                raise ValueError(f"'table_data' de example #{i} de '{func_name}' doit être une string s’il est présent.")


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


def AppScript_dataset_converter() -> None:
    try:
        dataset: AppScriptDataset = load_dataset()

        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

        with open(os.path.join(OUTDIR, "AppScript_dataset.txt"), 'w', encoding='utf-8') as f:
            for i, entry_name in enumerate(dataset, start=1):
                entry = dataset[entry_name]

                placeholders = entry['examples'][0]['input_data'].keys()
                escaped_code_lines = escape_curly_except_placeholders(entry['function_code'], placeholders)
                raw_function_code = "\\n".join(escaped_code_lines)

                for example in entry['examples']:
                    function_code = raw_function_code.format(**example['input_data'])
                    table_data = "<table_data>" + example.get('table_data', "") + "</table_data>"

                    for prompt in example['prompts']:
                        output = "<fim_start>" + prompt + table_data + convert_to_fim(function_code)
                        f.write(output + '\n')

    except Exception as e:
        print(f"AppScript_dataset_converter : {type(e).__name__} : {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    AppScript_dataset_converter()
