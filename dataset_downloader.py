import os, gc
from typing import Any, Generator
from datasets import load_dataset, IterableDataset

OUTDIR = "datasets"


def stream_dataset(dataset: IterableDataset, limit: int) -> Generator[dict[str, str], Any, None]:
    if limit < 0:
        for example in dataset:
            yield example
        return

    for i, example in enumerate(dataset):
        if i > limit:
            break
        yield example


def download_dataset(
    name: str,
    path: str,
    dataset_name: str,
    column: str = 'text',
    limit: int = -1
) -> None:
    dataset: IterableDataset = load_dataset(
        path,
        name=dataset_name,
        split='train',
        streaming=True,
    )

    try:
        if column not in dataset.column_names:
            raise KeyError(f"The dataset doesn't have a column named '{column}'")

        if not os.path.exists(OUTDIR):
            os.mkdir(OUTDIR)

        with open(os.path.join(OUTDIR, name + ".txt"), 'w', encoding='utf-8') as f:
            for example in stream_dataset(dataset, limit):
                text = example[column].strip().replace('\n', "\\n")
                if text:  # Only write non-empty text
                    f.write(text + '\n')

    except Exception as e:
        print(f"{name} : {type(e).__name__} : {e}")
    finally:
        gc.collect()  # DON'T REMOVE, else python segfault


if __name__ == "__main__":
    # download_dataset(
    #     "CommonCrawl_french",
    #     "BramVanroy/CommonCrawl-CreativeCommons",
    #     "CC-MAIN-2019-30-fra",
    #     limit=100
    # )
    download_dataset("sib200_french", "mteb/sib200", "fra_Latn", limit=100)
