import os, gc
from typing import Any, Generator
from datasets import load_dataset, IterableDataset

OUTDIR = "datasets"


def stream_dataset(dataset: IterableDataset, limit: int) -> Generator[dict, Any, None]:
    for i, example in enumerate(dataset):
        if i > limit:
            break
        yield example


def download_CommonCrawl_french() -> None:
    # Load the French subset using streaming
    dataset: IterableDataset = load_dataset(
        "BramVanroy/CommonCrawl-CreativeCommons",
        name='CC-MAIN-2019-30-fra',
        split='train',
        streaming=True,
    )

    with open(os.path.join(OUTDIR, "CommonCrawl_french.txt"), 'w', encoding='utf-8') as f:
        try:
            for example in stream_dataset(dataset, 100):
                text: str = example.get('text', "")
                text = text.strip().replace('\n', "\\n")
                if text:  # Only write non-empty text
                    f.write(text + '\n')

        except Exception as e:
            print(f"CommonCrawl_french : Error : {e}")
        finally:
            gc.collect()  # DONT REMOVE, else python segfault


if __name__ == "__main__":
    download_CommonCrawl_french()
