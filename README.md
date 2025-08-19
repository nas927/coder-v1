# python 3.12 minimum is required

git clone https://github.com/nas927/coder-v1.git

cd coder-v1
pip install -r requirements.txt

# To train the model 

Place all your data in datasets folder
The extension must be txt
Each line should contain 1 data

> # Download dataset

> Open file dataset_downloader.py
> Change the dataset_name and column name in function download_dataset
> Search the data in [huggingface](https://huggingface.co/datasets)
> Repeat that for each dataset you want to have 
> And then launch
> ```sh
> python dataset_downloader.py
> ```

> # Transform whole dataset in one file
> ```sh
> python dataset_process.py
> ```
> That will put all your data formated in a file named all-in-one.txt

> # Fit tokenizer
> ```sh
> python tokenizer.py
>

# Fit the model
```sh
python train.py
```
You can change epochs in the file or in the command
- --epochs 100

You can use arg --help to help you
```sh
python train.py --help
```

# To infer

Open the file inference.py and change the text
python inference.py
output will be the prediction

You can now define the text and top_k, top_p, temperature and max_tokens to generate
- --text you "Your text"
- --top_k 20
- --top_p 1.0
- --temperature 1.0
- --max_tokens 100

You can use arg --help to help you

```sh
python inference.py --help
```