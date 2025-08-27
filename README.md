# python 3.10 minimum is required

```sh
git clone https://github.com/nas927/coder-v1.git
cd coder-v1

python3 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

# To train the model 

Place all your data in datasets folder<br>
The extension must be txt<br>
Each line should contain 1 data

> # Download dataset
>
> Open file dataset_downloader.py<br>
> Change the path, dataset_name and column name in function download_dataset<br>
> Search the data in [huggingface](https://huggingface.co/datasets)<br>
> Repeat that for each dataset you want to have <br>
> And then launch
> ```sh
> python dataset_downloader.py
> ```

> # Transform whole dataset in one file
> Open file dataset_process.py<br>
> Uncomment transform_dataset only if your datasets is in the good format txt and all in datasets folder<br>
> else<br>
> Put your datasets in other_datasets folder if you are sure that every file with the same extension have same columns name then use convert_to_txt specify first arg is all the column you want to put in a line second arg is the extension name
> ```sh
> python dataset_process.py
> ```
> That will put all your data formated in a file named all-in-one.txt

> # Fit tokenizer
> ```sh
> python tokenizer.py
> ```

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

# Fit the model with lora

```sh
python train.py --lora 1 --lora-path model.pt
```
- --lora is between 0 and 1 for true
- --lora-only default 1 if you want to train only lora not all model with lora
- --lora-r for low rank default 8
- --lora-alpha alpha default 2 * low rank
- --lora-path is the path where you want to save lora defaut is ./best_model_lora.pt

You can use arg --help to help you

```sh
python train.py --help
```

# To infer

Open the file inference.py and change the text

```sh
python inference.py
```

Only for inference.py you can load lora 
```sh
python inference.py --lora path_model.pt
```
Among options you have
- --lora-path where you want to save your lora if lora-only is true

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

Or 

```sh
python hf_infer.py --help
```

For hugging face inference

# Convert the model to huggingface model 

First of all train the model and then when you have your pt file launch 

```sh
python convert_to_hf.py --help
```

# Convert to gguf

Check here to see how you can do that : https://github.com/ggml-org/llama.cpp/discussions/2948

# Use it in ollama

```sh
ollama serve
```

On an other terminal

```sh
cd ./your_folder/Modelfile
ollama create YourModelName
ollama run YourModelName
```

or Open the file in gguf folder
see the (llama cpp Makefile doc)[https://github.com/ollama/ollama/blob/main/docs/modelfile.md#parameter]
To see what you have to do
Open the file convert-to-gguf.bat 
- PATH_LLAMA_CPP location of llama.cpp
- PATH_HUGGING_COMPATIBLE location of huggingface converted model with convert_to_hf.py
- PATH_GGUF gguf folder in this folder
- MODEL_NAME if you change if change model name after "FROM" in your Modelfile

When everything is ok launch
```sh
./convert-to-gguf.bat
```

# Change d_model

- The model paratemeter have 1.7b for now you can change it
- Go to train.py and change d_model and d_ff.
- Or 
```sh
python train.py --d_model 512 --d_ff 2048
```
When converting to hf 
```sh
python convert_to_hf.py --d_model 512 --d_ff 2048
```
- d_ff should be 2.7 times bigger than d_model.
- d_model should be a multiple of num_heads as d_model % num_heads == 0