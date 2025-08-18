# python 3.10 minimum is required

```sh
git clone https://github.com/nas927/coder-v1.git
cd coder-v1

python3 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

# To train the model 

Place all your data in datasets folder<br>
The extension must be **.txt**<bt>
Each line should contain 1 data

## First

```sh
python3 dataset_process.py
```

That will put all your data formated in a file named **all-in-one.txt**

after that, you can launch :

```sh
python3 train.py
```

You can change epochs in the file or in the command **--epochs 100**

# To infer

Open the file inference.py and change the text

```sh
python3 inference.py
```

output will be the prediction

You can now define the text and **top_k**, **top_p**, **temperature** and **max_tokens** to generate
- --text you "Your text"
- --top_k 20
- --top_p 1.0
- --temperature 1.0
- --max_tokens 100

You can use arg --help to display some help

```sh
python3 inference.py --help
```
