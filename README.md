# python 3.12 minimum is required

git clone https://github.com/nas927/coder-v1.git

cd coder-v1
pip install -r requirements.txt

# To train the model 

Place all your data in datasets folder
The extension must be txt
Each line should contain 1 data

## First

python dataset_process.py

That will put all your data formated in a file named all-in-one.txt

after that, you can launch :
python train.py
You can change epochs in the file

# To infer

Open the file inference.py and change the text
python inference.py
output will be the prediction