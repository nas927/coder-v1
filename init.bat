pip install -r requirements.txt
python dataset_process.py
python tokenizer.py
.\launch.bat
python convert_to_hf.py
python hf_infer.py