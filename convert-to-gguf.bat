set PATH_LLAMA_CPP="C:\Users\dieum\Documents\programmation\fine-tune\test_gguf\llama.cpp"
set PATH_HUGGING_COMPATIBLE="C:\Users\dieum\Documents\programmation\gs_excel_project\llm\huggingf_compatible"
set PATH_GGUF="C:\Users\dieum\Documents\programmation\gs_excel_project\llm\gguf"
set MODEL_NAME="model.gguf"

mkdir gguf
cd %PATH_LLAMA_CPP%
REM --outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}
python convert_hf_to_gguf.py %PATH_HUGGING_COMPATIBLE% --outfile %PATH_GGUF%/%MODEL_NAME% --outtype q8_0
cd %PATH_GGUF%
start "" ollama serve
timeout /t 6 /nobreak
ollama create coder
ollama run coder

pause