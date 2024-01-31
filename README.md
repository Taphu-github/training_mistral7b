There are three files there
1. for finetuning the Mistral7b
2. for inferencing the model which is uploaded to the huggingface
3. for inferencing the saved model as gguff format

to convert the model to gguf format:
1. load the model on higher bits than 8-bits, such as 16 or 32 bits.
2. Finetune the model using lora adapters.
3. Save the model and adapter both
4. convert the model to gguf using the following command
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert.py vicuna-hf \
  --outfile vicuna-13b-v1.5.gguf \
  --outtype q8_0

