#import pandas to work with tables
import pandas as pd
from datasets import load_dataset, Dataset
import json
from trl import SFTTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import transformers


df = pd.read_excel('new_data1.xlsx') 
data_json = df.to_json(orient="records", indent=0)
json_dataset = json.loads(data_json)#
training_data = []
for i in json_dataset:
    # Check if 'Prompt' or 'Response' is None and replace it with an empty string
    prompt = i.get('Prompt', '') or ''
    response = i.get('Response', '') or ''
    # Concatenate the strings
    prompt_str = "### Human: " + prompt
    response_str = "### Assistant: " + response
    training_data.append({"prompt": prompt_str, "response": response_str})

train_dataset = Dataset.from_list(training_data)
model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, quantization_config=bnb_config)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,  # Llama 2 7b, same as before# Same quantization config as before
    device_map={"":0},
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=128, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
def formatting_func(example):
    text = f"### Question: {example['prompt']}\n ### Answer: {example['response']}"
    return text

max_length=512
def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
def formatting_prompts_func(x):
    output_texts = []
    for i in range(len(x['prompt'])):
        text = f"### Question: {x['prompt'][i]}\n ### Answer: {x['response'][i]}"
        output_texts.append(text)
    return output_texts
#pip uninstall bitsandbytes

import transformers
from datetime import datetime
import bitsandbytes as bnb
project = "finetune-withLinux"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        optim = "paged_adamw_8bit",
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        fp16=False,
        logging_steps=25,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
# 'adamw_bnb_8bit', 'adamw_8bit', 'lion_8bit'

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


#Inferencing

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7b, same as before# Same quantization config as before
    device_map={"":0},
    trust_remote_code=True,
    quantization_config=bnb_config,
)
base_model.config.use_cache = True
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "mistral-finetune-withLinux/checkpoint-500")

eval_prompt = "### Question : List the divisions of GovTech Agency of Bhutan?"
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=512, repetition_penalty=1.15)[0], skip_special_tokens=True))



##saving the merged model
merged_model = ft_model.merge_and_unload()
merged_model.save_pretrained("model_name")

ft_model.push_to_hub("Taphu/chatbot_mistral7b")
eval_tokenizer.push_to_hub("Taphu/chatbot_mistral7b")

