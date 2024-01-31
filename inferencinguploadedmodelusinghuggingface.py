from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("Taphu/chatbot_mistral7b")

inputs = tokenizer("### Question: Question I want to ask", return_tensors="pt").to("cuda")

model = AutoPeftModelForCausalLM.from_pretrained(
    "Taphu/chatbot_mistral7b",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id
)
import time
st_time = time.time()
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(time.time()-st_time)
