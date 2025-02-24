import torch
# 导入依赖包
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig



model = AutoModelForCausalLM.from_pretrained("/mnt/HDD/syj/model/glm-4-9b", device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
model
tokenizer = AutoTokenizer.from_pretrained('/mnt/HDD/syj/model/glm-4-9b', use_fast=False, trust_remote_code=True)
# 将tokenizer的pad_token设置为eos_token，这样在进行填充时，会使用eos_token作为填充符号
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token)

