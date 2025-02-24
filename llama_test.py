import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random
import torch
import os
from datasets import load_dataset
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from peft import LoraConfig
import bitsandbytes as bnb
import os
import re

device_string = "cuda:0"

def get_latest_folder(path):
    # 获取路径下所有文件夹的名称
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # 根据文件夹的创建时间进行排序
    folders.sort(key=lambda x: os.path.getctime(os.path.join(path, x)))

    # 获取最新创建的文件夹名
    latest_folder = folders[-1] if folders else None

    return latest_folder

latest_model_name = get_latest_folder("./lbsn_model")
model_id = f"./lbsn_model/{latest_model_name}"

access_token = "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"
system_message = "你是一个有用的POI检索助手，你可以根据POI的描述和GeoHash排序。"

def generate_test_data():
    train_data = np.load("./llm_finetune_test_data.npy", allow_pickle=True).tolist()
    
    dataset = {"messages": [], "pos_id": []}
    for data in tqdm(train_data):        
        # candidate_list = [f"{index}: {record['address']}" for index, record in enumerate(data["records"])]
        candidate_list = [f"id: {index} text: {record['address']} geohash: {record['geohash']}" for index, record in enumerate(data["records"])]
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"\n 查询：{data['query']} \n 候选列表：{candidate_list}"},
        ]
        
        dataset["messages"].append(messages)
        dataset["pos_id"].append(data["pos_index"])
    dataset = Dataset.from_dict(dataset)
    return dataset

test_dataset = generate_test_data()

tokenizer = AutoTokenizer.from_pretrained(
    model_id, token=access_token,
    padding_side="left", add_eos_token=True, add_bos_token=True, 
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    # device_map={"": device_string}, 
    device_map = "auto",
    token=access_token,
    attn_implementation="sdpa"
)
model.eval()
# model = torch.compile(model)

def extract_numbers(text):
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    return numbers

K_list = [1, 3, 5, 10]
acc_K = {K: 0 for K in K_list}
total = len(test_dataset)
in_count = 0
batch_size = 1
with torch.no_grad():
    for i in tqdm(range(0, total, batch_size)):
        data = test_dataset[i: i + batch_size]
        messages, pos_id = data["messages"], data["pos_id"]
        inputs = tokenizer([tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_length=8192)
        outputs_text = tokenizer.batch_decode(outputs)
        print(outputs_text)

# for i, data in enumerate(tqdm(test_dataset)):
#     messages, pos_id = data["messages"], data["pos_id"]
#     if pos_id != -1:
#         in_count += 1
#         prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device_string)
#         outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id)
#         outputs_text = tokenizer.decode(outputs[0])
#         outputs_target = outputs_text[len(prompts): -len("<|eot_id|>")]
#         rank_result = [int(num) for num in extract_numbers(outputs_target)]
#         for K in K_list:
#             if pos_id in rank_result[:K]:
#                 acc_K[K] += 1
    
#     print(i, acc_K, in_count)