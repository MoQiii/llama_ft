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
from peft import LoraConfig, get_peft_model,TaskType
import bitsandbytes as bnb

# accelerate launch llama_finetune.py --num_processes 2

# train_data = np.load("./llm_finetune_train_data.npy", allow_pickle=True).tolist()
# test_data = np.load("./llm_finetune_test_data.npy", allow_pickle=True).tolist()

# local_rank = os.getenv("LOCAL_RANK")
# device_string = "cuda:" + str(local_rank)
# print("device_string", device_string)

# dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
# print()

# device_string = "cuda:0"

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "/mnt/HDD/syj/model/Llama-2-7b"
access_token = "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"
system_message = "你是一个有用的POI检索助手，你可以根据查询对候选POI的相似度进行排序。"

def generate_finetune_data():
    train_data = np.load("./llm_finetune_train_data.npy", allow_pickle=True).tolist()
    
    dataset = {"messages": []}
    for data in tqdm(train_data):
        records = [{"real_rank": index, "address": record["address"], "geohash": record["geohash"]} for index, record in enumerate(data["records"])]
        random.shuffle(records)
        rank_map = dict(sorted({record["real_rank"]: index for index, record in enumerate(records)}.items()))
        rank_list = list(rank_map.values())
        record_list = [[record["address"], record["geohash"]] for record in records]
        
        candidate_list = [f"id: {i} | address: {address} | geohash: {geohash}" for i, (address, geohash) in enumerate(record_list)]
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"请根据查询对候选POI的相似度进行排序。 \n 查询：{data['query']} \n 候选列表：{candidate_list}"},
            {"role": "assistant", "content": str(rank_list)},
        ]
        
        dataset["messages"].append(messages)
    dataset = Dataset.from_dict(dataset)
    return dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/HDD/syj/model/glm-4-9b", 
    device_map={'':torch.cuda.current_device()},
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config,  
    trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(
    '/mnt/HDD/syj/model/glm-4-9b',
    # use_fast=False, 
    # padding_side="left",
    # add_eos_token=True, 
    # add_bos_token=True, 
    trust_remote_code=True
    )

# tokenizer = AutoTokenizer.from_pretrained(
#     model_id, token=access_token,
#     padding_side="left", add_eos_token=True, add_bos_token=True, 
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     quantization_config=bnb_config, 
#     # device_map={"": device_string}, 
#     device_map = "auto",
#     token=access_token,
#     attn_implementation="sdpa"
# )

# lora_config = LoraConfig(
#     r=8,
#     target_modules = find_all_linear_names(model),
#     task_type="CAUSAL_LM",
# )
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    target_modules=["query_key_value"],
    inference_mode=False,  # 训练模式
    r=4,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

dataset = generate_finetune_data()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=2,
        max_steps=1,
        # num_train_epochs=10,
        learning_rate=5e-4,
        fp16=True,
        save_strategy="steps", # epoch
        save_steps=100,
        output_dir="/mnt/HDD/syj/syj_poi/poi_query/ft_model/lora/",
        save_total_limit=5,
        optim="paged_adamw_8bit",
        logging_dir="./logs/training.log",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        # packing=True,
        # max_seq_length=5120,
        # gradient_checkpointing_kwargs={'use_reentrant':False}
        # strategy="ddp_find_unused_parameters_false"
    ),
    # peft_config=lora_config,
    packing=True,
    max_seq_length=5120,
)

trainer.train(resume_from_checkpoint=False)

# 保存 LoRA 模型
trainer.save_model()

# 合并模型
merged_model = trainer.model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained(
    "/mnt/HDD/syj/syj_poi/poi_query/ft_model/meged_model",
    safe_serialization=True
)
tokenizer.save_pretrained("/mnt/HDD/syj/syj_poi/poi_query/ft_model/meged_model")
