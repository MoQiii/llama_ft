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
from trl import SFTTrainer,DPOTrainer
from trl.trainer import ConstantLengthDataset
from peft import LoraConfig
import bitsandbytes as bnb
from unsloth import FastLanguageModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"



# model_id = "/mnt/HDD/syj/model/Llama-2-7b"
# model_id = "/mnt/HDD/TSC_xx/finetune_data/model/meta_llama3.1_8b"
model_id = "/mnt/HDD/Rensifei/Llama3-Chinese-8B-Instruct"
# /mnt/HDD/Rensifei/chatglm-6b
# model_id = "/mnt/HDD/syj/syj_poi/poi_query/ft_model2"
# access_token = "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"
# system_message = "你是一个聪明的POI检索助手，你可以根据查询对候选POI的相似度进行排序。"

alpaca_prompt = """你是一个聪明的POI检索助手，您可以根据查询和geohash对候选POI进行排序。

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# alpaca_prompt_cn = """你是一个聪明的POI检索助手，你可以根据查询对候选POI的相似度进行排序。

# Instruction:
# {}

# Input:
# {}

# Response: 
# {}"""

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)
 
#  query geohash  record_list address geohash

max_seq_length = 8192
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8,
    lora_dropout = 0.0001,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   #对于非常长的上下文，为True或unsloth
    random_state = 3407,           
    use_rslora = False,
    loftq_config = None,
)

def formatting_prompts_func(examples):
    querys = examples["query"]
    geohashs = examples["geohash"]
    record_lists = examples["records"]
    texts=[]
    for query, geohash, record_list in zip(querys, geohashs, record_lists): 
        # records的原始记录为真实的排名
        records = [{"real_rank": index, "address": record["address"], "geohash": record["geohash"]} 
                   for index, record in enumerate(record_list)]
        # 随机打乱
        random.shuffle(records)
        rank_list = []
        candidate_list = []
        # 随机打乱的records给上ID作为打乱的序号传给LLM，返回的为ID的列表
        for i, record in enumerate(records):
            rank_list.append(record["real_rank"])
            candidate_list.append(f"id: {i} | address: {record['address']} | geohash: {record['geohash']}")
        # 构造prompt
        instruction = f"基于地址和geohash对候选兴趣点列表进行排序，返回候选列表的数字ID。下面总共有一条查询和其对应的geohash。此外，还有你需要排序的100条地址和geohash对，对应的ID从0到99。你需要返回排序后的ID，把你认为和查询更相似的地址的ID排到更前面。"
        inputs = f"查询是: {query},geohash是: {geohash} ,候选poi列表是: {candidate_list}"
        output = "返回的对应的id如下: " +",".join(map(str,rank_list))
        text = alpaca_prompt.format(instruction, inputs, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="./n_train_data.json",split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = transformers.TrainingArguments(
        dataloader_drop_last=True,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 200,
        # num_train_epochs=2,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1, 
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/mnt/HDD/syj/syj_poi/poi_query/ft_model2",
        report_to="none"
    ),
    # peft_config=lora_config,
)

trainer.train(resume_from_checkpoint=False)
lora_path='/mnt/HDD/syj/syj_poi/poi_query/ft_model2'
model.save_pretrained_merged(lora_path, tokenizer, save_method = "merged_16bit")