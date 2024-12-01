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

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ["WANDB_API_KEY"] = 'e525beaf7491e4835d63e7a662caa12ffacff534' # 将引号内的+替换成自己在wandb上的一串值
# os.environ["WANDB_MODE"] = "offline" # 离线 （此行代码不用修改）
# os.environ["WANDB_PROJECT"] = "llama3-8b-peft-alpaca"
# os.environ["WANDB_NOTEBOOK_NAME"] = "llama3-8b-peft-alpaca"

# device_map={"": PartialState().process_index}

# accelerate launch llama_finetune.py --num_processes 2

# train_data = np.load("./llm_finetune_train_data.npy", allow_pickle=True).tolist()
# test_data = np.load("./llm_finetune_test_data.npy", allow_pickle=True).tolist()

# local_rank = os.getenv("LOCAL_RANK")
# device_string = "cuda:" + str(local_rank)
# print("device_string", device_string)

# dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
# print()

# device_string = "cuda:1"

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# access_token = "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"
# system_message = "你是一个有用的POI检索助手，你可以根据查询对候选POI的相似度进行排序。"

model_id = "/mnt/HDD/TSC_xx/finetune_data/model/meta_llama3.1_8b"
# model_id = "/mnt/HDD/Rensifei/Llama3-Chinese-8B-Instruct"
# model_id = "/mnt/HDD/syj/syj_poi/poi_query/ft_model2"
access_token = "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"
system_message = "你是一个聪明的POI检索助手，你可以根据查询对候选POI的相似度进行排序。"

alpaca_prompt = """You are a useful POI retrieval assistant, you can rank candidate POIs according to the similarity of the query.

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
lora_config = LoraConfig(
    # r=8,
    # target_modules = find_all_linear_names(model),
    task_type="QUESTION_ANS",
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   #对于非常长的上下文，为True或unsloth
    random_state = 42,           #3407？？
    use_rslora = False,
    loftq_config = None,
)

# def generate_finetune_data():
#     train_data = np.load("./llm_finetune_train_data.npy", allow_pickle=True).tolist()
    
#     dataset = {"messages": []}
#     for data in tqdm(train_data):
#         records = [{"real_rank": index, "address": record["address"], "geohash": record["geohash"]} for index, record in enumerate(data["records"])]
#         random.shuffle(records)
#         rank_map = dict(sorted({record["real_rank"]: index for index, record in enumerate(records)}.items()))
#         rank_list = list(rank_map.values())
#         record_list = [[record["address"], record["geohash"]] for record in records]
        
#         candidate_list = [f"id: {i} | address: {address} | geohash: {geohash}" for i, (address, geohash) in enumerate(record_list)]
#         instruction = f"Please rank the candidate POIs by their similarity according to the query. \n query：{data['query']} \n candidate POIs：{candidate_list}"
#         messages = alpaca_prompt.format(instruction, "", str(rank_list)) + tokenizer.eos_token
#         dataset["messages"].append(messages)
#     dataset = Dataset.from_dict(dataset)
#     return dataset


# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )





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
        instruction = f"Sort the list of candidate POIs based on query and geohash similarity. Returns the ID of the candidate list for."
        inputs = f"query is: {query},geohash is: {geohash} ,The list of candidate POIs is: {candidate_list}"
        output = "The corresponding ids are returned as follows: " +",".join(map(str,rank_list))
        text = alpaca_prompt.format(instruction, inputs, output) + tokenizer.eos_token
        texts.append(text)
    if len(texts)==0:
        print("=============================================================")
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
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 40,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/mnt/HDD/syj/syj_poi/poi_query/ft_model2",
        # report_to="wandb"
    ),
)

# trainer = DPOTrainer(
#     model = model,
#     ref_model = None,
#     args = transformers.TrainingArguments(
#         per_device_train_batch_size = 4,
#         gradient_accumulation_steps = 4,
#         warmup_ratio = 0.1,
#         num_train_epochs = 3,
#         fp16 = not torch.cuda.is_bfloat16_supported(),
#         bf16 = torch.cuda.is_bfloat16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         seed = 42,
#         output_dir = "/mnt/HDD/syj/syj_poi/poi_query/ft_model2",
#     ),
#     beta = 0.1,
#     train_dataset = dataset,
#     # eval_dataset = YOUR_DATASET_HERE,
#     tokenizer = tokenizer,
#     max_length = max_seq_length,
#     dataset_num_proc = 2,
#     max_prompt_length = 8192,
# )



trainer.train(resume_from_checkpoint=False)
lora_path='/mnt/HDD/syj/syj_poi/poi_query/ft_model2'
model.save_pretrained_merged(lora_path, tokenizer, save_method = "merged_16bit")