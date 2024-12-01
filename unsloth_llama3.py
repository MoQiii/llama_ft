from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import notebook_login
import wandb
import os
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from llama_finetune import generate_finetune_data



device_id = "cuda:1"
max_seq_length = 8192
dtype = None
load_in_4bit = True
val_set_size = 0.05
mask = False
cutoff_len = 5000
train_on_inputs = False
import bitsandbytes as bnb
import os
os.environ["WANDB_API_KEY"] = 'e525beaf7491e4835d63e7a662caa12ffacff534' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline" # 离线 （此行代码不用修改）

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/HDD/TSC_xx/finetune_data/model/unsloth_llama3_8b_1",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
#
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

# 处理数据

alpaca_prompt = """You are an expert in traffic management. You can use your knowledge of traffic domain knowledge to solve this traffic signal control tasks.

Instruction:
{}

Input:
{}

Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = load_dataset("json", data_files="/mnt/HDD/TSC_xx/finetune_data/data/jinan_advancedcolight_real_2500_11_7_1.json")
# dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset["train"]
  # 设定验证集的比例
if val_set_size > 0:
    train_val = dataset.train_test_split(
        test_size=val_set_size, seed=2024
    )
    train_data = train_val["train"].map(formatting_prompts_func,batched=True)
    val_data = train_val["test"].map(formatting_prompts_func,batched=True)
else:
    train_data = dataset.map(formatting_prompts_func)
    val_data = None

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset= val_data,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 100,
        num_train_epochs=1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/mnt/HDD/TSC_xx/finetune_data/finetune_model/llama3_unsloth_100epoch_11_7_jinan_2500",
        # report_to="wandb"
    ),
)

os.environ["WANDB_PROJECT"] = "llama3-8b-peft-alpaca"
os.environ["WANDB_NOTEBOOK_NAME"] = "llama3-8b-peft-alpaca"

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f'GPU = {gpu_stats.name}. Max memory = {max_memory} GB.')
print(f'{start_gpu_memory} GB of memory reserved')


#%%
trainer_stats = trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")


lora_path='/mnt/HDD/TSC_xx/finetune_data/finetune_model/llama3_unsloth_100epoch_11_7_jinan_2500'
model.save_pretrained_merged(lora_path, tokenizer, save_method = "merged_16bit")