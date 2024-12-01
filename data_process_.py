import numpy as np
from datasets import Dataset
import json

# Step 1: 加载 .npy 文件
def load_npy_data(file_path):
    """
    加载 .npy 文件并返回数据。
    """
    data = np.load(file_path, allow_pickle=True).tolist()  # 加载为字典列表
    return data

# Step 2: 转换为 Hugging Face 支持的格式
def convert_to_hf_format(data):
    """
    将数据转换为 Hugging Face Dataset 支持的字典格式。
    """
    # 提取字段
    queries = [item["query"] for item in data]
    geohashes = [item["geohash"] for item in data]
    records_list = [item["records"] for item in data]

    # 构造为字典格式
    formatted_data = {
        "query": queries,
        "geohash": geohashes,
        "records": records_list
    }
    return formatted_data

# Step 3: 保存为 JSON 文件（可选，用于 load_dataset 方法）
def save_as_individual_json(data, output_file):
    """
    将数据保存为 JSON Lines 格式（每行一个 JSON 对象）。
    """
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for item in data:
    #         json_line = {
    #             "query": item["query"],
    #             "geohash": item["geohash"],
    #             "records": item["records"]
    #         }
    #         json.dump(json_line, f, ensure_ascii=False)
    #         f.write("\n")  # 每行一个 JSON 对象
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)  # 使用缩进和UTF-8编码保存

# Step 4: 加载为 Hugging Face Dataset
def create_hf_dataset(formatted_data):
    """
    通过 Hugging Face Dataset 创建数据集。
    """
    dataset = Dataset.from_dict(formatted_data)
    return dataset

# 主函数
if __name__ == "__main__":
    # 指定 .npy 文件路径
    # npy_file_path = "./llm_finetune_train_data.npy" 
    npy_file_path = "./llm_finetune_test_data.npy"  

    # 加载数据
    data = load_npy_data(npy_file_path)

    # 转换数据格式
    # formatted_data = convert_to_hf_format(data)

    # 保存为 JSON 文件（可选）
    json_file_path = "test_data.json"
    save_as_individual_json(data, json_file_path)

    # 加载为 Hugging Face Dataset
    # dataset = create_hf_dataset(formatted_data)

    # 打印数据集的一些信息
    # print(dataset)
    # print(dataset[0])  # 查看第一条数据
