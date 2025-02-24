import json
import random
from transformers import AutoTokenizer

def process_and_save_data(input_file, output_file, tokenizer):
    """
    读取JSON文件，处理后直接保存为指定格式的JSON文件
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        tokenizer: 分词器
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储所有处理后的数据
    processed_data = []
    
    # 处理每条数据
    for example in data:
        query = example["query"]
        geohash = example["geohash"]
        record_list = example["records"]
        
        records = [{"real_rank": index, 
                   "address": record["address"], 
                   "geohash": record["geohash"]} 
                  for index, record in enumerate(record_list)]
        
        random.shuffle(records)
        
        rank_list = []
        candidate_list = []
        
        for i, record in enumerate(records):
            rank_list.append(record["real_rank"])
            candidate_list.append(
                f"id: {i} | address: {record['address']} | geohash: {record['geohash']}"
            )
            
        instruction = "基于地址和geohash对候选兴趣点列表进行排序，返回候选列表的数字ID。下面总共有一条查询和其对应的geohash。此外，还有你需要排序的候选地址和geohash对，对应的ID从0开始。你需要返回排序后的ID，把你认为和查询更相似的地址的ID排到更前面。"
        
        inputs = f"查询是: {query}, geohash是: {geohash}, 候选poi列表是: {candidate_list}"
        
        output = "返回的对应的id如下: " + ",".join(map(str, rank_list))
        
        # 按照指定格式构造数据
        formatted_item = {
            "instruction": instruction,
            "input": inputs,
            "output": output
        }
        
        processed_data.append(formatted_item)
    
    # 保存为指定格式的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，共处理 {len(processed_data)} 条数据")
    print(f"结果已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 初始化tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("your_model_name")
    
    # 输入输出文件路径
    input_file = "./n_train_data.json"
    output_file = "poi_rerank_data_all.json"
    
    # 处理并保存数据
    process_and_save_data(input_file, output_file, None)