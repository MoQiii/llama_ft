import json

# from sklearn.metrics import PredictionErrorDisplay

def parse_ids(s: str) -> list[int]:
    """解析字符串中的ID列表"""
    return [int(x.strip()) for x in s.split(":")[-1].split(",")]

top_1=[0]
top_5=list(range(5))
top_10=list(range(10))
top_20=list(range(20))
top_50=list(range(50))
def calculate_accuracy(predict: list[int], label: list[int],top_n) -> int:
    """
    计算单个样本的Top-n准确率：
    """
    # 1. 找到predict中数值0-n的索引位置
    predict_indices_of_0_to_9 = {
        idx for idx, num in enumerate(predict) if num in top_n
    }
    
    # 2. 找到label中元素0的索引
    try:
        label_0_index = label.index(0)
    except ValueError:
        # 如果label中没有0，视为错误（根据需求调整）
        return 0
    
    # 3. 检查是否存在
    return 1 if label_0_index in predict_indices_of_0_to_9 else 0

def calculate_mrr(predict: list[int], label: list[int]) -> float:
    """
    计算单个样本的MRR贡献值：
    1. 找到label中第一个0的索引作为正确答案的位置
    2. 在predict中查找该位置第一次出现的排名
    3. 返回倒数排名（1/rank），若未找到则返回0
    """
    try:
        correct_pos = label.index(0)  # 获取正确答案的位置
    except ValueError:
        return 0.0  # label中没有0，无法计算
    
    try:
        rank = predict.index(correct_pos) + 1  # 计算1-based排名
        return 1.0 / rank
    except ValueError:
        return 0.0  # predict中未找到正确位置

def main(input_file: str,predict_colunm_name:str,label_colunm_name:str):
    """主函数：计算所有样本的Top-10准确率"""
    with open(input_file, "r") as f:
        data = json.load(f)
    top_n = top_10
    n = len(top_n)
    total_correct = 0
    total_mrr = 0
    for item in data:
        try:
            predict = parse_ids(item[predict_colunm_name])
            label = parse_ids(item[label_colunm_name])
        except KeyError:
            continue
        total_correct += calculate_accuracy(predict, label,top_n)
        total_mrr += calculate_mrr(predict, label)
    accuracy = total_correct / len(data)
    total_mrr = total_mrr / len(data)
    print(f"mrr: {total_mrr:.4f}")
    print(f"top_{n}准确率: {accuracy:.4f}")
# file_path ='gpt-4o-mini_output.json'
file_path ='qwen-72b_output.json'
# file_path ='./saves/GLM-4-9B/lora/eval_2025-03-11-09-31-10/generated_predictions.jsonl'

if __name__ == "__main__":
    
    main(file_path,"qwen-72b_content","label")


# rel_path='data/poi_rerank_test_index.json'

# def extract_json_output(json_path):
#     with open(rel_path, "r", encoding="utf-8") as f:
#         data = json.load(f)  # 读取整个 JSON 数组

#     outputs = []
#     for item in data:
#         if "output" in item:
#             # 提取冒号后的内容，并分割成数值列表
#             indices_str = item["output"].split(": ")[-1]
#             indices = list(map(int, indices_str.split(",")))
#             outputs.append(indices)
#     return outputs

# def extract_jsonl_predict(jsonl_path):
#     predicts = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line.strip())
#             if "predict" in item:
#                 # 提取冒号后的内容，并分割成数值列表
#                 indices_str = item["predict"].split(": ")[-1]
#                 indices = list(map(int, indices_str.split(",")))
#                 predicts.append(indices)
#     return predicts

# # 计算测试数据无label时准确率
# def main2(input_file: str):
#     """主函数：计算所有样本的Top-10准确率"""
#     labels=extract_json_output("")
#     predicts=extract_jsonl_predict("")
#     top_n = top_1
#     n = len(top_n)
#     total_correct = 0
#     for label,predict in zip(labels,predicts):
#         # predict = parse_ids(item["predict"])
#         # label = parse_ids(item["label"])
        
#         # 确保数据完整性（可选）
#         # if len(predict) != 100 or len(label) != 100:
#         #     print(f"数据长度错误，跳过该样本")
#         #     continue
        
#         total_correct += calculate_top10_accuracy(predict, label,top_n)
    
#     accuracy = total_correct / len(labels)
#     print(f"top_{n}准确率: {accuracy:.4f}")
