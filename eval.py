# import json
# import math

# def compute_ndcg(predict, label, k=None):
#     rel = {id: (len(label) - pos) for pos, id in enumerate(label)}
#     dcg = 0.0
#     for i, id in enumerate(predict):
#         rank = i + 1
#         if k is not None and rank > k:
#             break
#         rel_score = rel.get(id, 0)
#         dcg += rel_score / math.log2(rank + 1)
#     idcg = 0.0
#     for i, id in enumerate(label):
#         rank = i + 1
#         if k is not None and rank > k:
#             break
#         rel_score = rel.get(id, 0)
#         idcg += rel_score / math.log2(rank + 1)
#     return dcg / idcg if idcg != 0 else 0.0

# def compute_mrr(predict, label):
#     if not label:
#         return 0.0
#     target_id = label[0]
#     try:
#         pos = predict.index(target_id) + 1
#         return 1.0 / pos
#     except ValueError:
#         return 0.0

# def compute_topk_accuracy(predict, label, k):
#     true_top = set(label[:k])
#     pred_top = set(predict[:k])
#     return len(true_top & pred_top) / k

# def evaluate_jsonl(file_path, top_k_list=[1, 5, 10,20,50]):
#     ndcg_scores = []
#     mrr_scores = []
#     topk_acc = {k: [] for k in top_k_list}

#     with open(file_path, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             # 提取ID列表
#             predict_str = data['predict'].split(': ')[-1].strip()
#             label_str = data['label'].split(': ')[-1].strip()
#             predict = list(map(int, predict_str.split(',')))
#             label = list(map(int, label_str.split(',')))

#             # 计算NDCG
#             ndcg = compute_ndcg(predict, label)
#             ndcg_scores.append(ndcg)

#             # 计算MRR
#             mrr = compute_mrr(predict, label)
#             mrr_scores.append(mrr)

#             # 计算Top-K正确率
#             for k in top_k_list:
#                 acc = compute_topk_accuracy(predict, label, k)
#                 topk_acc[k].append(acc)

#     # 计算平均值
#     results = {
#         'NDCG': sum(ndcg_scores) / len(ndcg_scores),
#         'MRR': sum(mrr_scores) / len(mrr_scores),
#         'TopK_Accuracy': {k: sum(acc)/len(acc) for k, acc in topk_acc.items()}
#     }
#     return results

# # 使用示例 
# # file_path ='./saves/GLM-4-9B/lora/eval_2025-02-07-19-07-13/generated_predictions.jsonl'
# file_path ='./saves/GLM-4-9B/lora/eval_2025-02-10-15-44-50/generated_predictions.jsonl'
# # file_path ='./saves/Llama-3.1-8B-Instruct/lora/eval_2025-02-22-14-14-36/generated_predictions.jsonl'
# results = evaluate_jsonl(file_path)
# print(f"NDCG: {results['NDCG']}")
# print(f"MRR: {results['MRR']}")
# for k, acc in results['TopK_Accuracy'].items():
#     print(f"Top-{k} Accuracy: {acc}")

import json
import math
from typing import List, Dict

def parse_ids(s: str) -> List[int]:
    """解析包含ID列表的字符串"""
    ids_str = s.split(":")[-1].strip()
    return [int(x.strip()) for x in ids_str.split(",")]

def calculate_accuracy(predicted: List[int], label: List[int]) -> float:
    """计算准确率：预测位置与真实位置完全匹配的比例"""
    # return sum(1 for p, l in zip(predicted, label) if p == l) / len(predicted)
    return sum(1 for p, l in zip(predicted, label) if p == l and l==0)

def calculate_ndcg(predicted: List[int], label: List[int]) -> float:
    """计算NDCG指标"""
    # 构建ID到真实位置的映射字典
    id2rel = {id: (100 - idx) for idx, id in enumerate(label)}
    
    # 计算预测排序的DCG
    dcg = sum(
        id2rel[id] / math.log2(pos + 2)  # pos从0开始，因此+2
        for pos, id in enumerate(predicted)
    )
    
    # 计算理想排序的IDCG
    idcg = sum(
        (100 - idx) / math.log2(idx + 2)
        for idx in range(len(label))
    )
    
    return dcg / idcg if idcg != 0 else 0.0

def calculate_mrr(predicted: List[int], label: List[int]) -> float:
    """计算MRR指标"""
    # 构建预测ID到位置的映射字典
    id2pos = {id: pos for pos, id in enumerate(predicted)}
    
    # 计算每个真实ID的倒数排名
    reciprocal_ranks = [
        1 / (id2pos[id] + 1)  # +1因为位置从0开始
        for id in label
    ]
    
    return sum(reciprocal_ranks) / len(label)

def main(input_file: str):
    """主计算流程"""
    # 读取数据
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    
    # 初始化累计值
    total_acc, total_ndcg, total_mrr = 0.0, 0.0, 0.0
    
    for item in data:
        predicted = parse_ids(item["predict"])
        label = parse_ids(item["label"])
        
        # 指标计算
        acc = calculate_accuracy(predicted, label)
        # ndcg = calculate_ndcg(predicted, label)
        # mrr = calculate_mrr(predicted, label)
        
        # 累计结果
        total_acc += acc
        # total_ndcg += ndcg
        # total_mrr += mrr
    
    # 输出平均结果
    print(f"Average Accuracy: {total_acc/len(data):.4f}")
    print(f"Average NDCG: {total_ndcg/len(data):.4f}")
    print(f"Average MRR: {total_mrr/len(data):.4f}")
file_path ='./saves/GLM-4-9B/lora/eval_2025-02-10-15-44-50/generated_predictions.jsonl'
file_path ='./saves/GLM-4-9B/lora/eval_new_prompt/generated_predictions.jsonl'

if __name__ == "__main__":
    main(file_path)