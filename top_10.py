import json

def parse_ids(s: str) -> list[int]:
    """解析字符串中的ID列表"""
    return [int(x.strip()) for x in s.split(":")[-1].split(",")]

def calculate_top10_accuracy(predict: list[int], label: list[int]) -> int:
    """
    计算单个样本的Top-10准确率：
    检查label中元素0的索引是否在predict中数值0-9对应的索引集合中
    """
    # 1. 找到predict中数值0-9的索引位置
    predict_indices_of_0_to_9 = {
        idx for idx, num in enumerate(predict) if num in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    }
    
    # 2. 找到label中元素0的索引
    try:
        label_0_index = label.index(0)
    except ValueError:
        # 如果label中没有0，视为错误（根据需求调整）
        return 0
    
    # 3. 检查是否存在
    return 1 if label_0_index in predict_indices_of_0_to_9 else 0

def main(input_file: str):
    """主函数：计算所有样本的Top-10准确率"""
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    
    total_correct = 0
    for item in data:
        predict = parse_ids(item["predict"])
        label = parse_ids(item["label"])
        
        # 确保数据完整性（可选）
        if len(predict) != 100 or len(label) != 100:
            print(f"数据长度错误，跳过该样本")
            continue
        
        total_correct += calculate_top10_accuracy(predict, label)
    
    accuracy = total_correct / len(data)
    print(f"Top-10准确率: {accuracy:.4f}")
file_path ='./saves/GLM-4-9B/lora/eval_2025-02-10-15-44-50/generated_predictions.jsonl'
if __name__ == "__main__":
    main(file_path)