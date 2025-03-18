import json
import re

# def remove_content_spaces(input_file, output_file):
#     """
#     读取JSON文件并去除内容中的空格后保存
    
#     Args:
#         input_file: 输入JSON文件路径
#         output_file: 输出JSON文件路径
#     """
#     # 读取JSON文件
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 处理每条数据，去除内容中的空格
#     for item in data:
#         item["instruction"] = item["instruction"].replace(" ", "")
#         item["input"] = item["input"].replace(" ", "")
#         item["output"] = item["output"].replace(" ", "")
    
#     # 保存处理后的JSON文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=None)
    
#     print(f"处理完成，结果已保存到: {output_file}")

# def get_api_repsonses_index(text):
#     text = """根据提供的查询地址和geohash，我们可以对候选兴趣点列表进行排序。首先，查询地址是“富阳区体育中心内旅游体育厅”，对应的geohash是“wtm6cnuh81vz”。我们需要根据geohash与查询地址的地理接近程度来排序候选列表。\n\n1. 对于候选列表中的每个地理位置，我们计算其geohash与查询geohash的相似度（一般来说，geohash的前缀相同表示位置较近）。\n2. 对候选地址与查询地址的语义相似度进行评估。与“体育中心”相关的地址会被给予更高的排序优先级。\n3. 最终得到的排序根据这两方面的结果进行。\n\n经过排序，以下是候选地址的索引:\n\n```\n@25, 39, 17, 19, 9, 52, 4, 5, 7, 19, 61, 33, 55, 0, 1, 11, 39, 2, 32, 10, 79, 38, 18, 24, 21, 8, 6, 3, 12, 20, 40, 14, 12, 27, 26, 31, 40, 41, 15, 18, 29, 34, 43, 46, 49, 50, 53, 54, 56, 58, 63, 64, 66, 73, 74, 75, 78, 81, 85, 86, 88, 89, 90, 92, 93, 94, 96, 97, 98, 99, 77, 72, 71, 86, 81, 84, 87, 67, 64, 82, 80, 83, 76, 62, 37, 39, 70, 68, 69, 60, 58, 59, 29, 88, 90@\n```\n\n这个索引列表可以帮助提高候选兴趣点的相关性，找出离查询地址最接近及最相关的地点。"""  # 你的完整文本
#     # 提取两个@之间的内容
#     start = text.find('@') + 1
#     end = text.find('@', start)
#     numbers_str = text[start:end]

#     # 清洗并转换格式
#     numbers_list = [
#         int(num.strip()) 
#         for num in numbers_str.replace('\n', '').split(',') 
#         if num.strip().isdigit()
#     ]

#     print(numbers_list)

def extract_number_list(text):
    """从文本中提取两个@之间的数字列表"""
    try:
        # 使用正则表达式匹配两个@之间的所有内容（包括换行符）
        match = re.search(r'@(.*?)@', text, re.DOTALL)
        if not match:
            print("未找到@标记对")
            return None
        
        numbers_str = match.group(1).strip()
        # 清洗数据：移除所有空白字符和特殊符号
        clean_str = re.sub(r'\s+', '', numbers_str)
        # 分割并验证数字
        return [int(num) for num in clean_str.split(',') if num.isdigit()]
    except (AttributeError, ValueError) as e:
        print(f"数据解析失败: {str(e)}")
        return None
column='gpt-4o-mini'
def process_json_file(input_file, output_file):
    """处理JSON文件并保存结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_data = []
        for index, item in enumerate(data):
            if column not in item:
                print(f"第 {index} 项缺少{column}字段")
                continue
                
            original_text = item[column]
            if not isinstance(original_text, str):
                print(f"第 {index} 项{column}字段不是字符串类型")
                continue
                
            numbers = extract_number_list(original_text)
            if numbers:
                # 保留原始文本在新字段中
                item['original_response'] = original_text  
                item[f'{column}_content'] = ','.join([str(i) for i in numbers])
                print(f"第 {index} 项处理成功，找到 {len(numbers)} 个数字")
            else:
                print(f"第 {index} 项未找到有效数字列表")
                item[column] = []  # 保持数据结构一致

            processed_data.append(item)

        # 写入 JSONL 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
        print(f"处理完成，结果已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"文件处理失败: {str(e)}")
        return False
# 处理api返回
def process_json_file_api(input_file, output_file):
    """处理JSON文件并保存结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        processed_data = []
        for index, item in enumerate(data[-1:]):
            # if column not in item:
            #     print(f"第 {index} 项缺少{column}字段")
            #     continue
                
            original_text = item['choices'][0]['message']['content']
            # if not isinstance(original_text, str):
            #     print(f"第 {index} 项{column}字段不是字符串类型")
            #     continue
                
            numbers = extract_number_list(original_text)
            if numbers:
                # 保留原始文本在新字段中
                item['original_response'] = original_text  
                item[f'{column}_content'] = ','.join([str(i) for i in numbers])
                print(f"第 {index} 项处理成功，找到 {len(numbers)} 个数字")
            else:
                print(f"第 {index} 项未找到有效数字列表")
                item[column] = []  # 保持数据结构一致

            processed_data.append(item)

        # 写入 JSONL 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
        print(f"处理完成，结果已保存到 {output_file}")
        return True
    except Exception as e:
        print(f"文件处理失败: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    input_file = 'responses_gpt-4o-mini_1.jsonl'
    output_file = f"{column}_output.json" 
    process_json_file_api(input_file,output_file)

# import json
# if __name__ == "__main__":

#     # 输入文件和输出文件路径
#     input_file = "data/poi_rerank_test_index_rel.json"
#     output_file = "data/poi_rerank_test_index_rel.json"

#     # 读取整个 JSON 文件
#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)  # 假设 data 是一个列表或字典

#     # 将每个对象中的 "output" 字段的值设为空字符串
#     if isinstance(data, list):
#         for item in data:
#             if "output" in item:
#                 item["output"] = ""  # 赋值为空字符串
#     elif isinstance(data, dict):
#         if "output" in data:
#             data["output"] = ""  # 赋值为空字符串
#     else:
#         print("未知的 JSON 结构，请检查文件格式。")

#     # 将修改后的数据写入新文件
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)