import json
import time
import requests
input_file = 'data/poi_rerank_test_index.json'

with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)
labels=[]
payloads = []
# model_name = "gpt-4o-mini"   ok
# model_name = "llama-3.1-70b"
model_name = "qwen-72b"
for item in data:
    input_content = item["input"]
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": f"基于地址和geohash对候选兴趣点列表进行排序，返回候选列表的数字索引。下面总共有一条查询和其对应的geohash。此外，还有你需要排序的候选地址和geohash对，对应的索引从0开始。你需要返回排序后的索引,也只需要返回数字索引,总共100个索引，从0-99，返回的示例如：@13，14，0，5，99@等100个数,序列用@包裹。{input_content}"
            }
        ],
        "stream": False
    }
    payloads.append(payload)
    labels.append(item["output"])
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-Klf5SE107OW4EEnf5hRoDWJhvIn6Wm3KAxNfCCbL5WEaJUiT'
}

url = "https://api.gptgod.work/v1/chat/completions"
# url = "https://api.gptgod.online/v1/chat/completions"
start = time.perf_counter()
max_retries = 6
retry_delay = 20
timeout = 30  # 设置连接和读取超时时间（秒）

def save_index_to_file(index_list, filename):
    """将整数索引列表保存到文件"""
    with open(filename, 'a') as file:
        for num in index_list:
            file.write(f"{num}\n")

def read_index_from_file(filename):
    """从文件读取并返回整数列表"""
    loaded_list = []
    with open(filename, 'r') as file:
        for line in file:
            loaded_list.append(int(line.strip()))
    return loaded_list

with open(f"responses_{model_name}_1.jsonl", "a", encoding="utf-8") as log_file:
    indexs=read_index_from_file("failure_qwen-72b_1_index.txt")
    payloads = [payloads[i] for i in indexs]
    for index, payload in enumerate(payloads):
        item = data[index]
        retries = 0
        success = False
        
        while retries < max_retries and not success:
            try:
                print(f"Processing item {index + 1}/{len(payloads)}...")
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                response.raise_for_status()  # 自动检查HTTP状态码

                # 处理响应数据
                response_json = response.json()   
                response_json["label"] = labels[index]           
                log_file.write(json.dumps(response_json, ensure_ascii=False) + "\n")
                log_file.flush()

                try:
                    result = response_json['choices'][0]['message']['content']
                    item[model_name] = result.strip()
                except (KeyError, IndexError) as e:
                    print(f"Failed to parse response for item {index + 1}: {str(e)}")
                    item[model_name] = ""

                success = True
                print(f"Successfully processed item {index + 1}")

            except requests.exceptions.Timeout as e:
                print(f"Timeout occurred on item {index + 1}, retry {retries + 1}/{max_retries}")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error {e.response.status_code} for item {index + 1}: {str(e)}")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response for item {index + 1}: {e}")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
            except Exception as e:
                
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
        if not success:
            item[model_name] = ""
            print(f"Max retries reached for item {index + 1}")
            save_index_to_file([index],f"failure_{model_name}_1_index.txt")






# 保存更新后的数据
output_filename = f"updated_data_{model_name}_1.json"
with open(output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(data[0:2800], outfile, ensure_ascii=False, indent=4)

print(f"Process completed. Updated data saved to {output_filename}")
end = time.perf_counter()
print(f"耗时: {end - start:.4f} 秒")