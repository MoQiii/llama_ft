import http.client
import json
from socket import timeout
import time

input_file = 'data/poi_rerank_test_index.json'

with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

payloads = []
model_name = "gpt-4o-mini"
for item in data:
    input_content = item["input"]
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": f"基于地址和geohash对候选兴趣点列表进行排序，返回候选列表的数字索引。下面总共有一条查询和其对应的geohash。此外，还有你需要排序的候选地址和geohash对，对应的索引从0开始。你需要返回排序后的索引,也只需要返回索引,总共100个索引，从0-99，返回的示例如：@13，14，0，5，99@等100个数,序列用@包裹。{input_content}"
            }
        ],
        "stream": False
    })
    payloads.append(payload)

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer sk-Klf5SE107OW4EEnf5hRoDWJhvIn6Wm3KAxNfCCbL5WEaJUiT'
}

max_retries = 6
retry_delay = 10

# 打开日志文件用于记录响应
with open(f"responses_{model_name}.jsonl", "a", encoding="utf-8") as log_file:
    for index, payload in enumerate(payloads):
        item = data[index]
        retries = 0
        success = False
        if index==20:
            break
        while retries < max_retries and not success:
            conn = http.client.HTTPSConnection("api.gptgod.work")
            
            try:
                print(f"Processing item {index + 1}/{len(payloads)}...")
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                if res.status == 200:
                    data_response = res.read().decode("utf-8")
                    response_json = json.loads(data_response)
                    
                    # 写入响应日志
                    log_file.write(json.dumps(response_json) + "\n")
                    log_file.flush()
                    
                    # 解析并存储结果到原始数据
                    try:
                        result = response_json['choices'][0]['message']['content']
                        item[model_name] = result.strip()
                    except (KeyError, IndexError) as e:
                        print(f"Failed to parse response for item {index + 1}: {str(e)}")
                        item[model_name] = ""
                    
                    success = True
                    print(f"Successfully processed item {index + 1}")
                else:
                    print(f"HTTP Error {res.status} for item {index + 1}")
                    raise Exception(f"HTTP Error: {res.status}")
                
            except (timeout, TimeoutError):
                print(f"Timeout occurred on item {index + 1}, retry {retries + 1}/{max_retries}")
                retries += 1
                time.sleep(retry_delay)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON response for item {index + 1}: {e}")
                break
            except Exception as e:
                print(f"Error processing item {index + 1}: {str(e)}")
                retries += 1
                if retries < max_retries:
                    print(f"Retrying item {index + 1}... ({retries}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached for item {index + 1}")
                    item[model_name] = ""
                    break
            finally:
                conn.close()

# 保存更新后的数据
output_filename = f"updated_data_{model_name}.json"
with open(output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)

print(f"Process completed. Updated data saved to {output_filename}")