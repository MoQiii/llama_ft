# from unsloth import FastLanguageModel

from utils.llama_utils import load_json, dump_json, get_state_detail, state2text, getPrompt, action2code, code2action, eight_phase_list, four_phase_list, torch_gc
import os
import time
import numpy as np
import wandb
from utils.cityflow_env import CityFlowEnv
import utils.config as config
from utils.aft_rank_loss_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_int8_training
from torch.optim.lr_scheduler import StepLR
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from copy import deepcopy
import re
import json
import shutil
import concurrent.futures
import copy
import random
from bitsandbytes.nn import modules
import copy
max_length=5000
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
import os
os.environ["WANDB_API_KEY"] = 'e525beaf7491e4835d63e7a662caa12ffacff534' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）

four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}

alpaca_prompt = """You are an expert in traffic management. You can use your knowledge of traffic domain knowledge to solve this traffic signal control tasks.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def custom_deepcopy(obj, memo=None):
    if memo is None:
        memo = {}

    if isinstance(obj, dict):
        copied_obj = obj.__class__()
        memo[id(obj)] = copied_obj
        for k, v in obj.items():
            copied_obj[custom_deepcopy(k, memo)] = custom_deepcopy(v, memo)
        return copied_obj
    elif isinstance(obj, list):
        copied_obj = obj.__class__()
        memo[id(obj)] = copied_obj
        for item in obj:
            copied_obj.append(custom_deepcopy(item, memo))
        return copied_obj
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return copy.deepcopy(obj, memo)
    else:
        return obj  # 对于无法复制的对象，直接返回原对象或替换为描述字符串

# 使用 custom_deepcopy 函数代替标准的深拷贝
def merge(dic1, dic2):
    dic_tmp = custom_deepcopy(dic1)
    dic_tmp.update(custom_deepcopy(dic2))
    return dic_tmp

# 检查路径是否存在，如果不存在就创建路径
def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            pass
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            pass
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])
def remove_unserializable(data):
    if isinstance(data, dict):
        return {k: remove_unserializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [remove_unserializable(v) for v in data]
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        return str(data)  # 或者返回 None
# 将配置信息保存到指定的目录中
def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]

    # 移除不可序列化的对象
    serializable_agent_conf = remove_unserializable(dic_agent_conf)
    serializable_traffic_env_conf = remove_unserializable(dic_traffic_env_conf)

    with open(os.path.join(path, "agent.conf"), "w") as f:
        json.dump(serializable_agent_conf, f, indent=4)
    with open(os.path.join(path, "traffic_env.conf"), "w") as f:
        json.dump(serializable_traffic_env_conf, f, indent=4)

# 复制交通路网
def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))



class LLMAgent:
    def __init__(self, dic_agent_conf, tokenizer, llama_model):
        self.dic_agent_conf = dic_agent_conf
        self.tokenizer = tokenizer
        self.llama_model = llama_model
        self.phases = four_phase_list
        self.system_prompt = load_json("./prompts/prompt_llama.json")["system_prompt"]

    def get_action(self, state, env, cluster):
        from transformers import TextStreamer
        import re

        combined_state = []
        self.tokenizer.pad_token_id = 500
        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_length": 8000
        }
        for intersection_name in cluster:
            intersection = env.intersection_dict[intersection_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, env)
            state_txt = self.state2table(statistic_state)[0]
            combined_state.append({
                "intersection": intersection_name,
                "state_txt": state_txt,
                "state": statistic_state,
                "state_incoming": statistic_state_incoming,
                "avg_speed": mean_speed
            })
        avg_speeds = [info["avg_speed"] for info in combined_state]
        prompt = getPrompt(self, combined_state, avg_speeds, env)
        prompt = prompt[1]['content']
        inputs = self.tokenizer(
            [
                alpaca_prompt.format(
                    prompt,
                    "",
                    ""
                )
            ], return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(self.tokenizer)

        response_ids = None
        response = None
        while response is None:
            try:
                response_gen = self.llama_model.generate(**inputs, pad_token_id=1000,streamer=text_streamer, max_new_tokens=1024)
                response = self.tokenizer.batch_decode(response_gen, skip_special_tokens=True)[0]
            except Exception as e:
                response_gen = self.llama_model.generate(**inputs, pad_token_id=1000,streamer=text_streamer, max_new_tokens=1024)
                response = self.tokenizer.batch_decode(response_gen, skip_special_tokens=True)[0]
            signals = {}
            for state_info in combined_state:
                intersection_name = state_info["intersection"]
                signal_answer_pattern = fr'<signal intersection="{intersection_name}">(.*?)</signal>'
                match = re.search(signal_answer_pattern, response)
                if not match or (match.group(1) not in four_phase_list):
                    response = None
                    break
                signals[intersection_name] = match.group(1)

        actions = []
        for intersection_name, signal_text in signals.items():
            action = action2code(signal_text) if signal_text in four_phase_list else 0
            actions.append(action)

        return actions

    def state2table(self, state):
        state_txt = "lane,early queued,average waiting time,segment 1,segment 2,segment 3,segment 4\n"
        max_queue_len = 0
        for lane in state:
            queue_len = int(state[lane]['queue_len'])
            avg_wait_time = int(state[lane]['avg_wait_time'])
            max_queue_len = queue_len if queue_len > max_queue_len else max_queue_len
            state_txt += f"{location_dict_detail[lane[0]]} {direction_dict[lane[1]]} lane,{queue_len},{avg_wait_time}s"

            for i, n in enumerate(state[lane]['cells']):
                n = int(n)
                state_txt += f",{n}"
            state_txt += "\n"

        return state_txt, max_queue_len
class LLama_Test:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llama_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.initialize()
        self.phases = four_phase_list
        # self.system_prompt = load_json("./prompts/prompt_llama.json")["system_prompt"]

        # 初始化一个 agent
        self.agent = LLMAgent(self.dic_agent_conf, self.tokenizer, self.llama_model)

    # 初始化大模型
    def initialize_llm(self):
        device_map = "sequential"
        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
        # FastLanguageModel.for_inference(self.llama_model)
        # self.llama_model.eval()
        self.tokenizer.pad_token_id = 500
        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_length": 8000
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    import concurrent.futures

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()
        start_time = time.time()
        state_action_log = []
        avg_speeds = []

        # 每个代理对应的道路
        roads_of_interest = {
            "agent_1": ["intersection_1_1", "intersection_1_2", "intersection_1_3"],
            "agent_2": ["intersection_2_1", "intersection_2_2", "intersection_2_3"],
            "agent_3": ["intersection_3_1", "intersection_3_2", "intersection_3_3"],
            "agent_4": ["intersection_4_1", "intersection_4_2", "intersection_4_3"],
        }
        # 3600/30=120
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break

            action_list = []

            while len(action_list) < 12:
                action_list.clear()  # 清空action_list，以确保重新获取动作

                # 单线程获取每个代理的动作
                for agent, roads in roads_of_interest.items():
                    actions = self.agent.get_action(
                        [state[i] for i, inter in enumerate(self.env.list_intersection) if inter.inter_name in roads],
                        self.env,
                        roads
                    )
                    action_list.extend(actions)

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)

            current_time = self.env.get_current_time()
            state = next_state

            total_reward += sum(rewards)

            # 计算每个路口等待车辆数量
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            # 计算每个车辆的等待时间
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)


        # 计算每辆车在测试期间的总旅行时间等指标
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time
        }

        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results


    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)

        model_name = self.dic_traffic_env_conf['MODEL_NAME']
        roadnet = self.roadnet
        trafficflow = self.trafficflow
        num_phases = len(self.dic_traffic_env_conf.get('PHASE', ''))

        group_name = f"{model_name}-{roadnet}-{trafficflow}-{num_phases}_Phases"
        if len(group_name) > 128:
            import hashlib
            group_name = hashlib.md5(group_name.encode()).hexdigest()

        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=group_name,
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    '''
    ======================= Class Utils =======================
    '''
    # 计算每个状态中的车辆数量
    def get_vehicle_num(self, states):
        veh_nums = []

        for state in states:
            vehicle_num = 0

            for lane_data in state.values():
                # 检查 'queue_len' 键是否存在
                queue_len = lane_data.get('queue_len', 0)
                vehicle_num += queue_len

                # 处理 'cells' 键，如果不存在则返回空列表
                cells = lane_data.get('cells', [])
                vehicle_num += sum(cells)

            veh_nums.append(vehicle_num)

        return veh_nums

    # 计算规范化的奖励值
    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards
    def state2table(self, state):
        state_txt = "lane,early queued,average waiting time,segment 1,segment 2,segment 3,segment 4\n"
        max_queue_len = 0
        for lane in state:
            queue_len = int(state[lane]['queue_len'])
            avg_wait_time = int(state[lane]['avg_wait_time'])
            max_queue_len = queue_len if queue_len > max_queue_len else max_queue_len
            state_txt += f"{location_dict_detail[lane[0]]} {direction_dict[lane[1]]} lane,{queue_len},{avg_wait_time}s"

            for i, n in enumerate(state[lane]['cells']):
                n = int(n)
                state_txt += f",{n}"
            state_txt += "\n"

        return state_txt, max_queue_len
