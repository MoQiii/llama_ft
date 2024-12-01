from transformers import TextStreamer
import re



def get_action(self, state, env, cluster):
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