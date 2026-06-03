column_prompt_components = {
    "column_developer": "<|start|>developer<|message|># Instructions\nYou are a stepwise adder that reasons concisely. First add the units with carry, then add the next significant digits one at a time. Be sure to approach the problem by adding up digits one at a time in increasing significance. Once reasoning has concluded, give the user only the final answer.<|end|>",
    "column_restatement": '''The user asks: "What is {num1}+{num2}?" The developer instructions: "You are a stepwise adder that reasons concisely. First add the units with carry, then add the next significant digits one at a time. Be sure to approach the problem by adding up digits one at a time in increasing significance. Once reasoning has concluded, give the user only the final answer."''',
    "column_reasoning_prefix": "\n\nSo we need to do stepwise addition:",
    "column_reasoning_step": " {digit_name}: {num1_digit}+{num2_digit}={digit_sum}{prev_carry}{write_and_carry}",
    "prev_carry": " plus carry 1 = {digit_sum}",
    "write_and_carry": ", write {write} carry {carry}",
    "column_reasoning_suffix": " So result {sum}. Provide only the final answer. So output: {sum}."
}

digit_num_to_name = {
    1: "units",
    2: "Tens",
    3: "Hundreds",
    4: "Thousands",
    5: "Ten Thousands",
    6: "Hundred Thousands",
    7: "Millions",
    8: "Ten Millions",
    9: "Hundred Millions",
    10: "Billions",
}

name_to_digit_num = {v: k for k, v in digit_num_to_name.items()}

stepwise_prompt_components = {
    "stepwise_developer": "<|start|>developer<|message|># Instruction: In analysis, add two numbers stepwise. In final, output only the sum.<|end|>",
    "stepwise_restatement": '''The user asks: "What is {num1}+{num2}?" The instruction from the developer says: "In the analysis, add two numbers stepwise. For the final output, output only the sum." So we need to do the addition stepwise in the analysis, and then in the final output, just output the sum.''',
    "stepwise_reasoning_prefix": " So I need to do stepwise addition in analysis, then output only the sum in final. So in analysis, I will show the stepwise addition: {num1} + {num2}. Let's do it: {num1} + {num2} = {num1} + {num2_components} =",
    "stepwise_reasoning_step": " {intermediate_sum} + {next_digit} = {sum}",
    "stepwise_reasoning_suffix": ". So the sum is {sum}. In final, output only \"{sum}\"."
}

vanilla_prompt_components = {
    "vanilla_reasoning": '''The user asks: "What is {num1}+{num2}?" This is a simple arithmetic question. The answer is {sum}. The user likely expects a straightforward answer. There's no trick. So respond with "{sum}". Possibly add a brief explanation. But the user didn't ask for explanation. Just answer. So answer: {sum}.''',
}

prompt_components_OSS = {
    "system": "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: high\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
    "user": "<|start|>user<|message|>What is {num1}+{num2}?<|end|>",
    "analysis_prefix": "<|start|>assistant<|channel|>analysis<|message|>",
    "output_prefix": "<|end|><|start|>assistant<|channel|>final<|message|>",
    **column_prompt_components,
    **stepwise_prompt_components,
    **vanilla_prompt_components,
}

prompt_components_R1 = {
    "user": '''<｜begin▁of▁sentence｜><｜User｜>What is {num1}+{num2}? Do not tell me how to do it, just give me the answer. For example, "The answer is 397".<｜Assistant｜><think>''',
    "restatement": "\nI need to add {num1} and {num2}.",
    "reasoning": "\n\nFirst, I'll add the hundreds place: {hundreds1} + {hundreds2} equals {hundreds_sum}.\n\nNext, I'll add the tens place: {tens1} + {tens2} equals {tens_sum}.\n\nThen, I'll add the units place: {units1} + {units2} equals {units_sum}.\n\nFinally, I'll combine these results: {hundreds_sum} + {tens_sum} + {units_sum} equals ",
    "result": "{sum}.",
    "output_prefix": "\n</think>\n\nThe answer is \\boxed{",
}

intervene_ids_stepwise_3_digit = {
    "all": list(range(7,28)),
    "restatement": [7,8,9,10,11,12,13],
    "reasoning": [14,15,16,17,18,19,20,21,22,23,24,25],
    "result": [26,27],
    "copy": [7,8,9,10,11,12,13,17,18,20,21,23,24,26,27],
    "scale": [14,15,16],
    "reasoning_copy": [17,18,20,21,23,24],
    "intermediate_sums": [19,22,25],
}

intervene_ids_stepwise_3_digit_h = {
    "all": [8,10,12,14,18,19,20,22,23,25,26,27],
    "restatement": [8,10,12],
    "reasoning": [14,18,19,20,22,23,25],
    "result": [26,27],
}

intervene_ids_column_3_digit_c = {
    "all": [8,10,11,12,24,25],
    "restatement": [8],
    "reasoning": [10,11,12],
    "result": [24,25],
}

intervene_ids_vanilla_2_digit_h = {
    "all": [8,9,10,11],
    "restatement": [8],
    "reasoning": [],
    "result": [9,10,11],
}

intervene_ids_R1_3_digit = {
    "all": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
    "restatement": [3,4],
    "reasoning": [5,6,7,8,9,10,11,12,13,14,15,16],
    "result": [17],
}

intervene_ids_R1_3_digit_h = {
    "all": [4,6,7,14,17],
    "restatement": [4],
    "reasoning": [6,7,14],
    "result": [17],
}

intervene_id_to_tok_pos_stepwise_3_digit_h = {
    6: 93,
    8: 111,
    10: 201,
    12: 211,
    14: 217,
    18: 229,
    19: 232,
    20: 235,
    21: 238,
    22: 241,
    23: 244,
    25: 250,
    26: 257,
    27: 265,
}

intervene_id_to_tok_pos_column_3_digit_h = {
    8: 156,
    10: 239,
    11: 241,
    12: 245,
    24: 285,
    25: 297,
}

intervene_id_to_tok_pos_vanilla_2_digit_h = {
    6: 69,
    8: 87,
    9: 100,
    10: 118,
    11: 141,
}

intervene_id_to_tok_pos_R1_3_digit_h = {
    4: 45,
    6: 60,
    7: 63,
    14: 110,
    17: 119,
}
