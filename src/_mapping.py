column_prompt_components = {
    "column_developer": "<|start|>developer<|message|># Instructions\nYou are a stepwise adder that reasons concisely. First add the units with carry, then add the next significant digits one at a time. Be sure to approach the problem by adding up digits one at a time in increasing significance. Once reasoning has concluded, give the user only the final answer.<|end|>",
    "column_reasoning_prefix": "So we need to do stepwise addition:",
    "column_reasoning_step": " {digit_name}: {num1_digit}+{num2_digit}={digit_sum}{prev_carry}.{write_and_carry}.",
    "prev_carry": " plus carry {carry} = {digit_sum}",
    "write_and_carry": " write {digit_sum} carry {carry}",
}

digit_num_to_name = {
    1: "units",
    2: "tens",
    3: "hundreds",
    4: "thousands",
    5: "ten thousands",
    6: "hundred thousands",
    7: "millions",
    8: "ten millions",
    9: "hundred millions",
    10: "billions",
}

name_to_digit_num = {v: k for k, v in digit_num_to_name.items()}

stepwise_prompt_components = {
    "stepwise_developer": "<|start|>developer<|message|># Instruction: In analysis, add two numbers stepwise. In final, output only the sum.<|end|>",
    "stepwise_restatement": '''The user asks: "What is {num1}+{num2}?" The instruction from the developer says: "In the analysis, add two numbers stepwise. For the final output, output only the sum." So we need to do the addition stepwise in the analysis, and then in the final output, just output the sum."''',
    "stepwise_reasoning_prefix": " So I need to do stepwise addition in analysis, then output only the sum in final. So in analysis, I will show the stepwise addition: {num1} + {num2}. Let's do it: {num1} + {num2} = {num1} + {num2_components} =",
    "stepwise_reasoning_step": " {intermediate_sum} + {next_digit} = {sum}",
    "stepwise_reasoning_suffix": ". So the sum is {sum}. In final, output only \"{sum}\"."
}

stepwise_intervene_loc_3_digit = {
    "all": list(range(5,28)),
    "user_question": [5,6],
    "restatement": [7,8,9,10],
    "reasoning": range(11,26),
    "result": [26,27],
    "copy": [7,8,9,10,11,12,13,17,20,23],
    "intermediate_sums": [19,22,25],
}

prompt_components = {
    "system": "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: high\n=\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
    "user": "<|start|>user<|message|>What is {num1}+{num2}?<|end|>",
    "analysis_prefix": "<|start|>assistant<|channel|>analysis<|message|>",
    "output_prefix": "<|end|><|start|>assistant<|channel|>final<|message|>",
    **column_prompt_components,
    **stepwise_prompt_components,
}