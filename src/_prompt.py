import re
from _mapping import prompt_components

def get_scaled_digits(digits):
    scaled_digits = []
    for i, digit in enumerate(digits):
        if digit == 0:
            continue
        scaled_digit = digit * 10**(len(digits) - i - 1)
        scaled_digits.append(scaled_digit)
    return scaled_digits

def get_stepwise_reasoning_steps(num_1_num, num_2_scaled_digits):
    reasoning_steps = []
    sum = num_1_num

    for i, num_2_scaled_digit in enumerate(num_2_scaled_digits):
        next_sum = sum + num_2_scaled_digit
        reasoning_step = prompt_components["stepwise_reasoning_step"].format(intermediate_sum=sum, next_digit=num_2_scaled_digit, sum=next_sum)
        reasoning_steps.append(reasoning_step)
        sum = next_sum
    reasoning_steps_concat = ",".join(reasoning_steps)

    return reasoning_steps_concat

def get_stepwise_user_prompt(num1_num, num2_num):
    system_prompt = prompt_components["system"]
    developer_instructions = prompt_components["stepwise_developer"]
    user_prompt = prompt_components["user"].format(num1=num1_num, num2=num2_num)
    return system_prompt + developer_instructions + user_prompt

def get_stepwise_prompt(num_2_digits, num_1_num, num_2_num):

    user_prompt = get_stepwise_user_prompt(num_1_num, num_2_num)
    analysis_prefix = prompt_components["analysis_prefix"]
    restatement = prompt_components["stepwise_restatement"].format(num1=num_1_num, num2=num_2_num)
    num_2_scaled_digits = get_scaled_digits(num_2_digits)
    num_2_components = " + ".join(map(str, num_2_scaled_digits))
    reasoning_prefix = prompt_components["stepwise_reasoning_prefix"].format(num1=num_1_num, num2=num_2_num, num2_components=num_2_components)
    reasoning_steps = get_stepwise_reasoning_steps(num_1_num, num_2_scaled_digits)
    reasoning_suffix = prompt_components["stepwise_reasoning_suffix"].format(sum=num_1_num + num_2_num)
    output_prefix = prompt_components["output_prefix"]

    return user_prompt + analysis_prefix + restatement + reasoning_prefix + reasoning_steps + reasoning_suffix + output_prefix

def divide_prompt(intervention_id, prompt):
    
    # Find all numbers in the prompt using regex
    number_pattern = r'\d+'
    matches = list(re.finditer(number_pattern, prompt))
    print(matches)
    
    if intervention_id >= len(matches):
        # If intervention_id is out of range, return the original prompt with empty splits
        return prompt, "", ""
    
    # Get the ith match (0-indexed)
    match = matches[intervention_id]
    print(match)
    start_pos = match.start()
    end_pos = match.end()
    
    # Split the prompt
    before = prompt[:start_pos]
    number = prompt[start_pos:end_pos]
    after = prompt[end_pos:]
    
    return before, number, after

