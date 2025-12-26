import re
from _mapping import prompt_components_OSS, prompt_components_R1
import ast

def get_scaled_digits(digits):
    scaled_digits = []
    for i, digit in enumerate(digits):
        scaled_digit = digit * 10**(len(digits) - i - 1)
        scaled_digits.append(scaled_digit)
    return scaled_digits

def get_stepwise_reasoning_steps(num_1_num, num_2_scaled_digits):
    reasoning_steps = []
    sum = num_1_num

    for i, num_2_scaled_digit in enumerate(num_2_scaled_digits):
        next_sum = sum + num_2_scaled_digit
        reasoning_step = prompt_components_OSS["stepwise_reasoning_step"].format(intermediate_sum=sum, next_digit=num_2_scaled_digit, sum=next_sum)
        reasoning_steps.append(reasoning_step)
        sum = next_sum
    reasoning_steps_concat = ",".join(reasoning_steps)

    return reasoning_steps_concat

def get_stepwise_user_prompt(num1_num, num2_num):
    system_prompt = prompt_components_OSS["system"]
    developer_instructions = prompt_components_OSS["stepwise_developer"]
    user_prompt = prompt_components_OSS["user"].format(num1=num1_num, num2=num2_num)
    return system_prompt + developer_instructions + user_prompt

def get_stepwise_prompt(num_1_digits, num_2_digits, num_1_num, num_2_num):

    user_prompt = get_stepwise_user_prompt(num_1_num, num_2_num)
    analysis_prefix = prompt_components_OSS["analysis_prefix"]
    restatement = prompt_components_OSS["stepwise_restatement"].format(num1=num_1_num, num2=num_2_num)
    num_2_scaled_digits = get_scaled_digits(num_2_digits)
    num_2_components = " + ".join(map(str, num_2_scaled_digits))
    reasoning_prefix = prompt_components_OSS["stepwise_reasoning_prefix"].format(num1=num_1_num, num2=num_2_num, num2_components=num_2_components)
    reasoning_steps = get_stepwise_reasoning_steps(num_1_num, num_2_scaled_digits)
    reasoning_suffix = prompt_components_OSS["stepwise_reasoning_suffix"].format(sum=num_1_num + num_2_num)
    output_prefix = prompt_components_OSS["output_prefix"]

    return user_prompt + analysis_prefix + restatement + reasoning_prefix + reasoning_steps + reasoning_suffix + output_prefix

def get_R1_prompt(num_1_digits, num_2_digits, num_1_num, num_2_num):

    user_prompt = prompt_components_R1["user"].format(num1=num_1_num, num2=num_2_num)
    restatement = prompt_components_R1["restatement"].format(num1=num_1_num, num2=num_2_num)
    reasoning = prompt_components_R1["reasoning"].format(units1=num_1_digits[2], units2=num_2_digits[2], units_sum=num_1_digits[2] + num_2_digits[2], tens1=num_1_digits[1], tens2=num_2_digits[1], tens_sum=num_1_digits[1] + num_2_digits[1], hundreds1=num_1_digits[0], hundreds2=num_2_digits[0], hundreds_sum=num_1_digits[0] + num_2_digits[0])
    result = prompt_components_R1["result"].format(sum=num_1_num + num_2_num)
    output_prefix = prompt_components_R1["output_prefix"]

    return user_prompt + restatement + reasoning + result + output_prefix

def divide_prompt(intervention_id, prompt):
    
    # Find all numbers in the prompt using regex
    number_pattern = r'\d+'
    matches = list(re.finditer(number_pattern, prompt))
    
    if intervention_id >= len(matches):
        # If intervention_id is out of range, return the original prompt with empty splits
        return prompt, "", ""
    
    # Get the ith match (0-indexed)
    match = matches[intervention_id]
    start_pos = match.start()
    end_pos = match.end()
    
    # Split the prompt
    before = prompt[:start_pos]
    number = prompt[start_pos:end_pos]
    after = prompt[end_pos:]
    
    return before, number, after

def get_intervened_prompt(intervention_ids, base_prompt, source_prompt):
    if not intervention_ids:
        return base_prompt
    # Sort intervention_ids from lowest to highest number
    intervention_ids = sorted(intervention_ids)
    
    # Pop the highest number
    if intervention_ids:
        last_id = intervention_ids.pop()

    base_before, base_number, base_after = divide_prompt(last_id, base_prompt)
    source_before, source_number, source_after = divide_prompt(last_id, source_prompt)

    return get_intervened_prompt(intervention_ids, base_before, source_before) + source_number + base_after

def get_counterfactual_sum_OSS_3_digit(intervention_id, **row):
    '''
    Get the counterfactual sum for the given intervention id, base 1 number, base 2 digits, and source number.
    Args:
        intervention_id: The intervention id.
        base_1_num: The base 1 number.
        base_2_digits: The base 2 digits.
        source_number: The source number.
    Returns:
        The counterfactual sum.
    '''

    # Get the base and source numbers
    base_1_num = row['base_1_num']
    base_2_num = row['base_2_num']
    base_sum = row['base_sum']
    source_1_num = row['source_1_num']
    source_2_num = row['source_2_num']
    source_sum = row['source_sum']

    # Get the scaled digits
    base_2_scaled_digits = get_scaled_digits(ast.literal_eval(row['base_2_digits']))
    source_2_scaled_digits = get_scaled_digits(ast.literal_eval(row['source_2_digits']))

    if intervention_id == 7:
        return source_1_num + base_2_num
    elif intervention_id == 8:
        return base_1_num + source_2_num
    elif intervention_id == 9:
        return source_1_num + base_2_num
    elif intervention_id == 10:
        return base_1_num + source_2_num
    elif intervention_id == 11:
        return source_1_num + base_2_num
    elif intervention_id == 12:
        return base_1_num + source_2_num
    elif intervention_id == 13:
        return source_1_num + base_2_num
    elif intervention_id == 14:
        return base_1_num + source_2_scaled_digits[0] + base_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 15:
        return base_1_num + base_2_scaled_digits[0] + source_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 16:
        return base_1_num + base_2_scaled_digits[0] + base_2_scaled_digits[1]  + source_2_scaled_digits[2]
    elif intervention_id == 17:
        return source_1_num + base_2_num
    elif intervention_id == 18:
        return base_1_num + source_2_scaled_digits[0] + base_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 19:
        return source_1_num + source_2_scaled_digits[0] + base_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 20:
        return source_1_num + source_2_scaled_digits[0] + base_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 21:
        return base_1_num + base_2_scaled_digits[0] + source_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 22:
        return source_1_num + source_2_scaled_digits[0] + source_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 23:
        return source_1_num + source_2_scaled_digits[0] + source_2_scaled_digits[1]  + base_2_scaled_digits[2]
    elif intervention_id == 24:
        return base_1_num + base_2_scaled_digits[0] + base_2_scaled_digits[1]  + source_2_scaled_digits[2]
    elif intervention_id == 25:
        return source_sum
    elif intervention_id == 26:
        return source_sum
    elif intervention_id == 27:
        return source_sum

def get_counterfactual_sum_R1_3_digit(intervention_id, **row):
    '''
    Get the counterfactual sum for the given intervention id, base 1 number, base 2 digits, and source number.
    Args:
        intervention_id: The intervention id.
        base_1_num: The base 1 number.
        base_2_digits: The base 2 digits.
        source_number: The source number.
    Returns:
        The counterfactual sum.
    '''

    # Get the base and source numbers
    base_1_num = row['base_1_num']
    base_2_num = row['base_2_num']
    base_sum = row['base_sum']
    source_1_num = row['source_1_num']
    source_2_num = row['source_2_num']
    source_sum = row['source_sum']

    # Get the scaled digits
    base_1_scaled_digits = get_scaled_digits(ast.literal_eval(row['base_1_digits']))
    base_2_scaled_digits = get_scaled_digits(ast.literal_eval(row['base_2_digits']))
    source_1_scaled_digits = get_scaled_digits(ast.literal_eval(row['source_1_digits']))
    source_2_scaled_digits = get_scaled_digits(ast.literal_eval(row['source_2_digits']))

    if intervention_id == 3:
        return source_1_num + base_2_num
    elif intervention_id == 4:
        return base_1_num + source_2_num
    elif intervention_id == 5:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + source_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 6:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + base_1_scaled_digits[2] + source_2_scaled_digits[2]
    elif intervention_id == 7:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + source_1_scaled_digits[2] + source_2_scaled_digits[2]
    elif intervention_id == 8:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + source_1_scaled_digits[1] + base_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 9:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + base_1_scaled_digits[1] + source_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 10:
        return base_1_scaled_digits[0] + base_2_scaled_digits[0] + source_1_scaled_digits[1] + source_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 11:
        return source_1_scaled_digits[0] + base_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 12:
        return base_1_scaled_digits[0] + source_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 13:
        return source_1_scaled_digits[0] + source_2_scaled_digits[0] + base_1_scaled_digits[1] + base_2_scaled_digits[1] + base_1_scaled_digits[2] + base_2_scaled_digits[2]
    elif intervention_id == 14:
        return source_1_num + source_2_num