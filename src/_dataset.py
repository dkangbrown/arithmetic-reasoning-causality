"""
Helper functions related to creating datasets in create_dataset_csv.ipynb.
"""

import random
random.seed(42)
import pandas as pd
import _prompt

def create_dataset(num_digits=3, num_samples=128):

    add_ds = []
    for _ in range(num_samples):
        base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        base_2 = [random.randint(1, 9) for _ in range(num_digits)]
        source_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        source_2 = [random.randint(1, 9) for _ in range(num_digits)]
        
        base_1_num, base_2_num, source_1_num, source_2_num = 0, 0, 0, 0
        for i in range(num_digits):
            base_1_num = 10 * base_1_num + base_1[i]
            base_2_num = 10 * base_2_num + base_2[i]
            source_1_num = 10 * source_1_num + source_1[i]
            source_2_num = 10 * source_2_num + source_2[i]
        
        base_sum = base_1_num + base_2_num
        source_sum = source_1_num + source_2_num

        add_ds.append({
            "base_1_digits": base_1,
            "base_2_digits": base_2,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
        })

    return add_ds

def create_h_dataset(num_digits=3, num_samples=256):

    add_ds = []
    for _ in range(num_samples):
        retry = True
        while retry:
            base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
            base_2 = [random.randint(1, 9) for _ in range(num_digits)]
            source_1 = base_1.copy()
            source_2 = []
            for i in range(len(base_2)):
                if i == 0:
                    source_2.append(random.choice([x for x in range(1,10) if x != base_2[i]]))
                else:
                    source_2.append(base_2[i])
            
            base_1_scaled_digits = _prompt.get_scaled_digits(base_1)
            base_2_scaled_digits = _prompt.get_scaled_digits(base_2)
            source_1_scaled_digits = _prompt.get_scaled_digits(source_1)
            source_2_scaled_digits = _prompt.get_scaled_digits(source_2)

            base_1_num = sum(base_1_scaled_digits)
            base_2_num = sum(base_2_scaled_digits)
            source_1_num = sum(source_1_scaled_digits)
            source_2_num = sum(source_2_scaled_digits)

            # base_intermediate_sums = []
            # base_intermediate_sum = base_1_num
            # for source_scaled_digit in source_2_scaled_digits:
            #     base_intermediate_sum += source_scaled_digit
            #     base_intermediate_sums.append(base_intermediate_sum)
            
            # source_intermediate_sums = []
            # source_intermediate_sum = source_1_num
            # for base_scaled_digit in base_2_scaled_digits:
            #     source_intermediate_sum += base_scaled_digit
            #     source_intermediate_sums.append(source_intermediate_sum)
            
            base_sum = base_1_num + base_2_num
            source_sum = source_1_num + source_2_num

            retry = len(str(base_sum)) != num_digits or len(str(source_sum)) != num_digits
            
            # retry = False
            # for base_intermediate_sum, source_intermediate_sum in zip(base_intermediate_sums, source_intermediate_sums):
            #     if len(str(base_intermediate_sum)) != len(str(source_intermediate_sum)):
            #         retry = True

        add_ds.append({
            "base_1_digits": base_1,
            "base_2_digits": base_2,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
        })

    return add_ds

def create_h1_dataset(num_digits=3, num_samples=256):

    add_ds = []
    for _ in range(num_samples):
        retry = True
        while retry:
            base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
            base_2 = [random.randint(1, 9) for _ in range(num_digits)]
            source_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
            source_2 = [random.randint(1, 9) for _ in range(num_digits)]
            
            base_1_scaled_digits = _prompt.get_scaled_digits(base_1)
            base_2_scaled_digits = _prompt.get_scaled_digits(base_2)
            source_1_scaled_digits = _prompt.get_scaled_digits(source_1)
            source_2_scaled_digits = _prompt.get_scaled_digits(source_2)

            base_1_num = sum(base_1_scaled_digits)
            base_2_num = sum(base_2_scaled_digits)
            source_1_num = sum(source_1_scaled_digits)
            source_2_num = sum(source_2_scaled_digits)

            # base_intermediate_sums = []
            # base_intermediate_sum = base_1_num
            # for source_scaled_digit in source_2_scaled_digits:
            #     base_intermediate_sum += source_scaled_digit
            #     base_intermediate_sums.append(base_intermediate_sum)
            
            # source_intermediate_sums = []
            # source_intermediate_sum = source_1_num
            # for base_scaled_digit in base_2_scaled_digits:
            #     source_intermediate_sum += base_scaled_digit
            #     source_intermediate_sums.append(source_intermediate_sum)
            
            base_sum = base_1_num + base_2_num
            source_sum = source_1_num + source_2_num

            retry = len(str(base_sum)) != num_digits or len(str(source_sum)) != num_digits
            
            # retry = False
            # for base_intermediate_sum, source_intermediate_sum in zip(base_intermediate_sums, source_intermediate_sums):
            #     if len(str(base_intermediate_sum)) != len(str(source_intermediate_sum)):
            #         retry = True

        add_ds.append({
            "base_1_digits": base_1,
            "base_2_digits": base_2,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
        })

    return add_ds

def create_h2_dataset(num_digits=3, num_samples=256):

    add_ds = []
    for _ in range(num_samples):
        
        base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        base_2 = [random.randint(1, 9) for _ in range(num_digits)]
        source_1 = base_1.copy()
        source_2 = []
        for i in range(len(base_2)):
            if i == 0:
                source_2.append(random.choice([x for x in range(1,10) if x != base_2[i]]))
            else:
                source_2.append(base_2[i])
        
        base_1_scaled_digits = _prompt.get_scaled_digits(base_1)
        base_2_scaled_digits = _prompt.get_scaled_digits(base_2)
        source_1_scaled_digits = _prompt.get_scaled_digits(source_1)
        source_2_scaled_digits = _prompt.get_scaled_digits(source_2)

        base_1_num = sum(base_1_scaled_digits)
        base_2_num = sum(base_2_scaled_digits)
        source_1_num = sum(source_1_scaled_digits)
        source_2_num = sum(source_2_scaled_digits)
        
        base_sum = base_1_num + base_2_num
        source_sum = source_1_num + source_2_num

        add_ds.append({
            "base_1_digits": base_1,
            "base_2_digits": base_2,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
        })

    return add_ds

def create_c_dataset(num_digits=3, num_samples=256):

    add_ds = []
    for _ in range(num_samples):
        retry = True
        while retry:
            if num_digits >= 3:
                base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-3)] + [random.randint(1, 9), random.randint(0, 8)]
            else:
                base_1 = [random.randint(1, 9), random.randint(0, 8)]
            base_2 = []
            for i in range(num_digits):
                if i == 0:
                    base_2.append(random.randint(1, 9))
                elif i == num_digits - 1:
                    base_2.append(random.choice([x for x in range(0,10) if base_1[i] + x < 10]))
                elif i == num_digits - 2:
                    base_2.append(random.choice([x for x in range(0,10) if base_1[i] + x >= 10]))
                else:
                    base_2.append(random.choice([x for x in range(0,10) if base_1[i] + x + 1 >= 10]))
            source_1 = base_1.copy()
            source_2 = []
            for i in range(num_digits):
                if i == num_digits - 1:
                    source_2.append(random.choice([x for x in range(0,10) if base_1[i] + x < 10 and base_2[i] != x]))
                else:
                    source_2.append(base_2[i])
            
            base_1_scaled_digits = _prompt.get_scaled_digits(base_1)
            base_2_scaled_digits = _prompt.get_scaled_digits(base_2)
            source_1_scaled_digits = _prompt.get_scaled_digits(source_1)
            source_2_scaled_digits = _prompt.get_scaled_digits(source_2)

            base_1_num = sum(base_1_scaled_digits)
            base_2_num = sum(base_2_scaled_digits)
            source_1_num = sum(source_1_scaled_digits)
            source_2_num = sum(source_2_scaled_digits)
            
            base_sum = base_1_num + base_2_num
            source_sum = source_1_num + source_2_num

            retry = len(str(base_sum)) != num_digits or len(str(source_sum)) != num_digits

        add_ds.append({
            "base_1_digits": base_1,
            "base_2_digits": base_2,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
        })

    return add_ds


def create_prompt_dataframe(add_ds, get_prompt_fn, divide_num=100):
    prompt_data = []

    for add_ds_entry in add_ds:
        base_1_digits = add_ds_entry["base_1_digits"]
        base_2_digits = add_ds_entry["base_2_digits"]
        base_1_num = add_ds_entry["base_1_num"]
        base_2_num = add_ds_entry["base_2_num"]
        base_sum = add_ds_entry["base_sum"]
        source_1_digits = add_ds_entry["source_1_digits"]
        source_2_digits = add_ds_entry["source_2_digits"]
        source_1_num = add_ds_entry["source_1_num"]
        source_2_num = add_ds_entry["source_2_num"]
        source_sum = add_ds_entry["source_sum"]

        base_prompt = get_prompt_fn(base_1_digits, base_2_digits, base_1_num, base_2_num)
        truncated_base_prompt, factual_output, _ = _prompt.divide_prompt(divide_num, base_prompt)
        source_prompt = get_prompt_fn(source_1_digits, source_2_digits, source_1_num, source_2_num)
        truncated_source_prompt, counterfactual_output, _ = _prompt.divide_prompt(divide_num, source_prompt)

        prompt_data.append({
            "base_1_digits": base_1_digits,
            "base_2_digits": base_2_digits,
            "base_1_num": base_1_num,
            "base_2_num": base_2_num,
            "base_sum": base_sum,
            "source_1_digits": source_1_digits,
            "source_2_digits": source_2_digits,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
            "source_sum": source_sum,
            "base_prompt": truncated_base_prompt,
            "source_prompt": truncated_source_prompt,
            "factual_output": factual_output,
            "counterfactual_output": counterfactual_output,
        })

    return pd.DataFrame(prompt_data)


def divide_prompts(
    prompt_data,
    intervention_ids_dict,
    get_counterfactual_sum_fn,
):
    divided_prompts = []

    for _, row in prompt_data.iterrows():
        base_prompt = row["base_prompt"]
        source_prompt = row["source_prompt"]
        counterfactual_row = row.to_dict()
        for digit_column in (
            "base_1_digits",
            "base_2_digits",
            "source_1_digits",
            "source_2_digits",
        ):
            if isinstance(counterfactual_row.get(digit_column), list):
                counterfactual_row[digit_column] = repr(counterfactual_row[digit_column])

        for intervention_id in intervention_ids_dict["all"]:
            base_before, base_number, base_after = _prompt.divide_prompt(intervention_id, base_prompt)
            source_before, source_number, source_after = _prompt.divide_prompt(intervention_id, source_prompt)
            if base_after == "" or source_after == "":
                break
            counterfactual_sum = get_counterfactual_sum_fn(intervention_id, **counterfactual_row)

            divided_prompts.append({
                "base_1_digits": row["base_1_digits"],
                "base_2_digits": row["base_2_digits"],
                "base_1_num": row["base_1_num"],
                "base_2_num": row["base_2_num"],
                "base_sum": row["base_sum"],
                "source_1_digits": row["source_1_digits"],
                "source_2_digits": row["source_2_digits"],
                "source_1_num": row["source_1_num"],
                "source_2_num": row["source_2_num"],
                "source_sum": row["source_sum"],
                "intervention_id": intervention_id,
                "base_before": base_before,
                "base_number": base_number,
                "base_after": base_after,
                "source_before": source_before,
                "source_number": source_number,
                "source_after": source_after,
                "counterfactual_sum": counterfactual_sum,
                "in_restatement": intervention_id in intervention_ids_dict["restatement"],
                "in_reasoning": intervention_id in intervention_ids_dict["reasoning"],
                "in_result": intervention_id in intervention_ids_dict["result"],
            })

    return pd.DataFrame(divided_prompts)