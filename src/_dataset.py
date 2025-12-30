import random
random.seed(42)
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