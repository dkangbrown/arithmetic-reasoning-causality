import random
random.seed(42)

def create_dataset(num_digits=2, num_samples=128):

    add_ds = []
    for _ in range(num_samples):
        base_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        base_2 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        source_1 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        source_2 = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(num_digits-1)]
        
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
            "source_1_digits": source_1,
            "source_2_digits": source_2,
            "source_1_num": source_1_num,
            "source_2_num": source_2_num,
        })

    return add_ds