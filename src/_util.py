import os
import torch
from transformers import Mxfp4Config, AutoModelForCausalLM, AutoTokenizer
import csv
import re

def load_OSS(model_id="openai/gpt-oss-20b", path="/users/dkang33/scratch/model_cache", device="auto"):
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map=device,
        cache_dir=path,
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=path)
    tokenizer.pad_token_id = 19742
    model.config.pad_token_id = 19742

    return model, tokenizer

def load_R1(model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", path="/users/dkang33/scratch/model_cache", device="auto", dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=path, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=path)
    tokenizer.pad_token_id = 93
    model.config.pad_token_id = 93

    return model, tokenizer

def create_csv_file(dir, filename, header, overwrite=False):
    '''
    Create a CSV file with the given header.
    Args:
        dir: The directory to create the file in.
        filename: The name of the file to create.
        header: The header of the file.
    Returns:
        The filepath of the created file.
    '''
    filepath = os.path.join(dir, filename)
    
    if not overwrite:
        # If file exists, append _1, _2, _3, etc. until we find a non-existing filepath
        counter = 1
        base_filepath = filepath
        while os.path.exists(filepath):
            name, ext = os.path.splitext(base_filepath)
            filepath = f"{name}_{counter}{ext}"
            counter += 1

    # Write header to file
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    return filepath

def write_to_csv(filepath, data):
    '''
    Write data to a CSV file.
    '''
    with open(filepath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def lookup_module(model, hook_name):
    if hook_name == '':
        return model
    d = dict(model.named_modules())
    return d[hook_name] if hook_name in d else None

def get_final_output(generated_text, output_type="immediate", intervene_id=None):
    '''
    Get the final output from the generated output.
    '''
    if output_type == "immediate":
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            return numbers[0]
        else:
            raise ValueError("No number found in the generated text")
    elif output_type == "final_output":
        if "<|channel|>final<|message|>" in generated_text:
            final_channel = generated_text.split('<|start|>assistant<|channel|>final<|message|>')[-1]
            # Extract the first occurring number from final_channel
            numbers = re.findall(r'\d+', final_channel)
            if numbers:
                return numbers[0]
            else:
                raise ValueError("No number found in the final channel")
        elif "The answer is " in generated_text:
            answer = generated_text.split("The answer is ")[-1]
            # Extract the first occurring number from the answer
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return numbers[0]
            else:
                raise ValueError("No number found in the answer")
        else:
            raise ValueError(f"No final output found in the generated text, for intervene_id: {intervene_id}")
    else:
        raise ValueError("Invalid output type")

def print_GPU_availbility():
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("Available devices:")
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {device_name}")
        print(torch.cuda.memory_summary(device=torch.device(f"cuda:{i}"), abbreviated=False))
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device count: {torch.cuda.device_count()}")