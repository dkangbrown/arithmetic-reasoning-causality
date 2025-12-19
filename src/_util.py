import torch
from transformers import Mxfp4Config, AutoModelForCausalLM, AutoTokenizer

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
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_deepseek(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", path="/users/dkang33/scratch/model_cache", device="auto"):
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=path)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=path)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer