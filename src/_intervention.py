import torch
from transformers import Mxfp4Config, AutoModelForCausalLM, AutoTokenizer
import gc
from typing import Any, Callable, Tuple, List, Dict

def forward_with_cache(model, input_ids, module_names=None, pre_hook=True, **kwargs):
    '''
    Forward pass through the model and cache the activations of the specified modules.
    Args:
        model: The model to forward pass through.
        input_ids: The input ids to forward pass through the model.
        modules: The modules to cache the activations of. If None, all layers will be cached.
    Returns:
        The output of the model.
    '''
    # Initialize the activations dictionary
    activations = {}
    hooks = []
    # Define the cache hook
    def get_cache_hook(module_name):
        def cache_hook(module, input):
            activations[module_name] = input[0]
        return cache_hook
    
    # If no modules are specified, cache all residual stream representations
    if module_names == None:
        module_names = []
        for layer in range(len(model.model.layers)):
            module_names.append(f"model.layers.{layer}")

    # Register the cache hook
    if pre_hook:
        for name, module in model.named_modules():
            if name not in module_names:
                continue
            cache_hook_handle = module.register_forward_pre_hook(get_cache_hook(name))
            hooks.append(cache_hook_handle)
    else:
        for name, module in model.named_modules():
            if name not in module_names:
                continue
            cache_hook_handle = module.register_forward_hook(get_cache_hook(name))
            hooks.append(cache_hook_handle)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_ids=input_ids, **kwargs)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    # Return the output and the activations
    return output, activations

def get_batch_intervene_hook(activation):
    def intervene_hook(module, input):
        """
        A forward pre-hook function to inspect and modify the input arguments.
        `module`: The module to which the hook is attached.
        `args`: A tuple containing the input tensors for the module's forward method.
        
        Returns:
            Modified input for the module's forward pass.
        """
        input = (activation,)
        return input
    return intervene_hook

def get_batch_token_intervene_hook(activations, tok_locs, num_toks=1):
    assert len(activations) == len(tok_locs)
    def intervene_hook(module, input):
        """
        A forward pre-hook function to inspect and modify the input arguments.
        `module`: The module to which the hook is attached.
        `args`: A tuple containing the input tensors for the module's forward method.
        
        Returns:
            Modified input for the module's forward pass.
        """
        for i, (activation, tok_loc) in enumerate(zip(activations, tok_locs)):
            input[0][i, tok_loc-num_toks+1:tok_loc+1] = activation
        return input
    return intervene_hook

def get_batch_multitoken_intervene_hook(activation, tok_pos_list):
    def intervene_hook(module, input):
        """
        A forward pre-hook function to inspect and modify the input arguments.
        `module`: The module to which the hook is attached.
        `args`: A tuple containing the input tensors for the module's forward method.
        
        Returns:
            Modified input for the module's forward pass.
        """
        input[0][:, tok_pos_list] = activation[:, tok_pos_list]
        return input
    return intervene_hook

def get_attention_freeze_hooks(model, tokens):
    module_names = [f"model.layers.{i}.self_attn.q_proj" for i in range(len(model.model.layers))]\
                 + [f"model.layers.{i}.self_attn.k_proj" for i in range(len(model.model.layers))]
    _, cache = forward_with_cache(model, tokens["input_ids"], attention_mask=tokens["attention_mask"], module_names=module_names)
    
    hooks = []
    for module_name in module_names:
        hook = {module_name: get_batch_intervene_hook(cache[module_name])}
        hooks.append(hook)
    return hooks

def prepare_batch_token_intervention(model, tokenizer, layer, base_before, base_number, base_after, source_before, source_number, module_format="model.layers.{layer}", num_toks=1):
    """
    Prepare the batch for token activation intervention.
    """
    source = [source_before[i] + str(source_number[i]) for i in range(len(source_before))]
    source_tokens = tokenizer(source, add_special_tokens=False, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    _, cache = forward_with_cache(model, source_tokens["input_ids"], attention_mask=source_tokens["attention_mask"])
    activations = list(cache[f"model.layers.{layer}"][:,-num_toks:,:])

    offsets = []
    for after in base_after:
        offsets.append(len(tokenizer(after, add_special_tokens=False, return_tensors="pt")["input_ids"][0]))

    base = [base_before[i] + str(base_number[i]) + base_after[i] for i in range(len(base_before))]
    tokens = tokenizer(base, add_special_tokens=False, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    seq_len = tokens["input_ids"].shape[1]
    tok_locs = [seq_len - offset - 1 for offset in offsets]

    hook = {module_format.format(layer=layer): get_batch_token_intervene_hook(activations, tok_locs, num_toks=num_toks)}

    del cache, activations, tok_locs
    torch.cuda.empty_cache()
    gc.collect()
    return tokens, hook

def prepare_batch_multitoken_intervention(model, tokenizer, layer, tok_pos_list, base_prompts, source_prompts, module_format="model.layers.{layer}"):
    """
    Prepare the batch for token activation intervention.
    """
    base_tokens = tokenizer(base_prompts, add_special_tokens=False, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    source_tokens = tokenizer(source_prompts, add_special_tokens=False, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    # print(tokenizer.convert_ids_to_tokens(base_tokens["input_ids"][0])[112])
    # print(tokenizer.convert_ids_to_tokens(source_tokens["input_ids"][0])[112])
    # print(tokenizer.convert_ids_to_tokens(base_tokens["input_ids"][0])[202])
    # print(tokenizer.convert_ids_to_tokens(source_tokens["input_ids"][0])[202])
    # print(tokenizer.convert_ids_to_tokens(base_tokens["input_ids"][0])[212])
    # print(tokenizer.convert_ids_to_tokens(source_tokens["input_ids"][0])[212])
    _, cache = forward_with_cache(model, source_tokens["input_ids"], attention_mask=source_tokens["attention_mask"])
    activation = cache[f"model.layers.{layer}"]
    hook = {module_format.format(layer=layer): get_batch_multitoken_intervene_hook(activation, tok_pos_list)}

    del cache, activation
    torch.cuda.empty_cache()
    gc.collect()
    return base_tokens, hook

def batch_intervene(model, input_ids, hooks, **kwargs):
    """
    Intervene on a batch of inputs.
    """
    hook_handles = []
    for hook in hooks:
        name, hook_fn = next(iter(hook.items()))
        hook_handle = dict(model.named_modules())[name].register_forward_pre_hook(hook_fn)
        hook_handles.append(hook_handle)
    with torch.no_grad():
        output = model(input_ids=input_ids, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return output

def get_intervened_label_probability(
    model: AutoModelForCausalLM,
    tokens: Dict[str, torch.Tensor],
    intervene_hooks: List[Dict[str, Callable]],
    labels: torch.Tensor, 
    labels_mask: torch.Tensor,
    attention_freeze = False,
    **kwargs
) -> torch.Tensor:
    """
    Calculate the probability of the label tokens given the input tokens and hooks.
    
    Args:
        model: The model to calculate the probability of the label tokens given the input tokens and hooks.
        tokens: The input token ids and attention mask.
        intervene_hooks: The hooks to intervene on the model.
        labels: The expected label token ids.
        labels_mask: Mask for valid label positions.
        attention_freeze: Whether to freeze the attention.
        
    Returns:
        Label probabilities.
    """
    labels_probs = torch.ones(labels.shape[0]).to(model.cfg.device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    for i in range(labels.shape[1]):
        if attention_freeze:
            attention_freeze_hooks = get_attention_freeze_hooks(model, tokens)
        else:
            attention_freeze_hooks = []
        with torch.no_grad():
            output = batch_intervene(model, input_ids, attention_freeze_hooks + intervene_hooks, attention_mask=attention_mask, **kwargs)
        
        # Calculate probability of first answer token at position -1 (after "is")
        probs = torch.softmax(output.logits[:, -1, :], dim=-1)
        label_tokens = labels[:, i]
        label_probs = probs[torch.arange(probs.shape[0]), label_tokens]
        label_probs = label_probs * labels_mask[:, i] + (~labels_mask[:, i])
        labels_probs = labels_probs * label_probs

        # update prompt tokens
        input_ids = torch.cat([input_ids, label_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1), True).to(model.cfg.device)], dim=1)

    return labels_probs

def get_label_probability(
    model: AutoModelForCausalLM,
    tokens: Dict[str, torch.Tensor],
    labels: torch.Tensor, 
    labels_mask: torch.Tensor,
    **kwargs
) -> torch.Tensor:
    """
    Calculate the probability of the label tokens given the input tokens and hooks.
    
    Args:
        model: The model to calculate the probability of the label tokens given the input tokens and hooks.
        tokens: The input token ids and attention mask.
        labels: The expected label token ids.
        labels_mask: Mask for valid label positions.
        
    Returns:
        Label probabilities.
    """
    labels_probs = torch.ones(labels.shape[0]).to(model.device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    for i in range(labels.shape[1]):
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Calculate probability of first answer token at position -1 (after "is")
        probs = torch.softmax(output.logits[:, -1, :], dim=-1)
        label_tokens = labels[:, i]
        label_probs = probs[torch.arange(probs.shape[0]), label_tokens]
        label_probs = label_probs * labels_mask[:, i] + (~labels_mask[:, i])
        labels_probs = labels_probs * label_probs

        # update prompt tokens
        input_ids = torch.cat([input_ids, label_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1), True).to(model.device)], dim=1)

    return labels_probs