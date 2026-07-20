from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class PromptConfig:
    model_type: str
    prompt_type: str = ""
    data_root: str = "data"

    @property
    def suffix(self) -> str:
        if not self.prompt_type:
            return ""
        return self.prompt_type if self.prompt_type.startswith("_") else f"_{self.prompt_type}"

    @property
    def stem(self) -> str:
        return self.suffix[1:] if self.suffix else ""

    @property
    def is_h_prompt(self) -> bool:
        return "h" in self.suffix

    def filename(self, kind: Literal["prompts", "divided_prompts"] = "prompts") -> str:
        return f"{kind}{self.suffix}.csv"

    def divided_filename(self) -> str:
        return self.filename("divided_prompts")

    def filepath(self, kind: Literal["prompts", "divided_prompts"] = "prompts") -> Path:
        filename = self.filename(kind)
        return repo_path(Path(self.data_root) / self.model_type / filename)

    def divided_filepath(self) -> Path:
        return self.filepath("divided_prompts")


@dataclass(frozen=True)
class InterventionConfig:
    intervention_loc: str
    intervention_ids: Optional[Sequence[int]] = None
    tok_pos_fn: Optional[Dict[int, int]] = None
    layers: Optional[Sequence[int]] = None
    module_format: str = "model.layers.{layer}"
    pre_hook: bool = True

    @property
    def location_suffix(self) -> str:
        if not self.intervention_loc:
            return ""
        return self.intervention_loc if self.intervention_loc.startswith("_") else f"_{self.intervention_loc}"


@dataclass(frozen=True)
class RunConfig:
    experiment_root: str = "experiments/activation_intervention"
    result_dir: str = ""
    batch_size: int = 24
    freeze_batch_size: int = 1
    overwrite: bool = False
    filename_order: Literal["prompt_location", "location_prompt"] = "prompt_location"
    use_prefixed_location_in_filename: bool = False
    output_filename: Optional[str] = None

    def resolved_batch_size(self, freeze_attention: bool) -> int:
        return self.freeze_batch_size if freeze_attention else self.batch_size


@dataclass(frozen=True)
class AttentionFreezeConfig:
    enabled: bool = False
    num_attention: int = 20
    dataset_fn: Optional[Callable] = None
    num_digits: int = 3
    prompt_fn: Optional[Callable] = None
    divide_num: int = 22
    seed: int = 42
    num_base_samples: int = 256

    @property
    def num_samples(self) -> int:
        return self.num_base_samples + self.num_attention


def load_model(model_type: str):
    import _util

    if "GPT-OSS" in model_type:
        return _util.load_OSS()
    if "R1" in model_type:
        return _util.load_R1()
    raise ValueError(f"Invalid model type: {model_type}")


def load_prompts(prompt_config: PromptConfig) -> pd.DataFrame:
    prompts = pd.read_csv(prompt_config.filepath())
    for column in ("base_sum", "source_sum"):
        if column in prompts.columns:
            prompts[column] = prompts[column].astype("Int64")
    return prompts


def load_divided_prompts(prompt_config: PromptConfig) -> pd.DataFrame:
    divided_prompts = pd.read_csv(prompt_config.filepath("divided_prompts"))
    for column in ("base_number", "source_number"):
        if column in divided_prompts.columns:
            divided_prompts[column] = divided_prompts[column].astype("Int64")
    return divided_prompts


def resolve_intervention_ids(
    model_type: str,
    prompt_type: str,
    intervention_loc: str,
    intervention_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    if intervention_ids is not None:
        return list(intervention_ids)

    import _mapping

    prompt_config = PromptConfig(model_type=model_type, prompt_type=prompt_type)
    if prompt_config.is_h_prompt:
        if model_type == "GPT-OSS_stepwise":
            intervention_ids_dict = _mapping.intervene_ids_stepwise_3_digit_h
        elif model_type == "GPT-OSS_vanilla":
            intervention_ids_dict = _mapping.intervene_ids_vanilla_2_digit_h
        elif model_type == "R1":
            intervention_ids_dict = _mapping.intervene_ids_R1_3_digit_h
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    else:
        if model_type == "GPT-OSS_stepwise":
            intervention_ids_dict = _mapping.intervene_ids_stepwise_3_digit
        elif model_type == "R1":
            intervention_ids_dict = _mapping.intervene_ids_R1_3_digit
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    if intervention_loc == "restatement":
        return list(intervention_ids_dict["restatement"])
    if intervention_loc == "reasoning":
        return list(intervention_ids_dict["reasoning"])
    if intervention_loc == "restatement_and_reasoning":
        return list(intervention_ids_dict["restatement"] + intervention_ids_dict["reasoning"])
    raise ValueError(f"Invalid intervention location: {intervention_loc}")


def build_tok_pos_list(tok_pos_fn: Dict[int, int], intervention_ids: Sequence[int]) -> List[int]:
    return [tok_pos_fn[intervention_id] for intervention_id in intervention_ids]


def build_output_filename(
    prompt_config: PromptConfig,
    intervention_config: InterventionConfig,
    run_config: RunConfig,
) -> str:
    if run_config.output_filename is not None:
        return run_config.output_filename

    prompt_part = prompt_config.stem
    location_part = (
        intervention_config.location_suffix
        if run_config.use_prefixed_location_in_filename
        else intervention_config.intervention_loc
    )
    if run_config.filename_order == "location_prompt":
        filename_stem = "_".join(part for part in (location_part, prompt_part) if part)
    else:
        filename_stem = "_".join(part for part in (prompt_part, location_part) if part)
    return f"{filename_stem}.csv"


def build_output_filepath(
    prompt_config: PromptConfig,
    intervention_config: InterventionConfig,
    run_config: RunConfig,
    header: Sequence[str],
):
    import _util

    output_dir = repo_path(Path(run_config.experiment_root)) / "output" / prompt_config.model_type / run_config.result_dir
    filename = build_output_filename(prompt_config, intervention_config, run_config)
    return _util.create_csv_file(str(output_dir), filename, header, overwrite=run_config.overwrite)


def build_run_output_path(
    prompt_config: PromptConfig,
    run_config: RunConfig,
) -> Path:
    if run_config.output_filename is None:
        raise ValueError("output_filename is required for generic run output paths")
    return (
        repo_path(Path(run_config.experiment_root))
        / "output"
        / prompt_config.model_type
        / run_config.result_dir
        / run_config.output_filename
    )


def build_run_output_filepath(
    prompt_config: PromptConfig,
    run_config: RunConfig,
    header: Sequence[str],
):
    import _util

    if run_config.output_filename is None:
        raise ValueError("output_filename is required for generic run output paths")
    output_dir = repo_path(Path(run_config.experiment_root)) / "output" / prompt_config.model_type / run_config.result_dir
    return _util.create_csv_file(str(output_dir), run_config.output_filename, header, overwrite=run_config.overwrite)


def write_to_csv(filepath, data) -> None:
    import _util

    _util.write_to_csv(filepath, data)


def prepare_label_tensors(tokenizer, batch_rows: pd.DataFrame):
    if pd.notna(batch_rows.iloc[0]["factual_output"]) and batch_rows.iloc[0]["factual_output"]:
        factual_labels_str = [str(value) for value in batch_rows["factual_output"].tolist()]
    else:
        factual_labels_str = [str(value) for value in batch_rows["base_sum"].tolist()]
    factual_labels = tokenizer(factual_labels_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
    factual_labels = factual_labels.squeeze(1)

    if pd.notna(batch_rows.iloc[0]["counterfactual_output"]) and batch_rows.iloc[0]["counterfactual_output"]:
        counterfactual_labels_str = [str(value) for value in batch_rows["counterfactual_output"].tolist()]
    else:
        counterfactual_labels_str = [str(value) for value in batch_rows["source_sum"].tolist()]
    counterfactual_labels = tokenizer(counterfactual_labels_str, add_special_tokens=False, return_tensors="pt")["input_ids"]
    counterfactual_labels = counterfactual_labels.squeeze(1)

    return factual_labels, counterfactual_labels


def prepare_sequence_label_tokens(tokenizer, labels: Sequence, device=None):
    label_tokens = tokenizer(
        [str(label) for label in labels],
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        padding_side="right",
    )["input_ids"]
    if device is not None:
        label_tokens = label_tokens.to(device)
    return label_tokens, label_tokens != tokenizer.pad_token_id


def build_intervened_prompts(batch_rows: pd.DataFrame, intervention_ids: Sequence[int]) -> List[str]:
    import _prompt

    return [
        _prompt.get_intervened_prompt(intervention_ids, row["base_prompt"], row["source_prompt"])
        for _, row in batch_rows.iterrows()
    ]


def build_number_prompts(
    batch_rows: pd.DataFrame,
    number_column: str,
    before_column: str = "base_before",
    after_column: Optional[str] = None,
) -> List[str]:
    prompts = []
    for _, row in batch_rows.iterrows():
        prompt = row[before_column] + str(row[number_column])
        if after_column is not None:
            prompt += row[after_column]
        prompts.append(prompt)
    return prompts


def build_attention_freeze_hooks(
    model,
    tokenizer,
    attention_config: AttentionFreezeConfig,
    modifier_fn: Optional[Callable] = None,
):
    if not attention_config.enabled:
        return []
    if attention_config.dataset_fn is None or attention_config.prompt_fn is None:
        raise ValueError("dataset_fn and prompt_fn are required when attention freeze is enabled")

    import random

    import _prompt
    from _intervention import get_attention_freeze_hooks

    modifier_fn = modifier_fn or (lambda prompt, add_ds_entry: prompt)
    random.seed(attention_config.seed)
    add_ds = attention_config.dataset_fn(
        num_digits=attention_config.num_digits,
        num_samples=attention_config.num_samples,
    )

    attention_prompts = []
    for add_ds_entry in add_ds[-attention_config.num_attention:]:
        base_prompt = attention_config.prompt_fn(
            add_ds_entry["base_1_digits"],
            add_ds_entry["base_2_digits"],
            add_ds_entry["base_1_num"],
            add_ds_entry["base_2_num"],
        )
        truncated_base_prompt = _prompt.divide_prompt(attention_config.divide_num, base_prompt)[0]
        attention_prompts.append(modifier_fn(truncated_base_prompt, add_ds_entry))

    attention_tokens = tokenizer(
        attention_prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)
    return get_attention_freeze_hooks(model, attention_tokens)
