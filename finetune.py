import copy
import json
import random
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, List, Sequence

import fire
import torch
import transformers
from loguru import logger
from torch.utils.data import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer

SEED = 322
IGNORE_INDEX = -100

random.seed(SEED)


def get_train_val_split(data_path: str, val_size: float) -> Dict:
    logger.warning("Loading data...")
    with open(data_path, "r") as f:
        list_data_dict = json.load(f)
    random.shuffle(list_data_dict)
    idx = int(len(list_data_dict) * (1 - val_size) + 1)
    train, val = list_data_dict[:idx], list_data_dict[idx:]
    return dict(train=train, val=val)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            # padding="longest",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, list_data_dict: List[Dict[str, str]], tokenizer: transformers.PreTrainedTokenizer
    ):
        super(SupervisedDataset, self).__init__()

        logger.warning("Formatting inputs...")
        sources = [example["context"] for example in list_data_dict]
        targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]

        logger.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# In[5]:


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train(
    model_name_or_path: str = "./weights/LLaMA_converted/7B",
    data_path: str = "./data/messages.json",
    val_size: float = 0.05,
    output_dir="./weights/LLaMA_trained/7B",
    batch_size: int = 32,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    warmup_ratio: float = 0.03,
    lr_scheduler_type: str = "cosine",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    wandb_project: str = "doppelganger",
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, device_map="auto"
    )
    logger.info(f"Loaded {model_name_or_path}")

    train_val = get_train_val_split(data_path=data_path, val_size=val_size)
    list_data_dict_train, list_data_dict_val = train_val["train"], train_val["val"]

    train_dataset = SupervisedDataset(list_data_dict=list_data_dict_train, tokenizer=tokenizer)
    val_dataset = SupervisedDataset(list_data_dict=list_data_dict_val, tokenizer=tokenizer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_size > 0 else False,
            report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
    )

    model = torch.compile(model)
    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
