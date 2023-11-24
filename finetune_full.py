import os
import torch
from typing import Dict, List

import fire
import wandb
from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer
from tokenizers import AddedToken

from utils.finetune_utils import DataCollatorForLanguageModelingChatML, prepare_dataset, print_trainable_parameters

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "<|im_end|>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_PAD_TOKEN = "</s>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, tokens_list: List = []
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) + tokenizer.add_tokens(tokens_list)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train(
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1",
    data_path: str = "./data/messages.json",
    output_dir: str = "./weights/full/",
    batch_size: int = 16,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_seq_length: int = 1024,
    fsdp: str = "full_shard auto_wrap",
    wandb_project: str = "doppelganger",
    logging_steps: int = 1,
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
        
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=max_seq_length, padding_side="right", use_fast=False
    )

    special_tokens_dict = dict()
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    tokens_list = [AddedToken("<|im_start|>", normalized=False)]

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokens_list=tokens_list,
        tokenizer=tokenizer,
        model=model,
    )

    data_collator = DataCollatorForLanguageModelingChatML(tokenizer=tokenizer)
    dataset = Dataset.from_dict({"session": prepare_dataset(data_path)})

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.init(project=wandb_project, job_type="train", anonymous="allow")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=500,
        logging_steps=logging_steps,
        logging_first_step=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        bf16=True,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        fsdp=fsdp,
        report_to=["wandb"] if int(os.environ.get("LOCAL_RANK", 0)) == 0 else [],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        dataset_text_field="session",
        max_seq_length=max_seq_length,
        packing=False,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
