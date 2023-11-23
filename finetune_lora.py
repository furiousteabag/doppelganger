import json
from typing import Any, Dict, List, Union

import fire
import torch
import wandb
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    MistralForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {round(100 * trainable_params / all_param, 2)}"
    )


def prepare_dataset(path: str) -> List[str]:
    with open(path, "r") as f:
        sessions = json.load(f)

    final_sessions = []
    for session in sessions:
        session_str = "\n".join([f"<|im_start|>{msg['author']}\n{msg['text']}<|im_end|>" for msg in session])
        final_sessions.append(session_str)

    return final_sessions


class DataCollatorForLanguageModelingChatML(DataCollatorForLanguageModeling):
    """
    Data collator for [ChatML](https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/chatml.md) format, like:
    ```
    <|im_start|>John Smith
    >>> damn, can't get around the 135 time limit
    >>> trying to do everything super optimally, but no luck<|im_end|>
    <|im_start|>Alexander Smirnov
    >>> yeah same
    >>> you still going with the same idea?<|im_end|>
    ```
    It will label any rows which are not starting with '>>>' to ignore.

    Reference data collator implementation: [DataCollatorForCompletionOnlyLM](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L56)

    Args:
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.start_token = self.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        self.end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        self.new_line_token = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.bos_token = self.tokenizer.bos_token_id

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            if_start = False
            for j in range(len(batch["labels"][i])):
                token = batch["labels"][i][j].item()

                if token == self.start_token:
                    if_start = True

                if if_start or token == self.bos_token:
                    batch["labels"][i][j] = self.ignore_index

                if token == self.new_line_token:
                    if_start = False

        return batch


def train(
    model_name_or_path: str = "ehartford/dolphin-2.2.1-mistral-7b",
    data_path: str = "./data/messages.json",
    output_dir: str = "./weights/LoRA/",
    batch_size: int = 16,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    lr_scheduler_type: str = "cosine",
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.001,
    max_seq_length: int = 1024,
    wandb_project: str = "doppelganger",
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    data_collator = DataCollatorForLanguageModelingChatML(tokenizer=tokenizer)

    dataset = Dataset.from_dict({"session": prepare_dataset(data_path)})

    # print(dataset[2500]["session"])
    # collator_res = data_collator([tokenizer(dataset["session"][2500], return_tensors="pt")["input_ids"][0]])
    # print(tokenizer.decode(collator_res["labels"][0][collator_res["labels"][0] != -100]))

    model: MistralForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        ),
        device_map={"": 0},
    )
    model.config
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )

    model = prepare_model_for_kbit_training(model)
    logger.info(model)

    model = get_peft_model(model, peft_config)
    logger.info(model)
    print_trainable_parameters(model)

    wandb.init(project=wandb_project, job_type="train", anonymous="allow")

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        save_steps=500,
        logging_steps=10,
        logging_first_step=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to=["wandb"],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        dataset_text_field="session",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(train)
