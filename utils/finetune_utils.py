import json
from typing import Any, Dict, List, Union

from loguru import logger
from transformers import DataCollatorForLanguageModeling


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
