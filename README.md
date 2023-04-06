# Dataset Preparation

TODO: add instruction and option to customize target_id

```bash
python prepare_dataset.py -i ./data/result.json -o ./data/messages.json
```

# Model Training

1. Get weights
2. Convert weights
```
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
python convert_llama_weights_to_hf.py --input_dir ./weights/LLaMA --model_size 7B --output_dir ./weights/LLaMA_converted/7B
rm convert_llama_weights_to_hf.py
```
