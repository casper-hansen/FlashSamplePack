# FlashSamplePack

FlashSamplePack is a separation of the sample packing logic from [axolotl](https://github.com/axolotl-ai-cloud/axolotl). The idea is to give you full control over your dataflow while providing you the same speedup (up to 20x) with minimal dependencies.

Supported attention implementations:
- Flash Attention
- Flex Attention

Supported trainers:
- Transformers
- TRL

## Why FlashSamplePack?

The implementation in TRL for packing does not fix the attention mechanism to avoid multiple samples to attend to each other. This causes issues with the loss value and is not appropriate for training models.

Additionally, this package can be used independently of Axolotl, offering users greater flexibility in implementating their training strategies.

## Install

```
pip install -e '.[train,flash]'
```

## Example

You can find an example that uses Huggingface Transformers + TRL to run supervised finetuning in examples/train.py. The aim is to provide high-performing configurations out of the box that you can copy and adapt to your dataset and needs.

Speed in everyday conversations on Qwen 2.5 7B (1x H100, tokens/second):
- Sample pack, flash attention: 10415.406
- Sample pack, flex attention: 8829.433
- NO sample pack, flash attention: 6701.847
