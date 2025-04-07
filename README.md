# FlashSamplePack

FlashSamplePack is a separation of the sample packing logic from [axolotl](https://github.com/axolotl-ai-cloud/axolotl). The idea is to give you full control over your dataflow while providing you the same speedup (up to 20x) with minimal dependencies.

## Install

```
pip install -e '.[train,flash]'
```

## Example

You can find an example that uses Huggingface Transformers + TRL to run supervised finetuning in examples/train.py.
