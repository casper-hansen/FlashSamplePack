# FlashSamplePack

FlashSamplePack is a separation of the sample packing logic from [axolotl](https://github.com/axolotl-ai-cloud/axolotl). The idea is to give you full control over your dataflow while providing you the same speedup (up to 20x) with minimal dependencies.
The main reason for packing multiple samples into a single sequence is to make more efficient use of memory, especially when you're limited to a batch size of 1 per GPU. This is particularly useful with long sequence lengths like 32k or 64k, where sample lengths can vary significantly. By packing samples together, you can greatly reduce the amount of wasted space from padding, which becomes important when increasing the batch size isn’t an option due to memory constraints.

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
pip install -e '.[train]'
```

## Example

You can find an example that uses Huggingface Transformers + TRL to run supervised finetuning in examples/train.py. The aim is to provide high-performing configurations out of the box that you can copy and adapt to your dataset and needs.

### Speed

#### Qwen 2.5 7B in BF16

In this case, we only achieve a very slight speedup because we are able to use batch size 2 for the no packing example.

DeepSpeed ZeRO-3 launch (LEN=65536, 8x H100, tokens/second/device)
- Sample pack, flash attention (bs=1): 250 steps / 0.698 samples/s = 358.17 seconds
- NO sample pack, flash attention (bs=2): 750 steps / 1.947 samples/s = 385.12 seconds

#### Mistral Nemo 12B in BF16

In this case we achieve a 3x speedup.

DeepSpeed ZeRO-3 launch (LEN=65536, 8x H100, tokens/second/device)
- Sample pack, flash attention (bs=1): 251 steps / 0.452 samples/s = 555.31 seconds
- NO sample pack, flash attention (bs=2): 750 steps / 0.979 samples/s = 766.09 seconds

### Running the example

#### Python

Running with Python seems to be most memory-optimal rather than with accelerate.

NOTE: Make sure to remove the deepspeed config and adjust batch size appropriately.

```
python examples/train.py
```

#### DeepSpeed

Running with DeepSpeed enables AutoTP which is claimed to speedup training 4x.

```
deepspeed --num_gpus=8 examples/train.py
```