# FlashSamplePack

FlashSamplePack is a separation of the sample packing logic from [axolotl](https://github.com/axolotl-ai-cloud/axolotl). The idea is to give you full control over your dataflow while providing you the same speedup (up to 20x) with minimal dependencies.

## Example

You can find an example that uses Huggingface Transformers + TRL to run supervised finetuning in examples/train.py.

- ON: {'train_runtime': 245.6366, 'train_samples_per_second': 4.071, 'train_steps_per_second': 4.071, 'train_tokens_per_second': 8337.519, 'train_loss': 0.843474586904049, 'epoch': 0.44}
- OFF: {'train_runtime': 214.1427, 'train_samples_per_second': 4.67, 'train_steps_per_second': 4.67, 'train_tokens_per_second': 803.203, 'train_loss': 0.8389686772823334, 'epoch': 0.44}
