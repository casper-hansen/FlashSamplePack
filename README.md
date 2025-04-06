# FlashSamplePack

FlashSamplePack is a separation of the sample packing logic from [axolotl](https://github.com/axolotl-ai-cloud/axolotl). The idea is to give you full control over your dataflow while providing you the same speedup (up to 20x) with minimal dependencies.

## Example

You can find an example that uses Huggingface Transformers + TRL to run supervised finetuning in examples/train.py.

- ON: {'train_runtime': 38.3893, 'train_samples_per_second': 58.871, 'train_steps_per_second': 7.372, 'train_tokens_per_second': 15097.544, 'train_loss': 0.7590425399925178, 'epoch': 1.0} {'loss': 0.7739, 'grad_norm': 6.53125, 'learning_rate': 0.0, 'num_tokens': 2022508.0, 'mean_token_accuracy': 0.766749382019043, 'epoch': 1.0}
- OFF: {'train_runtime': 38.0342, 'train_samples_per_second': 59.42, 'train_steps_per_second': 7.441, 'train_tokens_per_second': 14825.809, 'train_loss': 0.7579695512886182, 'epoch': 1.0} {'loss': 0.56, 'grad_norm': 5.46875, 'learning_rate': 0.0, 'num_tokens': 451468.0, 'mean_token_accuracy': 0.8321759104728699, 'epoch': 1.0}