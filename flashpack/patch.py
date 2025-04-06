import torch
import datasets
import importlib
import transformers
from typing import Optional, Union
from torch.utils.data import DataLoader
from accelerate import init_empty_weights
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data._utils.worker import _worker_loop
from transformers import AutoConfig, AutoModelForCausalLM, Trainer, trainer_utils
from flashpack.attention_utils import get_unpad_data


def patch_for_multipack(
    sampler, eval_sampler=None, model_name=None, has_remote_code=False
):
    # patch dataloading
    torch.utils.data._utils.worker._worker_loop = patched_worker_loop
    patch_fetchers()

    # patch model
    if has_remote_code:
        patch_remote(model_name)
    elif hasattr(transformers, "modeling_flash_attention_utils"):
        transformers.modeling_flash_attention_utils._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )

    # patch trainer
    class FlashTrainer(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if isinstance(train_dataset, datasets.Dataset):
                train_dataset = self._remove_unused_columns(
                    train_dataset, description="training"
                )
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description="training"
                )

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                # PATCH START
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
                # PATCH END
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = trainer_utils.seed_worker
                dataloader_params["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )

            # PATCH START
            self.accelerator.even_batches = False
            # PATCH END

            return self.accelerator.prepare(
                DataLoader(train_dataset, **dataloader_params)
            )

        def get_eval_dataloader(
            self, eval_dataset: Optional[Union[str, datasets.Dataset]] = None
        ) -> DataLoader:
            """
            Returns the evaluation [`~torch.utils.data.DataLoader`].

            Subclass and override this method if you want to inject some custom behavior.

            Args:
                eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                    If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
            """
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            # If we have persistent workers, don't do a fork bomb especially as eval datasets
            # don't change during training
            dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
            if (
                hasattr(self, "_eval_dataloaders")
                and dataloader_key in self._eval_dataloaders
                and self.args.dataloader_persistent_workers
            ):
                return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

            eval_dataset = (
                self.eval_dataset[eval_dataset]
                if isinstance(eval_dataset, str)
                else eval_dataset if eval_dataset is not None else self.eval_dataset
            )
            data_collator = self.data_collator

            if isinstance(eval_dataset, datasets.Dataset):
                eval_dataset = self._remove_unused_columns(
                    eval_dataset, description="evaluation"
                )
            else:
                data_collator = self._get_collator_with_removed_columns(
                    data_collator, description="evaluation"
                )

            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
            }

            if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
                # PATCH START
                if eval_sampler:
                    dataloader_params["batch_sampler"] = eval_sampler
                    del dataloader_params["batch_size"]
                else:
                    dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
                # PATCH END

                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )

            # accelerator.free_memory() will destroy the references, so
            # we need to store the non-prepared version
            eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
            if self.args.dataloader_persistent_workers:
                if hasattr(self, "_eval_dataloaders"):
                    self._eval_dataloaders[dataloader_key] = eval_dataloader
                else:
                    self._eval_dataloaders = {dataloader_key: eval_dataloader}

            # PATCH START
            self.accelerator.even_batches = False
            # PATCH END

            return self.accelerator.prepare(eval_dataloader)

    transformers.trainer.Trainer.get_train_dataloader = FlashTrainer.get_train_dataloader
    transformers.trainer.Trainer.get_eval_dataloader = FlashTrainer.get_eval_dataloader


def patch_remote(model_name):
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_* to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    parts = model_config.__class__.__module__.split(".")
    parts[-1] = parts[-1].replace("configuration_", "modeling_", 1)
    module_name = ".".join(parts)
    modeling_arch = importlib.import_module(module_name)
    if hasattr(modeling_arch, "_get_unpad_data"):
        modeling_arch._get_unpad_data = (  # pylint: disable=protected-access
            get_unpad_data
        )

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if isinstance(possibly_batched_index[0], list):
            data = [None for i in possibly_batched_index]
            for i, possibly_batched_index_ in enumerate(possibly_batched_index):
                if self.auto_collation:
                    if (
                        hasattr(self.dataset, "__getitems__")
                        and self.dataset.__getitems__
                    ):
                        data[i] = self.dataset.__getitems__(possibly_batched_index_)
                    else:
                        data[i] = [self.dataset[idx] for idx in possibly_batched_index_]
                else:
                    data[i] = self.dataset[possibly_batched_index_]
        else:
            if self.auto_collation:
                if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                    data = self.dataset.__getitems__(possibly_batched_index)
                else:
                    data = [self.dataset[idx] for idx in possibly_batched_index]
            else:
                data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


def patch_fetchers():
    torch.utils.data._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher
    torch.utils.data.dataloader._utils.fetch._MapDatasetFetcher = _MapDatasetFetcher


def patched_worker_loop(*args, **kwargs):
    patch_fetchers()
    return _worker_loop(*args, **kwargs)
