import sys
import torch
import datasets
import importlib
import transformers
from typing import Optional, Tuple, Union
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

    # patch flex attention
    patch_flex_wrapper()

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


def patch_flex_wrapper():
    # TODO remove this patch when transformers#37285 is merged and in a release
    is_torch_2_6 = torch.__version__.startswith("2.6")
    is_transformers_below_4_51 = transformers.__version__ < "4.51.0"

    if not (is_torch_2_6 and is_transformers_below_4_51):
        return

    from torch.nn.attention.flex_attention import flex_attention

    class WrappedFlexAttention:
        """
        We are doing a singleton class so that flex attention is compiled once when it's first called.
        """

        _instance = None
        _is_flex_compiled = False
        _compiled_flex_attention = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                # Create a new instance if one doesn't already exist
                cls._instance = super().__new__(cls)
            return cls._instance

        @torch.compiler.disable(recursive=False)
        def __init__(self):
            """
            Initialize or update the singleton instance.
            """
            if not self._is_flex_compiled:
                self._compiled_flex_attention = torch.compile(
                    flex_attention,
                    dynamic=False,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=True,
                )
                self._is_flex_compiled = True

        def __call__(self):
            return self._compiled_flex_attention

    transformers.integrations.flex_attention.WrappedFlexAttention = WrappedFlexAttention


def patch_flex_make_mask():
    is_torch_2_6 = torch.__version__.startswith("2.6")
    is_transformers_eq_4_51 = transformers.__version__ == "4.51.0"

    if not (is_torch_2_6 and is_transformers_eq_4_51):
        return

    from torch.nn.attention.flex_attention import (
        BlockMask,
    )
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_block_causal_mask_flex,
    )

    def patched_make_flex_block_causal_mask(
        attention_mask_2d: torch.Tensor,
        attention_chunk_size: Optional[int] = None,
        query_length=None,
        key_length=None,
        offsets: Optional[Tuple[Union[torch.Tensor, int], Union[torch.Tensor, int]]] = None,
    ) -> "BlockMask":
        """
        Create a block causal document mask for a batch of sequences, both packed and unpacked.
        Create Block causal logic and passing it into :func:`torch.nn.attention.flex_attention.create_block_mask`.
        The resultant BlockMask is a compressed representation of the full block causal
        mask. BlockMask is essential for performant computation of flex attention.
        See: https://pytorch.org/blog/flexattention/

        Args:
            attention_mask_2d (torch.Tensor): Attention mask for packed and padded sequences
            of shape (batch_size, total_seq_len). e.g.

            For unpacked sequence:
            [[1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0]]

            For packed sequence:
            [[1, 1, 1, 2, 2, 2, 0],
             [1, 1, 2, 2, 2, 3, 3]]

        Returns:
            BlockMask
        """

        batch_size, total_seq_len = attention_mask_2d.shape
        if not key_length:
            key_length = total_seq_len
        if not query_length:
            query_length = total_seq_len
        attention_mask_2d = torch.nn.functional.pad(
            attention_mask_2d, value=0, pad=(0, key_length)
        )
        device = attention_mask_2d.device
        document_ids = attention_mask_2d.clone()

        if attention_chunk_size is not None:
            # we create an arange, then we just // by chunk size to get [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            document_ids = (document_ids.fill_(1).cumsum(-1) - 1) // (
                attention_chunk_size
            )

        # Instead of passing a tensor mask, flex attention requires a mask_mod function
        # that determines which elements of QK^T should be included in the attention
        # computation prior to the softmax. For sample packing, we need both the
        # logic for both causal mask and document mask. See PyTorch's official
        # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
        def causal_mask_mod(
            batch_idx, head_idx, q_idx, kv_idx
        ):  # pylint: disable=unused-argument
            """
            Defines the logic of a block causal mask by combining both a standard causal mask
            and a block diagonal document mask.

            See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
            for an illustration.
            """
            causal_mask = q_idx >= kv_idx  # not valid when decoding
            document_mask = (
                document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
            )
            padding_mask = attention_mask_2d[batch_idx, q_idx] > 0
            final_mask = causal_mask & padding_mask & document_mask
            return final_mask

        if offsets is not None:
            q_offset = offsets[0]
            kv_offset = offsets[1]

            def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                offset_q = q_idx + q_offset
                offset_kv = kv_idx + kv_offset
                return causal_mask_mod(batch_idx, head_idx, offset_q, offset_kv)

        else:
            mask_mod = causal_mask_mod
        return create_block_causal_mask_flex(
            mask_mod=mask_mod,
            B=batch_size,
            H=None,  # attention head
            Q_LEN=query_length,
            KV_LEN=key_length,
            device=device,
            _compile=True,
        )

    for n in tuple(sys.modules):
        if ".modeling_" in n and "llama4" not in n:
            if hasattr(sys.modules[n], "make_flex_block_causal_mask"):
                print(n)
                sys.modules[n].make_flex_block_causal_mask = (
                    patched_make_flex_block_causal_mask
                )

    transformers.integrations.flex_attention.make_flex_block_causal_mask = (
        patched_make_flex_block_causal_mask
    )