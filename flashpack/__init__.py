from flashpack.patch import patch_for_multipack
from flashpack.attention_utils import get_unpad_data
from flashpack.chat_templates import *
from flashpack.collator import V2BatchSamplerDataCollatorForSeq2Seq
from flashpack.sampler import MultipackBatchSampler
from flashpack.dataset import (
    prepare_dataset,
    get_dataset_lengths,
    cache_dataset,
)
from flashpack.distributed import zero_first, is_local_main_process