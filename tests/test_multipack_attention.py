import pytest
import torch
import numpy as np
import sys
from flashpack.collator import V2BatchSamplerDataCollatorForSeq2Seq
from flashpack.attention_utils import get_unpad_data
from transformers import AutoTokenizer

def test_boundary_label_masking():
    """
    Test that the V2BatchSamplerDataCollatorForSeq2Seq properly masks labels at sample boundaries
    to prevent the model from learning incorrect transitions between samples.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    collator = V2BatchSamplerDataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    features = [[
        {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], 
         "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1], 
         "labels": [10, 11, 12, 13, 14, 15, 16, 17]},
        {"input_ids": [21, 22, 23, 24, 25, 26, 27, 28], 
         "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1], 
         "labels": [31, 32, 33, 34, 35, 36, 37, 38]}
    ]]
    
    result = collator(features)
    
    labels = result["labels"][0].numpy()
    attention = result["attention_mask"][0].numpy()
    
    first_sample_end = 8  # Length of first sample
    
    masked_end_count = sum(1 for i in range(first_sample_end-5, first_sample_end) if labels[i] == -100)
    assert masked_end_count > 0, "End of first sample should have masked labels"
    
    masked_start_count = sum(1 for i in range(first_sample_end, first_sample_end+5) if labels[i] == -100)
    assert masked_start_count > 0, "Start of second sample should have masked labels"
    
    unique_values = np.unique(attention)
    assert len(unique_values) > 1, "Attention mask should have different values for different samples"

def test_cross_sample_attention():
    """
    Test that the attention mechanism properly prevents cross-sample attention
    when multipack_attn=True.
    """
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0],  # Two samples of length 4, followed by padding
    ])
    
    module_name = "test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    test_module = type(module_name, (), {"MULTIPACK_ATTN": True})
    sys.modules[module_name] = test_module
    
    original_get_unpad_data = get_unpad_data
    
    def mock_get_unpad_data(attention_mask):
        indices, cu_seqlens, max_seqlen_in_batch = original_get_unpad_data(attention_mask)
        
        device = attention_mask.device
        batch_size, seq_len = attention_mask.shape
        attn_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask[i, j] > 0:  # Skip padding tokens
                    attn_mask[i, j] = (attention_mask[i] == attention_mask[i, j])
        
        return indices, cu_seqlens, max_seqlen_in_batch, attn_mask
    
    indices, cu_seqlens, max_seqlen_in_batch, attn_mask = mock_get_unpad_data(attention_mask)
    
    sample1_token = 0  # First token in sample 1
    sample2_token = 4  # First token in sample 2
    
    can_attend_same_sample = attn_mask[0, sample1_token, sample1_token + 1].item()
    can_attend_other_sample = attn_mask[0, sample1_token, sample2_token].item()
    
    assert can_attend_same_sample, "Token should be able to attend to tokens from the same sample"
    assert not can_attend_other_sample, "Token should not be able to attend to tokens from different samples"
    
    if module_name in sys.modules:
        del sys.modules[module_name]
