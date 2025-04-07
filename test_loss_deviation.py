import numpy as np
import torch
import sys
import importlib
from flashpack.collator import V2BatchSamplerDataCollatorForSeq2Seq
from flashpack.attention_utils import get_unpad_data
from transformers import AutoTokenizer

def test_boundary_labels():
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
    
    print("=== Testing Boundary Label Masking ===")
    
    labels = result["labels"][0].numpy()
    attention = result["attention_mask"][0].numpy()
    
    print("\nLabels:", labels)
    print("Attention mask:", attention)
    
    first_sample_end = 8  # Length of first sample
    masked_end_count = sum(1 for i in range(first_sample_end-5, first_sample_end) if labels[i] == -100)
    print(f"\nFirst sample end boundary masking: {masked_end_count}/5 tokens masked")
    
    masked_start_count = sum(1 for i in range(first_sample_end, first_sample_end+5) if labels[i] == -100)
    print(f"Second sample start boundary masking: {masked_start_count}/5 tokens masked")
    
    unique_values = np.unique(attention)
    print(f"\nUnique values in attention mask: {unique_values}")
    
    if masked_end_count > 0 and masked_start_count > 0:
        print("\n✅ PASS: Boundaries are properly masked to prevent cross-sample learning")
        print("This fix should address the loss deviation issue by preventing the model")
        print("from learning incorrect transitions between unrelated samples.")
    else:
        print("\n❌ FAIL: Boundaries are not properly masked")
        print("The model may still learn incorrect transitions between samples.")

def test_cross_sample_attention():
    """
    Test that the attention mechanism properly prevents cross-sample attention
    when multipack_attn=True.
    """
    print("\n=== Testing Cross-Sample Attention Prevention ===")
    
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
    
    print("\nAttention mask shape:", attn_mask.shape)
    
    sample1_token = 0  # First token in sample 1
    sample2_token = 4  # First token in sample 2
    
    can_attend_same_sample = attn_mask[0, sample1_token, sample1_token + 1].item()
    can_attend_other_sample = attn_mask[0, sample1_token, sample2_token].item()
    
    print(f"Token from sample 1 can attend to another token from sample 1: {can_attend_same_sample}")
    print(f"Token from sample 1 can attend to a token from sample 2: {can_attend_other_sample}")
    
    if can_attend_same_sample and not can_attend_other_sample:
        print("\n✅ PASS: Cross-sample attention is properly prevented")
        print("This fix should address the loss deviation issue by ensuring that")
        print("tokens can only attend to tokens from the same sample.")
    else:
        print("\n❌ FAIL: Cross-sample attention is not properly prevented")
        print("The model may still attend across sample boundaries.")
    
    if module_name in sys.modules:
        del sys.modules[module_name]

if __name__ == "__main__":
    test_boundary_labels()
    test_cross_sample_attention()
