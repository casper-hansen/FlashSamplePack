import numpy as np
import torch
from flashpack.collator import V2BatchSamplerDataCollatorForSeq2Seq
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

if __name__ == "__main__":
    test_boundary_labels()
