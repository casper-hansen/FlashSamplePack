import torch
import torch.nn.functional as F


@torch.jit.script
def get_max_seqlen_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    max_num = int(torch.max(attention_mask).item())
    batch_size, _ = attention_mask.shape
    counts = torch.zeros((batch_size, max_num), dtype=torch.int32)
    for i in range(1, max_num + 1):
        mask = attention_mask == i
        counts[:, i - 1] = torch.sum(mask, dim=-1).to(dtype=torch.int32)
    result = counts.flatten()
    nonzero_indices = torch.nonzero(result).squeeze(-1)
    return result[nonzero_indices]


@torch.jit.script
def get_unpad_data(attention_mask: torch.Tensor):
    device = attention_mask.device
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = (
        F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        .to(device=device)
        .detach()
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_cu_seqlens_from_pos_ids(position_ids):
    """generate a cumulative sequence length mask for flash attention using pos ids"""
    if len(position_ids.shape) == 1:
        position_ids = position_ids.unsqueeze(0)

    device = position_ids.device
    results = []
    max_seq_lens = []

    for row in position_ids:
        # Count the number of consecutive zeros from the right side
        padding_length = (row == 0).int().flip(dims=[0]).cumprod(dim=0).sum().item()

        # Adjust the row to exclude padding
        adjusted_row = row[:-padding_length] if padding_length else row.clone()

        # Find where the position resets to 0 (indicating a new sequence)
        seq_starts = torch.cat(
            [
                torch.tensor([True], dtype=torch.bool, device=device),
                adjusted_row[1:] == 0,
            ]
        )
        # Get the indices where the sequence starts
        start_indices = torch.cat(
            [
                torch.nonzero(seq_starts).unbind(dim=1)[0],
                torch.tensor([len(adjusted_row)], dtype=torch.int32, device=device),
            ]
        )
        # Calculate the sequence lengths
        seq_lengths = start_indices[1:] - start_indices[:-1]
        # Calculate the cumulative sequence lengths
        cu_seqlens = torch.cat(
            [torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)]
        )
        # Append the padding length to the cumulative sequence lengths
        if padding_length:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([len(row)], dtype=torch.int32, device=device)]
            )
        max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        results.append(cu_seqlens)
        max_seq_lens.append(max_seq_len)

    # Find the maximum value across all tensors
    max_value = max(t.max() for t in results)

    # Find the length of the longest tensor
    max_length = max(t.size(0) for t in results)

    # Pad each tensor to the same length and collect them in a list
    padded_results = [
        F.pad(t, (0, max_length - t.size(0)), "constant", max_value) for t in results
    ]

    return torch.stack(padded_results).to(dtype=torch.int32), torch.stack(max_seq_lens)
