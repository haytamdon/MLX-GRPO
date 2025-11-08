import mlx.core as mx

# Pad sequences to the same length
def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return mx.array([])

    # Find hte maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded_seq = mx.concatenate([seq, padding])

        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return mx.stack(padded_sequences)

__all__ = ["pad_sequences"]