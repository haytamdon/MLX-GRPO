from mlx import nn
import mlx.core as mx

def calculate_log_probs(model, sequences, a_toks):
    """Calculates the log probabilities of the generated answer tokens."""
    # Pass the full sequence (prompt + answer) to the model
    logits = model(sequences)

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)

    ## Find the actual positions where answer tokens should be extracted
    # This assumes a_toks contains the actual token IDs that were generated
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape

    # Calculate the starting position for answer tokens (assuming they're at the end)
    start_pos = seq_len - ans_len

    # Extract log probabilities for the answer portion of the sequence
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]

    # Create indices for gathering - ensure proper shape alignment
    indices = a_toks[:, :, None]

    # Extract log probabilities for the actual answer tokens
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)

    # Sum log probabilities across the answer sequence
    return mx.sum(selected_log_probs, axis=-1)

def grpo_loss_fn(model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon):
    """The GRPO loss function."""
    # Get log probs from the trainable model (π_θ)
    log_probs = calculate_log_probs(model, sequences, a_toks)

    # Get log probs from the reference model (π_ref) for KL penalty
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)

    # PPO-clip objective
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

    # KL penalty
    # Step 1: Calculate log(r) where r = π_ref / π_θ
    # log(r) = log(π_ref) - log(π_θ)
    log_ratio_for_kl = log_probs_ref - log_probs

    # Step 2: Calculate r itself by exponentiating log(r)
    # r = exp(log(r))
    ratio_for_kl = mx.exp(log_ratio_for_kl)

    # Step 3: Apply the paper's full formula: r - log(r) - 1
    kl_div = ratio_for_kl - log_ratio_for_kl - 1

    # The objective is to maximize this, so we return the negative for minimization
    loss = -mx.mean(policy_reward - beta * kl_div)
    return loss, mx.mean(policy_reward), mx.mean(kl_div)