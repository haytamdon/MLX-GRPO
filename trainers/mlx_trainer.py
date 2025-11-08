import mlx.nn as nn
from metrics.metrics_calculations import grpo_loss_fn, calculate_log_probs
import tqdm
import numpy as np
from mlx_lm import generate
import mlx.core as mx
from utils.utils import pad_sequences

def grpo_train_loop(
    model, model_old, model_ref, tokenizer, optimizer, train_set, 
    adapter_path, iters=200, group_size=4, batch_size=2, epsilon=0.2, 
    beta=0.01, update_every=10, max_ans_len=4
):
    # Create a grad function for the trainable model
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    
    losses = []
    all_rewards = []
    
    # Start training
    pbar = tqdm.tqdm(range(iters))
    for it in pbar:
        batch_prompts = []
        batch_answers = []
        
        # 1. Sample a batch of prompts
        indices = np.random.randint(0, len(train_set), batch_size)
        for i in indices:
            # The last word of the output is the ground truth answer (e.g., "ending4")
            prompt_text, answer_text = train_set[i]["output"].rsplit(" ", maxsplit=1)
            full_prompt = [
                {"role": "user", "content": train_set[i]["instruction"]},
                {"role": "assistant", "content": prompt_text}
            ]
            batch_prompts.append(full_prompt)
            batch_answers.append(answer_text)
        
        # 2. Rollout: Generate G responses for each prompt using the old model
        rollout_sequences = []
        rollout_rewards = []
        rollout_log_probs = []
        rollout_a_toks = []

        for i in range(batch_size):
            prompt_tokens = tokenizer.apply_chat_template(batch_prompts[i], continue_final_message=True)
            group_rewards = []

            for _ in range(group_size):
                # Generate a response
                response = generate(model_old, tokenizer, prompt_tokens, max_tokens=max_ans_len)
                answer_tokens = tokenizer.encode(response, add_special_tokens=False)

                # 3. Get Reward
                reward = 1.0 if batch_answers[i] in response else 0.0
                group_rewards.append(reward)

                # Store data for the optimization step
                full_sequence = mx.array(prompt_tokens + answer_tokens)
                rollout_sequences.append(full_sequence)
                rollout_a_toks.append(mx.array(answer_tokens))

            all_rewards.extend(group_rewards)
            rollout_rewards.append(mx.array(group_rewards))
        
        # 4. Compute Advantages
        advantages = []
        for rewards in rollout_rewards:
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8 # Add epsilon for stability
            adv = (rewards - mean_reward) / std_reward
            advantages.append(adv)
        
        advantages = mx.concatenate(advantages)
        sequences = pad_sequences(rollout_sequences, tokenizer.pad_token_id)
        a_toks = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)

        # Calculate log_probs with the old model for the ratio calculation
        old_log_probs = calculate_log_probs(model_old, sequences, a_toks)

        # 5. Optimization Step
        (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
            model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon
        )
        
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())
        pbar.set_description(f"Loss: {np.mean(losses[-10:]):.3f}, Mean Reward: {np.mean(all_rewards[-20:]):.3f}")
        
        # Sync old model weights
        if (it + 1) % update_every == 0:
            model_old.update(model.parameters())
            print(f"\nIter {it+1}: Synced old model weights.")
            
    # Final save of adapter weights
    model.save_weights(str(adapter_path / "adapters.safetensors"))
    print("Saved final weights to adapters/adapters.safetensors.")
    return losses, all_rewards