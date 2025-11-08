import tqdm
from mlx_lm import generate

def evaluate(model, tokenizer, num_test, test_set_eval):
    num_correct = 0
    for prompt, completion, answer in tqdm.tqdm(test_set_eval[:num_test]):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        # Use greedy decoding for evaluation
        response = generate(model, tokenizer, prompt=messages, max_tokens=4, temp=0.0)
        num_correct += (answer in response)
    return num_correct / num_test