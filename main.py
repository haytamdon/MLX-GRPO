import os
from dotenv import load_dotenv
from utils.download_data import load_json
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers
from trainers.mlx_trainer import grpo_train_loop
import os
from pathlib import Path
import time
import json
import yaml
import mlx.optimizers as optim
load_dotenv()

save_dir = os.environ["DATA_SAVE_DIR"]
np.random.seed(os.environ["RANDOM_SEED"])
adapter_path = Path(os.environ["ADAPTERS_DIR"])
config_path = Path(os.environ["CONFIGS_DIR"])
model_path = os.environ["MODEL_PATH"]
lora_config_file_name = os.environ["LORA_CONFIG_FILE_NAME"]
grpo_config_file_name = os.environ["GRPO_CONFIG_FILE_NAME"]

def get_data_subsets(train_set):
    perm = np.random.permutation(len(train_set))
    valid_size = int(0.1 * len(train_set))
    valid_set = [train_set[i] for i in perm[:valid_size]]
    train_set = [train_set[i] for i in perm[valid_size:]]
    return valid_set, train_set

if __name__ == "__main__":

    train_set, test_set = load_json("train"), load_json("test")
    valid_set, train_set = get_data_subsets(train_set)

    # Make a directory to save the adapter config and weights
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Load the LoRA config from the adapter path
    with open(adapter_path / lora_config_file_name, "r") as file:
        lora_config = json.load(file)
    
    # Load grpo hyperparameters from the configs path
    with open(config_path / grpo_config_file_name, "r") as file:
        grpo_config = yaml.safe_load(file)

    # Load the main model and tokenizer
    model, tokenizer = load(model_path)

    # Load the reference model
    model_ref, _ = load(model_path)
    model_ref.freeze()

    # Freeze the base model
    model.freeze()

    # Convert linear layers to lora layers
    linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])

    # Create the old model for rollouts
    model_old, _ = load(model_path)
    linear_to_lora_layers(model_old, lora_config["num_layers"], lora_config["lora_parameters"])
    model_old.update(model.parameters()) # Sync weights
    model_old.freeze()

    num_train_params = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    )

    # Put the model in training mode:
    model.train()

    # Make the optimizer:
    opt = optim.Adam(learning_rate=grpo_config["learning_rate"])

    print("Starting GRPO training...")
    start_time = time.time()

    # Run the custom GRPO training loop
    losses, rewards = grpo_train_loop(
        model = model,
        model_old = model_old,
        model_ref = model_ref,
        tokenizer = tokenizer,
        optimizer = opt,
        train_set = train_set,
        iters = grpo_config["iters"],
        group_size = grpo_config["group_size"],
        batch_size = grpo_config["batch_size"],
        epsilon = grpo_config["epsilon"],
        beta = grpo_config["beta"],
        update_every = grpo_config["update_every"],
        max_ans_len = grpo_config["max_ans_len"]
    )

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f}s")