import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(a, n=10):
    """Simple moving average for smoothing."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n

def plot_metrics(losses, rewards):
    """Plot GRPO training loss and reward curves."""
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("GRPO Loss", color=color)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Reward (Moving Avg)", color=color)
    ax2.plot(moving_average(rewards, n=50), color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("GRPO Training Loss and Reward")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    plt.savefig("plots/grpo_training_metrics.png")
    plt.close()
