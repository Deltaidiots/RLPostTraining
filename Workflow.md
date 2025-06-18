# Workflow Overview

This document provides a guide to the architecture and workflow of the `veRL` repository, designed to help new developers understand how to configure and run reinforcement learning experiments for large language models.

## 1. Core Concepts & Architecture

### 1.1. Distributed Training with Ray

`veRL` uses [Ray](https://www.ray.io/) to distribute the training process (actor training, rollout generation, critic training, etc.) across multiple GPUs or even multiple nodes. Ray provides a clean API to:

-   **Launch Worker Processes**: Each worker can hold a copy of a model and execute tasks in parallel.
-   **Orchestrate Remote Calls**: The main training loop can call functions on remote workers, like `.generate_sequences()` or `.update_actor()`.
-   **Gather Results**: Data is collected back in the main driver process, which runs the PPO loop.

This allows the system to scale from a single machine to a large cluster without changing the core logic.

### 1.2. Key Model Components in PPO

Our PPO implementation uses three key models:

-   **Actor**: The policy model that is actively being trained. It generates responses to prompts.
-   **Critic**: The value model that learns to estimate the expected return (value) from a given state. This is used to calculate advantages.
-   **Reference Model**: A frozen, initial copy of the actor model. It is used to calculate a KL-divergence penalty, which prevents the actor from deviating too far from the original policy during training, ensuring stability.

### 1.3. Worker Architecture & Parallelism

`veRL` supports different backends for model parallelism and efficient inference:

-   **Model Parallelism**: For training large models that don't fit on a single GPU, `veRL` supports backends like **FSDP** (Fully Sharded Data Parallel) and **Megatron-LM**.
-   **Inference/Rollout Engine**: For fast generation of sequences during the rollout phase, `veRL` can use highly optimized engines like **vLLM**.

These components are configured as "workers" that Ray manages.

### 1.4. The `DataProto` Object

Communication between workers is standardized using the `DataProto` object. It acts as a container for passing tensors (like token IDs, log-probs, and values) and other metadata between the different stages of the PPO loop.

## 2. Configuration with Hydra

Experiments are configured using [Hydra](https://hydra.cc/). This allows you to easily override parameters from the command line.

For example, in a training script, you will see arguments like:
```bash
python3 -m verl.trainer.main_ppo \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.project_name=MyExperiment
```
This system allows for modular and reproducible experiment configurations without hardcoding values.

## 3. The PPO Training Loop in Detail

The typical PPO loop in `veRL` is orchestrated by `RayPPOTrainer` and proceeds as follows:

1.  **Sample Prompts**: The trainer reads a batch of prompts from a dataset (e.g., a `.parquet` file).
2.  **Rollout Generation**: The `Actor` worker generates responses to the prompts. The log-probabilities of the generated tokens are also computed.
3.  **Compute Rewards**:
    -   For **rule-based rewards** (like the Countdown task), a function parses the generated text, compares it to a ground-truth answer, and returns a numeric score.
    -   For **model-based rewards**, a separate Reward Model (RM) is called to score the response.
4.  **Compute Values & KL-Divergence**:
    -   The `Critic` worker runs a forward pass to estimate the value of each generated sequence.
    -   The `Reference Model` is used to get the log-probabilities of the generated tokens, which are then used to compute the KL-divergence from the `Actor`'s policy.
5.  **Compute Advantages**: The main driver process gathers the rewards and values to compute advantages (e.g., using Generalized Advantage Estimation - GAE).
6.  **Update Critic**: The `Critic` worker's parameters are updated by minimizing the difference between its value predictions and the actual returns (value loss).
7.  **Update Actor**: The `Actor` worker's parameters are updated using the PPO objective, which incorporates the computed advantages and the KL-divergence penalty.
8.  **Repeat**: The process repeats for the next batch of prompts until the specified number of epochs is complete.

## 4. Experiment Tracking with WandB

`veRL` is integrated with tools like Weights & Biases (`wandb`) for experiment tracking. You can enable it via configuration:
```bash
trainer.logger=['wandb']
trainer.project_name=MyProject
```
This will log all important metrics (losses, rewards, gradients, etc.) to your `wandb` dashboard, allowing you to monitor training in real-time and compare results across different runs.

## 5. Example: The Countdown "Rule-Based" Reward

When you have a "rule-based" reward, the pipeline does not load a separate "reward model." Instead, it relies on a function that knows how to parse the LLM’s final answer and compare it to the ground truth.

### Data Preprocessing (`examples/data_preprocess/countdown.py`)
For the countdown task, each sample has a target integer and a list of numbers. The preprocessor formats the data and includes metadata to signal how the reward should be calculated:
```json
{
  "data_source": "countdown",
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {
    "style": "rule",
    "ground_truth": { "target": 98, "numbers": [44, 19, 35] }
  }
}
```
The `style: "rule"` flag tells the `RewardManager` in `veRL` to use the specific `compute_score` function associated with the `countdown` task.

### Reward Computation (`verl/utils/reward_score/countdown.py`)
This script contains the logic to:
1.  Parse the LLM's generated text (e.g., extracting the equation from `<answer>` tags).
2.  Safely evaluate the mathematical expression.
3.  Check if the result matches the `target` from the ground truth.
4.  Return a float reward (e.g., `1.0` for a correct answer, `0.0` otherwise).

This reward is then fed back into the PPO loop to update the actor model. If the parsing fails or the answer is incorrect, the model receives a low or zero reward, guiding it to produce better answers in the future.