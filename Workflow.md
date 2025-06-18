# Workflow Overview

## 1. Why Ray? High-Level Architecture

Ray is used so that the training process (actor training, rollout generation, critic training, etc.) can be distributed or parallelized across GPUs (or even multiple nodes). Even if you only have a single GPU or single node, veRL (and the code in `verl/`) uses Ray’s APIs as a clean way to:

- **Launch worker processes**: Each worker can hold a copy of the model in data-parallel or other parallel forms.
- **Orchestrate remote calls**: For example, calling `.generate_sequences()` on the rollout worker, `.update_actor()` on the actor worker, `.update_critic()` on the critic worker.
- **Gather results**: Results are collected back in the driver process (the single “controller” that runs the PPO loop).

### Example Code
```python
import ray
from verl.worker_group import RayWorkerGroup

ray.init()
actors_wg = RayWorkerGroup(...)
critics_wg = RayWorkerGroup(...)

# Main PPO loop calls remote functions
rollout_data = actors_wg.generate_sequences(...)
rollout_data = critics_wg.compute_values(...)
# Advantage computation done locally
actors_wg.update_actor(...)
critics_wg.update_critic(...)
```

## 2. PPO Pipeline Overview
The typical PPO loop in verl (and in the logs you see) goes like this:

### Sample Prompts
The trainer reads a batch of prompts (from your dataset parquet). For example, “Given these numbers, create an equation that equals X…”.

### Rollout Generation
The “rollout” worker (which can be vLLM or HuggingFace or some other inference engine) generates responses to those prompts using the current Actor model weights.
So you call actor_rollout_wg.generate_sequences(...).

### Compute Rewards
If you have a rule-based reward, the PPO trainer detokenizes the LLM’s output, calls the specialized rule-based “compute_score” function (for example “countdown” or “gsm8k” or “math”), and returns a numeric reward for each generated response.
Alternatively, if you had a model-based reward (an RM), the trainer would call the reward model worker to get the reward.

### Compute Values (Critic)
The “critic” worker (a learned value function) runs a forward pass to estimate the value of each generated response.

### Compute Advantages
The driver process merges everything: the log-probs, rewards, values, etc.
Then it computes advantages (e.g., GAE, or any advantage formula).

### Update Critic
The code calls critic_wg.update_critic(...) so that the Critic’s parameters are updated via gradient descent on an MSE or similar value loss.

### Update Actor
Finally, we call actor_rollout_wg.update_actor(...), applying the PPO objective (policy gradient with clipping, or KL penalties, etc.).
The Actor is now updated, which means next time we do rollouts, the model is slightly improved.

### Repeat
Go to the next batch of prompts, do it again for the specified number of steps/epochs.
In verl, the above steps are orchestrated by the RayPPOTrainer in verl/trainer/ppo/ray_trainer.py. The logs you see (like actor/entropy_loss, critic/vf_loss, actor/grad_norm, etc.) are metrics from that PPO loop.

## 3. The Countdown “Rule-Based” Reward
When you have a “rule-based” reward, the pipeline does not load a “reward model.” Instead, you rely on a function that knows how to parse the LLM’s final answer from the text and compare it to the ground truth.

### For the countdown task:
#### Data Preprocessing (examples/data_preprocess/countdown.py)
Each sample has a target integer, a list of “numbers,” and we ask the model to “create an equation that equals [target].”
That code also writes into the parquet something like
```json
{
  "data_source": "countdown",
  "prompt": [{"role": "user", "content": "..."}],
  "reward_model": {
    "style": "rule",
    "ground_truth": { "target": 37, "numbers": [3,5,10,...] }
  }
}
```
Notice the style: "rule" in "reward_model". That is how veRL’s trainer knows to call the rule-based reward function for “countdown.”

### RewardManager checks data_source or style
In the PPO code, there is a RewardManager that sees whether the dataset row has "style": "rule", or “model,” or something else. If "style": "rule", it calls the corresponding function, e.g. countdown.compute_score(...).

### countdown.compute_score (Typically in verl/utils/reward_score/countdown.py)
The function might parse the LLM output (like the <answer> (2+5)*7 </answer>), and then evaluate if that expression indeed equals the target. If it does, the reward is 1.0; if not, maybe 0, etc.
The exact logic is up to you. For example, you might do a regex to find the <answer> ... </answer> text, parse it, evaluate it numerically, check if it matches the target.
That final float is the “reward.”
Thus, during the PPO loop, the rollout model’s text is turned into a numeric reward by that countdown function. If your dataset or code can’t find the final answer (or can’t parse it), it might give reward=0 or throw an error.

### What if your dataset does not define “extract_solution”?
Then, if “style=rule,” the code that tries to parse the final answer will fail or produce reward=0 for everything. In practice, that means the model is not receiving any meaningful learning signal (and is likely to remain random or degrade).
Another possibility is that you do not set “style=rule” at all, but then the trainer tries to use something else (like a “model-based” reward). If you never provided that reward model, you get an error or zero reward.