# IT3105 Project 2 — MuZero-inspired Agent (JAX/Flax)

This repository contains an educational, MuZero-inspired implementation built for the course **AI Programming (IT3105)**. The goal of the project was to implement the core MuZero components—latent representation learning, model-based planning with MCTS, and training from self-play—using modern JAX tooling.

This is **not** reproduction of DeepMind’s MuZero. It is a working, end-to-end system that learns on discrete-action Gymnasium environments (notably CartPole), with several deliberate simplifications and known deviations documented below.

---

## What’s implemented

### Core MuZero-style components

-   **Representation network (h)**: maps preprocessed observation history → latent state
-   **Dynamics network (g)**: predicts next latent state + immediate reward from (latent state, action)
-   **Prediction network (f)**: predicts policy logits + value from latent state
-   **Planning (MCTS)**: PUCT-style search in latent space, producing an improved policy (visit counts) and value estimate
-   **Replay buffer**: episode-level storage with sampled unroll sequences for training

### Engineering / tooling

-   **JAX + Flax + Optax** for model definition and optimization
-   **`jax.jit`** for compiled forward passes and training steps
-   **`jax.vmap`** for batching sequence loss computation
-   **Checkpointing**: save/load params + optimizer state + training history + config snapshot
-   **Evaluation script**: run episodes from a specific checkpoint with rendering
-   **Training plots**: utilities to plot loss/reward histories

---

## Results

The implementation reaches strong performance on **CartPole-v1** using the provided configuration.

Recommended: add plots under `results/` and embed them here, for example:

-   `results/cartpole_training.png` (loss curves + avg reward)
-   `results/cartpole_eval.png` (optional)

---

## Repository structure (high-level)

-   `muzero.py` — main training loop / orchestration (self-play collection + training + checkpointing)
-   `mcts.py` — MuZero-style MCTS in latent space (PUCT-based)
-   `nn.py` — networks (h/g/f) + training step + checkpointing (JAX/Flax/Optax)
-   `state_manager.py` — adapter between environment and networks (h/g/f calls, legal actions)
-   `replay_buffer.py` — episode buffer + sequence sampling + n-step value targets
-   `utils.py` — config finalization + observation preprocessing + history stacking
-   `evaluate.py` — load checkpoint and run evaluation episodes with rendering
-   `game_simulator.py` - GymnasiumEnvManager is the environment adapter used by both training (muzero.py) and evaluation (evaluate.py).
-   `game_configs/` — YAML configuration files for environments
-   `plot_scripts/` — plotting utilities
-   `results/` - The created plots using output after training for 200 training steps on CartPole

---

## Quickstart

### 1) Install dependencies

```bash
pip install jax flax optax gymnasium ale-py numpy pyyaml matplotlib opencv-python
```

Notes:

-   Atari support typically requires additional dependencies and ROM setup (Gymnasium/ALE). This repo primarily targets discrete-action environments like CartPole.

### 2) Train

Run training using a YAML config. Ensure the command matches your entry point (adjust if needed):

```bash
python main.py --config game_configs/config_cartpole.yaml
```

Checkpoints and logs are written to the directory specified in:

```yaml
training:
    checkpoint_dir: "..."
```

### 3) Plot training history

If checkpoints generate a `muzero_history_<step>.json` file (or similar), use:

```bash
python ./plot_scripts/live_plot_history.py muzero_checkpoints/CartPole_v1_baseline --interval 100 --window 100
# or
python ./plot_scripts/static_plot_history.py ./muzero_checkpoints/CartPole_v1_baseline/muzero_history_200.json -o "Output file name" -w Windows_size
```

### 4) Evaluate a checkpoint (render)

```bash
python evaluate.py --config game_configs/config_cartpole.yaml --checkpoint_step 100 --num_episodes 1
```

---

## Configuration

Training and environment parameters are controlled via YAML configs in `game_configs/`.

Key sections:

-   `game_settings`: environment name, rendering, preprocessing, reward shaping
-   `global_network_vars`: input shape, latent dim, learning rate, seed
-   `neural_network`: h/g/f architecture, action encoding, unroll steps
-   `rlm`: training loop parameters (discount, max steps, episodes per train step, total steps)
-   `umcts`: MCTS parameters (simulations, exploration constant, rollout depth)
-   `episode_buffer`: replay buffer capacity
-   `training`: batch size, checkpointing interval, loss weights

---

## Discussion

### Differences from canonical MuZero

This implementation is “MuZero-inspired” and intentionally simplified. Notable differences include:

-   **MCTS includes a learned-model rollout**: after leaf expansion, the search may perform a rollout in the learned dynamics model before bootstrapping a final value. Canonical MuZero typically relies on value estimates and backed-up rewards without this rollout style.
-   **No root Dirichlet noise**: MuZero commonly injects Dirichlet noise at the root during self-play to encourage exploration. This is omitted here.
-   **No reanalysis**: MuZero-style pipelines often recompute value/policy targets with newer networks (“reanalysis”) for stability. This implementation trains on targets generated at data-collection time.
-   **Simplified target construction**: the replay buffer computes n-step value targets with bootstrapping using stored value estimates. (The TD horizon is effectively tied to `unroll_steps` here.)
-   **Discrete action spaces only**: the environment wrapper assumes `gym.spaces.Discrete` for legal actions and action sizing.
-   **PUCT/Q normalization**: common MuZero implementations normalize Q estimates during selection; this implementation does not include a full min-max normalization scheme.

#### Value-loss instability and mitigation

During experimentation, the value loss could become unstable in some configurations. This repo includes two practical mitigations:

1. Value-target scaling/clipping (optional)  
   If `game_settings.use_reward_shaping: true`, the replay buffer scales/clips value targets to a bounded range:

-   negative targets are clamped to 0,
-   targets are scaled by `max_episode_steps`,
-   then clipped to `[0, 1]`.

2. Gradient clipping (global norm)  
   The optimizer applies global-norm gradient clipping (Optax), which helps prevent catastrophic updates when targets or gradients spike.

These mitigations improved the training robustness in practice, but the implementation remains simplified relative to full MuZero.

### Delivery constraints

This project was completed under course time constraints. The implementation prioritizes:

-   a working end-to-end pipeline,
-   readable modular structure,
-   demonstrable learning on CartPole,
    over a fully faithful reproduction of all MuZero details.

---

## Future work

If extending this project, high-impact next steps would be:

-   Add root Dirichlet noise for exploration during self-play
-   Replace learned-model rollouts with a more canonical MuZero backup scheme
-   Add Q normalization (min-max) in PUCT selection
-   Add reanalysis to refresh targets with updated networks
-   Decouple TD horizon (n-step) from unroll length K
-   Improve evaluation + reproducibility tooling (multiple seeds, fixed evaluation protocol)

---
