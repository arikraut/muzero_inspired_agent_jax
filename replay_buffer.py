"""
replay_buffer.py

Episode-level replay buffer for MuZero.

Stores complete episodes (lists of transition dicts) and samples training
sequences of length K (unroll_steps), producing:
- NNr input (stacked observation history)
- actions, rewards, policies
- target values computed as n-step returns with bootstrap
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional
import logging
import time
from utils import stack_and_preprocess_history

logger = logging.getLogger(__name__)


class EpisodeBuffer:
    """
    Stores complete episodes of gameplay experience and samples sequences for training.

    Each step dict is expected to include:
      - obs, action, reward, policy, value
      - terminated / truncated flags
    """

    def __init__(self, config: Dict):
        """Initializes the buffer based on configuration settings."""
        self.cfg = config
        buffer_cfg = self.cfg.get("episode_buffer", {})
        rlm_cfg = self.cfg.get("rlm", {})
        nn_cfg = self.cfg.get("neural_network", {})

        self.buffer_size: int = int(buffer_cfg.get("buffer_size", 1000))
        self.buffer: deque[List[Dict]] = deque(maxlen=self.buffer_size)

        self.discount_factor: float = float(rlm_cfg.get("discount_factor", 0.997))
        self.unroll_steps: int = int(nn_cfg.get("unroll_steps", 5))
        self.representation_input_states: int = int(
            nn_cfg.get("representation_input_states", 1)
        )

        logger.info(
            f"EpisodeBuffer Initialized: Max episodes={self.buffer_size}, "
            f"Unroll steps (K/n)={self.unroll_steps}, "
            f"NNr history={self.representation_input_states}, "
        )

    def save_episode(self, episode: List[Dict]):
        """Adds a completed episode (list of step dictionaries) to the buffer."""
        if not episode:
            logger.warning("Attempted to save an empty episode. Skipping.")
            return

        required_len = self.unroll_steps + 1
        if len(episode) < required_len:
            logger.debug(
                f"Episode too short ({len(episode)} < {required_len}). Skipping save."
            )
            return

        logger.debug(f"Saving episode of length {len(episode)} to buffer.")
        self.buffer.append(episode)

        if len(self.buffer) % 100 == 0 or len(self.buffer) == self.buffer_size:
            logger.info(
                f"Episode Buffer size: {len(self.buffer)} / {self.buffer_size} episodes."
            )

    def _calculate_target_value(self, episode: List[Dict], state_index: int) -> float:
        """
        Calculates the n-step target value z_t for training.

        z_t = R_{t+1} + gamma*R_{t+2} + ... + gamma^(n-1)*R_{t+n} + gamma^n * v_{t+n}

        If the episode terminates/truncates before t+n, bootstrapping is omitted.
        If reward shaping is enabled, the return is scaled/clipped to [0, 1].
        """
        n = self.unroll_steps
        target_value_raw = 0.0
        episode_len = len(episode)

        for k in range(n):
            step_index = state_index + k
            if step_index < episode_len:
                reward = float(episode[step_index].get("reward", 0.0))
                target_value_raw += (self.discount_factor**k) * reward

                terminated_at_step = episode[step_index].get("terminated", False)
                truncated_at_step = episode[step_index].get("truncated", False)
                if terminated_at_step or truncated_at_step:
                    logger.debug(
                        f"Target value calc hit terminal/truncated state at step k={k} (index {step_index}). Bootstrap is 0."
                    )
                    final_target_value = target_value_raw
                    logger.debug(
                        f"  Calculated target z_{state_index}={final_target_value:.4f} (terminated at step {k})"
                    )
                    return float(final_target_value)
            else:
                logger.debug(
                    f"Target value calc truncated at step k={k} due to episode end (len {episode_len})."
                )
                final_target_value = target_value_raw
                logger.debug(
                    f"  Calculated target z_{state_index}={final_target_value:.4f} (episode ended early)"
                )
                return float(final_target_value)

        bootstrap_index = state_index + n
        bootstrap_value = 0.0
        if bootstrap_index < episode_len:
            bootstrap_value = float(episode[bootstrap_index].get("value", 0.0))

        target_value_raw += (self.discount_factor**n) * bootstrap_value

        if self.cfg["game_settings"].get("use_reward_shaping", False):
            max_possible_return = float(self.cfg["rlm"]["max_episode_steps"])
            if max_possible_return <= 0:
                max_possible_return = 1.0

            clamped_target_value = max(0.0, target_value_raw)
            scaled_target_value = clamped_target_value / max_possible_return

            final_target_value = np.clip(scaled_target_value, 0.0, 1.0)
            logger.debug(
                f"  Calculated raw target z_{state_index}={target_value_raw:.4f}, "
                f"Scaled target={final_target_value:.4f} (MaxReturn={max_possible_return})"
            )
        else:
            final_target_value = target_value_raw
            logger.debug(
                f"  Calculated raw target z_{state_index}={target_value_raw:.4f}, "
            )

        return float(final_target_value)

    def sample(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Samples a batch of training sequences from stored episodes.

        Returns arrays:
          - nnr_input: (B, ...) stacked + preprocessed observation history
          - actions: (B, K)
          - rewards: (B, K)
          - policies: (B, K, A)
          - values: (B, K) computed n-step targets
        """
        if not self.buffer:
            logger.warning("Cannot sample: Episode buffer is empty.")
            return None

        buffer_list = list(self.buffer)
        num_episodes = len(buffer_list)
        if num_episodes == 0:
            logger.warning("Cannot sample: Buffer empty after converting to list.")
            return None

        replace = num_episodes < batch_size
        if replace:
            logger.warning(
                f"Sampling episodes with replacement (Buffer {num_episodes} < Batch {batch_size})."
            )
        episode_indices = np.random.choice(
            num_episodes, size=batch_size, replace=replace
        )

        start_indices = []
        valid_episode_indices = []
        for ep_idx in episode_indices:
            episode = buffer_list[ep_idx]
            episode_len = len(episode)

            max_start_index_exclusive = episode_len - self.unroll_steps
            min_start_index = 0

            if max_start_index_exclusive > min_start_index:
                start_t = np.random.randint(min_start_index, max_start_index_exclusive)
                start_indices.append(start_t)
                valid_episode_indices.append(ep_idx)
            else:
                logger.debug(
                    f"Episode {ep_idx} (len {episode_len}) too short for unroll K={self.unroll_steps}. Skipping."
                )

        num_valid_samples = len(valid_episode_indices)
        if num_valid_samples == 0:
            logger.warning(
                f"No episodes long enough to sample a sequence (K={self.unroll_steps}). Batch empty."
            )
            return None
        if num_valid_samples < batch_size:
            logger.warning(
                f"Only found {num_valid_samples} valid sequences (requested {batch_size}). Using smaller batch."
            )
            batch_size = num_valid_samples

        batch_nnr_inputs = []
        batch_actions = []
        batch_rewards = []
        batch_policies = []
        batch_target_values = []

        for i in range(batch_size):
            ep_idx = valid_episode_indices[i]
            start_index = start_indices[i]
            episode = buffer_list[ep_idx]
            episode_len = len(episode)

            raw_obs_history_list = []
            successful_history = True
            for hist_step_offset in range(self.representation_input_states):
                obs_idx = start_index - hist_step_offset
                actual_obs_idx = max(0, obs_idx)
                if actual_obs_idx >= episode_len:
                    logger.error(
                        f"History index {actual_obs_idx} out of bounds. Ep {ep_idx}, len {episode_len}, start_t {start_index}."
                    )
                    successful_history = False
                    break

                obs = episode[actual_obs_idx].get("obs")
                if obs is None:
                    logger.error(
                        f"Missing 'obs' at index {actual_obs_idx} in ep {ep_idx}."
                    )
                    successful_history = False
                    break

                raw_obs_history_list.insert(0, obs)

            if not successful_history:
                continue

            try:
                initial_nnr_input = stack_and_preprocess_history(
                    raw_obs_history_list, self.cfg
                )
            except (ValueError, Exception) as e:
                logger.error(
                    f"Error processing NNr history (ep {ep_idx}, start {start_index}): {e}. Skipping sample.",
                    exc_info=True,
                )
                continue

            actions_sequence = []
            rewards_sequence = []
            policies_sequence = []
            target_values_sequence = []
            valid_sequence = True

            for k in range(self.unroll_steps):
                current_step_index = start_index + k
                if current_step_index >= episode_len:
                    logger.error(
                        f"Logic Error: Index {current_step_index} OOB during sequence extraction."
                    )
                    valid_sequence = False
                    break

                transition = episode[current_step_index]
                required_keys = ["action", "reward", "policy"]
                if not all(key in transition for key in required_keys):
                    missing = set(required_keys) - set(transition.keys())
                    logger.error(
                        f"Missing keys {missing} in transition at index {current_step_index}, ep {ep_idx}."
                    )
                    valid_sequence = False
                    break

                actions_sequence.append(transition["action"])
                rewards_sequence.append(float(transition["reward"]))

                policy_array = np.array(transition["policy"], dtype=np.float32)
                policies_sequence.append(policy_array)

                try:
                    target_val = self._calculate_target_value(
                        episode, current_step_index
                    )
                    target_values_sequence.append(target_val)
                except Exception as e_val:
                    logger.exception(
                        f"Error calculating target value z_{current_step_index}, ep {ep_idx}: {e_val}"
                    )
                    valid_sequence = False
                    break

            if not valid_sequence:
                continue

            batch_nnr_inputs.append(initial_nnr_input)
            batch_actions.append(actions_sequence)
            batch_rewards.append(rewards_sequence)
            batch_policies.append(policies_sequence)
            batch_target_values.append(target_values_sequence)

        final_batch_size = len(batch_nnr_inputs)
        if final_batch_size == 0:
            logger.warning(
                "No valid sequences could be processed in this batch attempt."
            )
            return None
        if final_batch_size < batch_size:
            logger.warning(
                f"Final batch size {final_batch_size} is smaller than initially requested {batch_size}."
            )

        try:
            batch = {
                "nnr_input": np.array(batch_nnr_inputs, dtype=np.float32),
                "actions": np.array(batch_actions, dtype=np.int32),
                "rewards": np.array(batch_rewards, dtype=np.float32),
                "policies": np.array(batch_policies, dtype=np.float32),
                "values": np.array(batch_target_values, dtype=np.float32),
            }
            logger.debug(
                f"Final Batch Shapes: NNR={batch['nnr_input'].shape}, A={batch['actions'].shape}, "
                f"R={batch['rewards'].shape}, P={batch['policies'].shape}, V={batch['values'].shape}"
            )
            return batch
        except ValueError as e_stack:
            logger.error(
                f"Error during final batch stacking: {e_stack}. Check sequence extraction logic.",
                exc_info=True,
            )
            if batch_actions:
                logger.error(
                    f"  Example Actions seq shape: {np.array(batch_actions[0]).shape}"
                )
            if batch_policies:
                logger.error(
                    f"  Example Policies seq shape: {np.array(batch_policies[0]).shape}"
                )
            return None
        except Exception as e_final:
            logger.exception(f"Unexpected error during final batch assembly: {e_final}")
            return None
