"""
muzero.py

Main training orchestration for a MuZero-style agent.

Defines ReinforcementLearningManager, which coordinates:
- Environment interaction (data collection via self-play episodes)
- MCTS planning in latent space
- Replay buffer storage
- Network training steps
- Checkpointing and lightweight training statistics logging
"""

import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import traceback
import logging
import os
import csv

from game_simulator import GymnasiumEnvManager
from nn import NeuralNetworkManager
from utils import (
    stack_and_preprocess_history,
    load_config,
    finalize_config,
)
from mcts import UMCTS
from replay_buffer import EpisodeBuffer
from state_manager import AbstractStateManager

Action = int
logger = logging.getLogger(__name__)


class ReinforcementLearningManager:
    """
    Orchestrates the MuZero training loop.

    High-level flow:
      1) Initialize environment + MuZero components (NN, abstract state manager, MCTS, replay buffer)
      2) Repeatedly:
         - Collect episodes using MCTS for action selection
         - Train networks from replay buffer batches
         - Periodically save checkpoints
         - Log basic training statistics (CSV)
    """

    def __init__(
        self,
        config_path: str,
        nn_manager_class=NeuralNetworkManager,
        abstract_manager_class=AbstractStateManager,
        env_manager_class=GymnasiumEnvManager,
        buffer_class=EpisodeBuffer,
        umcts_class=UMCTS,
    ):
        """Initializes the ReinforcementLearningManager and all its components."""
        logger.info("Initializing ReinforcementLearningManager...")
        logger.info(f"Loading base configuration from: {config_path}")
        base_config = load_config(config_path)

        logger.info("Initializing Environment Manager...")
        game_cfg = base_config.get("game_settings", {})
        try:
            env_creation_kwargs = game_cfg.get("env_kwargs", {})
            logger.info(f"  Passing extra kwargs to gym.make: {env_creation_kwargs}")

            self.env = env_manager_class(
                game_name=game_cfg.get("env_name", "CartPole-v1"),
                render_mode=game_cfg.get("render_mode", None),
                **env_creation_kwargs,
            )
        except Exception as e_env:
            logger.exception(
                f"Fatal error initializing Environment Manager: {e_env}. Exiting."
            )
            raise

        logger.info("Finalizing Configuration...")
        try:
            self.cfg = finalize_config(base_config, self.env)
        except Exception as e_cfg:
            logger.exception(f"Fatal error finalizing configuration: {e_cfg}. Exiting.")
            raise

        logger.info(
            "Initializing NN Manager, Abstract State Manager, Buffer, and MCTS..."
        )
        try:
            self.nnm = nn_manager_class(self.cfg)
            self.abstract = abstract_manager_class(self.env, self.nnm)
            self.buffer = buffer_class(self.cfg)
            self.umcts = umcts_class(self.nnm, self.abstract, self.cfg)
        except Exception as e_comp:
            logger.exception(
                f"Fatal error initializing core MuZero components: {e_comp}. Exiting."
            )
            raise

        try:
            self.rlm_cfg = self.cfg["rlm"]
            self.nn_cfg = self.cfg["neural_network"]
            self.train_cfg = self.cfg["training"]
            self.game_cfg = self.cfg["game_settings"]

            self.representation_input_states = int(
                self.nn_cfg["representation_input_states"]
            )
            self.action_temperature = float(self.rlm_cfg.get("action_temperature", 1.0))
            self.max_episode_steps = int(self.rlm_cfg["max_episode_steps"])
            self.num_episodes_per_train = int(
                self.rlm_cfg.get("episodes_per_train_step", 1)
            )
            self.total_training_steps = int(self.rlm_cfg["total_training_steps"])
            self.batch_size = int(self.train_cfg["batch_size"])

            self.checkpoint_dir = self.train_cfg.get(
                "checkpoint_dir", "muzero_checkpoints"
            )
            self.save_interval = int(self.train_cfg.get("save_checkpoint_interval", 0))
            self.load_step = int(self.train_cfg.get("load_checkpoint_step", 0))

            self.plot_log_filename = self.rlm_cfg.get(
                "plot_log_filename", "training_stats.csv"
            )
            self.plot_log_filepath = None
            self._plot_log_file = None
            self._plot_log_writer = None

        except KeyError as e_param:
            logger.exception(
                f"Fatal error accessing key in finalized config: {e_param}."
            )
            raise
        except Exception as e_param_other:
            logger.exception(
                f"Fatal error reading training parameters: {e_param_other}"
            )
            raise

        self.episodes_collected = 0
        self.training_steps_done = 0
        self.total_env_steps = 0

        logger.info("ReinforcementLearningManager Initialization Complete.")

    def run_training_loop(self):
        """Executes the main MuZero training loop."""
        logger.info("\n" + "=" * 50)
        logger.info("Starting MuZero Training Loop")
        logger.info(f" Target Training Steps: {self.total_training_steps}")
        logger.info(f" Environment: {self.game_cfg.get('env_name', 'N/A')}")
        logger.info(f" Checkpoint Directory: {self.checkpoint_dir}")

        # CSV logging for lightweight plotting/monitoring.
        self.plot_log_filepath = os.path.join(
            self.checkpoint_dir, self.plot_log_filename
        )
        logger.info(f" Plotting Stats Log: {self.plot_log_filepath}")
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            file_exists = os.path.isfile(self.plot_log_filepath)
            needs_header = (not file_exists) or (
                os.path.getsize(self.plot_log_filepath) == 0
            )

            self._plot_log_file = open(
                self.plot_log_filepath, "a", newline="", encoding="utf-8", buffering=1
            )
            self._plot_log_writer = csv.writer(self._plot_log_file)

            if needs_header:
                self._plot_log_writer.writerow(
                    ["TrainingStep", "AvgEpisodeSteps", "AvgEpisodeReward"]
                )
                self._plot_log_file.flush()
        except IOError as e:
            logger.error(
                f"Could not open or write header to plot log file {self.plot_log_filepath}: {e}"
            )
            self._plot_log_file = None
            self._plot_log_writer = None

        logger.info("=" * 50 + "\n")
        start_time = time.time()

        # Optional checkpoint resume.
        if self.load_step > 0:
            logger.info(f"Attempting to load checkpoint from step {self.load_step}...")
            load_successful = self.nnm.load_state(self.checkpoint_dir, self.load_step)
            if load_successful:
                self.training_steps_done = self.load_step
                logger.info(
                    f"--- Checkpoint loaded. Resuming training from step {self.training_steps_done + 1}. ---"
                )
            else:
                logger.warning(
                    "--- Checkpoint loading failed. Starting training from scratch. ---"
                )
                self.training_steps_done = 0
        else:
            logger.info(
                "No checkpoint loading requested. Starting training from scratch."
            )
            self.training_steps_done = 0

        try:
            while self.training_steps_done < self.total_training_steps:
                loop_step_start_time = time.time()
                current_train_iter = self.training_steps_done + 1
                logger.info(
                    f"\n===== Training Iteration {current_train_iter}/{self.total_training_steps} ====="
                )

                # Phase 1: data collection (self-play episodes).
                logger.info(
                    f"[Phase 1] Collecting {self.num_episodes_per_train} episode(s)..."
                )
                collected_steps_this_phase = 0
                collected_episodes_this_phase = 0
                total_reward_this_phase = 0.0
                collection_start_time = time.time()

                for i in range(self.num_episodes_per_train):
                    episode_num = self.episodes_collected + 1
                    logger.info(f"  Running episode {episode_num}...")
                    episode_len, episode_reward = self.run_episode()

                    if episode_len > 0:
                        self.episodes_collected += 1
                        collected_episodes_this_phase += 1
                        collected_steps_this_phase += episode_len
                        total_reward_this_phase += episode_reward
                        self.total_env_steps += episode_len
                    else:
                        logger.warning(
                            f"  Episode {episode_num} failed or had 0 length."
                        )

                    if self.training_steps_done >= self.total_training_steps:
                        logger.info(
                            "Target training steps reached during data collection. Stopping."
                        )
                        break

                collection_duration = time.time() - collection_start_time
                avg_ep_len = (
                    (collected_steps_this_phase / max(1, collected_episodes_this_phase))
                    if collected_episodes_this_phase > 0
                    else 0
                )
                avg_ep_reward = (
                    (total_reward_this_phase / max(1, collected_episodes_this_phase))
                    if collected_episodes_this_phase > 0
                    else 0.0
                )
                logger.info(
                    f"  Collection phase finished ({collection_duration:.2f}s). "
                    f"Collected: {collected_episodes_this_phase} ep, {collected_steps_this_phase} steps. "
                    f"Avg Len: {avg_ep_len:.1f}, Avg Reward: {avg_ep_reward:.2f}. "
                    f"Total Env Steps: {self.total_env_steps}. Buffer: {len(self.buffer.buffer)} ep."
                )

                if self._plot_log_writer:
                    try:
                        self._plot_log_writer.writerow(
                            [current_train_iter, avg_ep_len, avg_ep_reward]
                        )
                    except Exception as e_log:
                        logger.error(f"Error writing to plot log file: {e_log}")

                if self.training_steps_done >= self.total_training_steps:
                    break

                # Phase 2: training (sample from replay buffer).
                if len(self.buffer.buffer) > 0:
                    logger.info(
                        f"[Phase 2] Training network (Batch Size: {self.batch_size})..."
                    )
                    train_start_time = time.time()
                    loss_dict = self.nnm.train_networks(
                        self.buffer, batch_size=self.batch_size
                    )
                    train_duration = time.time() - train_start_time

                    if loss_dict:
                        self.training_steps_done += 1
                        loss_str = ", ".join(
                            [f"{k}: {v:.4g}" for k, v in loss_dict.items()]
                        )
                        logger.info(
                            f"  Training step {self.training_steps_done} successful ({train_duration:.2f}s). Losses: {loss_str}"
                        )
                    else:
                        logger.error(
                            f"  Training step {current_train_iter} FAILED. Check logs."
                        )
                else:
                    logger.warning("Skipping training phase: Replay buffer is empty.")

                # Phase 3: checkpointing.
                if (
                    self.save_interval > 0
                    and self.training_steps_done > 0
                    and (self.training_steps_done % self.save_interval == 0)
                ):
                    logger.info(
                        f"Saving checkpoint at training step {self.training_steps_done}..."
                    )
                    self.nnm.save_state(self.checkpoint_dir, self.training_steps_done)

                loop_step_duration = time.time() - loop_step_start_time
                total_elapsed_sec = time.time() - start_time
                est_remaining_sec = 0
                if self.training_steps_done > 0:
                    avg_time_per_step = total_elapsed_sec / self.training_steps_done
                    remaining_steps = (
                        self.total_training_steps - self.training_steps_done
                    )
                    est_remaining_sec = avg_time_per_step * remaining_steps
                total_elapsed_hms = time.strftime(
                    "%H:%M:%S", time.gmtime(total_elapsed_sec)
                )
                est_remaining_hms = (
                    time.strftime("%H:%M:%S", time.gmtime(est_remaining_sec))
                    if est_remaining_sec > 0
                    else "N/A"
                )
                logger.info(
                    f"Iteration {current_train_iter} duration: {loop_step_duration:.2f}s. "
                    f"Total elapsed: {total_elapsed_hms}. Est. remaining: {est_remaining_hms}."
                )

            logger.info("\n" + "=" * 50)
            logger.info("Training Loop Finished")
            logger.info(f" Target Training Steps: {self.total_training_steps}")
            logger.info(f" Actual Training Steps Completed: {self.training_steps_done}")
            logger.info(f" Total Environment Steps: {self.total_env_steps}")
            logger.info(f" Total Episodes Collected: {self.episodes_collected}")
            total_duration_sec = time.time() - start_time
            total_duration_hms = time.strftime(
                "%H:%M:%S", time.gmtime(total_duration_sec)
            )
            logger.info(f" Total Duration: {total_duration_hms}")
            logger.info("=" * 50 + "\n")

        except Exception as e_loop:
            logger.exception(f"An error occurred during the training loop: {e_loop}")
        finally:
            if self._plot_log_file:
                try:
                    self._plot_log_file.close()
                    logger.info(f"Closed plot log file: {self.plot_log_filepath}")
                except Exception as e_close:
                    logger.error(f"Error closing plot log file: {e_close}")

            self.env.close()
            logger.info("Environment closed.")

    def run_episode(self, seed: Optional[int] = None) -> Tuple[int, float]:
        """
        Runs one environment episode and pushes transitions to the replay buffer.

        Returns:
            (episode_steps, total_episode_reward); returns (0, 0.0) if the episode fails to start.
        """
        episode_data: List[Dict] = []
        step_count = 0
        episode_reward = 0.0
        terminated, truncated = False, False

        # Reset environment and initialize history window for the representation network.
        try:
            init_raw_obs, info = self.env.reset(seed=seed)
            logger.debug(
                f"  Env reset. Initial obs shape: {getattr(init_raw_obs, 'shape', 'N/A')}"
            )

            raw_obs_history = deque(
                [init_raw_obs] * self.representation_input_states,
                maxlen=self.representation_input_states,
            )
            current_raw_obs = init_raw_obs
        except Exception as e:
            logger.exception(f"  Error resetting environment: {e}. Skipping episode.")
            return 0, 0.0

        while not (terminated or truncated):
            loop_start_time = time.time()

            if step_count >= self.max_episode_steps:
                logger.info(
                    f"  Episode truncated at step {step_count}: Reached max steps ({self.max_episode_steps})."
                )
                truncated = True
                break

            # Build representation input and compute abstract state s_t.
            try:
                nnr_input = stack_and_preprocess_history(
                    list(raw_obs_history), self.cfg
                )
                logger.debug(
                    f"  Step {step_count}: Prepared NNr input shape {nnr_input.shape}"
                )
            except (ValueError, Exception) as e_hist:
                logger.exception(
                    f"  Error preparing NNr input (step {step_count}): {e_hist}. Truncating."
                )
                truncated = True
                break

            try:
                abstract_state_t = self.abstract.convert_game_states_to_abstract(
                    jnp.asarray(nnr_input)
                )
                logger.debug(
                    f"  Step {step_count}: Got abstract state s_{step_count} shape {abstract_state_t.shape}"
                )
            except Exception as e_abs:
                logger.exception(
                    f"  Error running Representation Net (step {step_count}): {e_abs}. Truncating."
                )
                truncated = True
                break

            # Plan with MCTS to obtain policy π_t and value estimate v_t.
            policy = None
            mcts_value = 0.0
            try:
                self.umcts.run_search(abstract_state_t)
                policy, mcts_value = self.umcts.get_policy_and_value(
                    temperature=self.action_temperature
                )
                logger.debug(
                    f"  Step {step_count}: MCTS Policy pi_{step_count}={np.round(policy, 3)}, Value v_{step_count}={mcts_value:.3f}"
                )
            except Exception as e_mcts:
                logger.exception(
                    f"  Error during MCTS search/get_policy (step {step_count}): {e_mcts}."
                )

            # Sample action from π_t (fallback to random legal action on error).
            action = -1
            policy_np = None
            if policy is not None:
                try:
                    if len(policy) != len(self.umcts.legal_actions):
                        raise ValueError(
                            f"Policy length ({len(policy)}) != legal actions ({len(self.umcts.legal_actions)})."
                        )
                    policy_np = np.array(policy, dtype=np.float32)
                    policy_sum = np.sum(policy_np)
                    if abs(policy_sum - 1.0) > 1e-5:
                        logger.debug(
                            f"  Normalizing MCTS policy (sum={policy_sum:.4f})."
                        )
                        policy_np /= max(policy_sum, 1e-8)
                    action = np.random.choice(self.umcts.legal_actions, p=policy_np)
                    logger.debug(
                        f"  Step {step_count}: Sampled action a_{step_count}={action} from MCTS policy."
                    )
                except (ValueError, Exception) as e_sample:
                    logger.warning(
                        f"  Error sampling action from MCTS policy (policy={policy}): {e_sample}. Choosing random action."
                    )
                    if self.umcts.legal_actions:
                        action = np.random.choice(self.umcts.legal_actions)
                    else:
                        logger.error(
                            "  Cannot select fallback action: No legal actions! Truncating."
                        )
                        truncated = True
                        break
            else:
                logger.warning("  MCTS policy was None. Choosing random action.")
                if self.umcts.legal_actions:
                    action = np.random.choice(self.umcts.legal_actions)
                else:
                    logger.error(
                        "  Cannot select fallback action: No legal actions! Truncating."
                    )
                    truncated = True
                    break

            if action == -1:
                logger.error(
                    "  Failed to select a valid action after MCTS/fallbacks. Truncating."
                )
                truncated = True
                break

            transition_data = {
                "obs": current_raw_obs,
                "action": action,
                "policy": (
                    policy_np
                    if policy_np is not None
                    else np.ones(len(self.umcts.legal_actions), dtype=np.float32)
                    / max(1, len(self.umcts.legal_actions))
                ),
                "value": float(mcts_value),
            }

            # Step the real environment and store transition outcome.
            try:
                next_raw_obs, reward, term_step, trunc_step, info = self.env.step(
                    action
                )
                episode_reward += reward

                terminated = terminated or term_step
                is_truncated_this_step = trunc_step or (
                    step_count + 1 >= self.max_episode_steps
                )
                truncated = truncated or is_truncated_this_step

                transition_data["reward"] = float(reward)
                transition_data["terminated"] = term_step
                transition_data["truncated"] = is_truncated_this_step

                episode_data.append(transition_data)
                logger.debug(
                    f"  Step {step_count}: Env step result: Reward R_{step_count+1}={reward:.2f}, Term={term_step}, Trunc={is_truncated_this_step}"
                )

            except Exception as e_step:
                logger.exception(
                    f"  Error stepping environment with action {action} (step {step_count}): {e_step}. Truncating."
                )
                truncated = True
                break

            raw_obs_history.append(next_raw_obs)
            current_raw_obs = next_raw_obs
            step_count += 1

            if self.game_cfg.get("render_mode") == "human":
                try:
                    self.env.render()
                except Exception as e_render:
                    logger.warning(f"  Failed to render environment: {e_render}")

            loop_end_time = time.time()
            logger.debug(
                f"  Step {step_count-1} processing duration: {loop_end_time - loop_start_time:.4f}s"
            )

        final_reason = (
            "Terminated" if terminated else ("Truncated" if truncated else "Unknown")
        )
        logger.info(
            f"  Episode finished. Steps: {step_count}, Total Reward: {episode_reward:.2f}, Reason: {final_reason}"
        )

        if episode_data:
            self.buffer.save_episode(episode_data)
        else:
            logger.warning(
                "  No transition data collected for this episode. Buffer not updated."
            )

        return step_count, episode_reward
