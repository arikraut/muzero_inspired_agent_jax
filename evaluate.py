# evaluate.py
import argparse
import os
import sys
import time
import logging
import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from mcts import UMCTS
from state_manager import AbstractStateManager
from game_simulator import GymnasiumEnvManager
from nn import NeuralNetworkManager
from utils import load_config, finalize_config, stack_and_preprocess_history

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

Action = int


def run_evaluation(config_path: str, checkpoint_step: int, num_episodes: int):
    """Loads a trained model and runs episodes with rendering."""
    logger.info(f"Loading base config from: {config_path}")
    base_config = load_config(config_path)

    render_mode = "human"
    logger.info(f"Setting render_mode to: {render_mode}")
    if "game_settings" not in base_config:
        base_config["game_settings"] = {}
    base_config["game_settings"]["render_mode"] = render_mode
    if "env_kwargs" not in base_config["game_settings"]:
        base_config["game_settings"]["env_kwargs"] = {}

    try:
        logger.info("Initializing Environment Manager...")
        env_manager = GymnasiumEnvManager(
            game_name=base_config["game_settings"]["env_name"],
            render_mode=render_mode,
            **(base_config["game_settings"].get("env_kwargs", {})),
        )

        logger.info("Finalizing Configuration...")
        cfg = finalize_config(base_config, env_manager)

        logger.info("Initializing Neural Network Manager...")
        nnm = NeuralNetworkManager(cfg)

        checkpoint_dir = cfg["training"]["checkpoint_dir"]
        logger.info(
            f"Attempting to load checkpoint step {checkpoint_step} from {checkpoint_dir}"
        )
        if not nnm.load_state(checkpoint_dir, checkpoint_step):
            logger.error("Failed to load checkpoint. Exiting.")
            env_manager.close()
            return
        logger.info(f"Successfully loaded checkpoint step {checkpoint_step}.")

        logger.info("Initializing Abstract State Manager and UMCTS...")
        asm = AbstractStateManager(env_manager, nnm)
        umcts = UMCTS(nnm, asm, cfg)

        representation_input_states = cfg["neural_network"][
            "representation_input_states"
        ]
        max_episode_steps = cfg["rlm"]["max_episode_steps"]

        total_rewards = []
        for i in range(num_episodes):
            logger.info(f"\n--- Starting Evaluation Episode {i+1}/{num_episodes} ---")
            episode_reward = 0.0
            step_count = 0
            terminated, truncated = False, False

            try:
                init_raw_obs, info = env_manager.reset()
                raw_obs_history = deque(
                    [init_raw_obs] * representation_input_states,
                    maxlen=representation_input_states,
                )
            except Exception as e:
                logger.exception(f"Env reset error: {e}")
                continue

            while not (terminated or truncated):
                if step_count >= max_episode_steps:
                    logger.info(f"Episode truncated at max steps {max_episode_steps}.")
                    truncated = True

                try:
                    nnr_input = stack_and_preprocess_history(list(raw_obs_history), cfg)
                except Exception as e:
                    logger.exception(f"History prep error step {step_count}: {e}")
                    truncated = True
                    break

                try:
                    abstract_state = asm.convert_game_states_to_abstract(
                        jnp.asarray(nnr_input)
                    )
                except Exception as e:
                    logger.exception(f"Abstract state error step {step_count}: {e}")
                    truncated = True
                    break

                try:
                    umcts.run_search(abstract_state)
                    policy, _ = umcts.get_policy_and_value(temperature=0.0)
                except Exception as e:
                    logger.exception(f"MCTS error step {step_count}: {e}")
                    legal_actions = env_manager.get_legal_actions()
                    action = np.random.choice(legal_actions) if legal_actions else 0
                    logger.warning("MCTS failed, taking random action.")
                else:
                    policy_np = np.array(policy)
                    action = int(np.argmax(policy_np))
                    legal_actions = env_manager.get_legal_actions()
                    if action not in legal_actions:
                        logger.warning(
                            f"Argmax action {action} not in legal actions {legal_actions}. Choosing legal action."
                        )
                        action = legal_actions[0] if legal_actions else 0

                try:
                    next_raw_obs, reward, term_step, trunc_step, info = (
                        env_manager.step(int(action))
                    )
                    episode_reward += reward
                except Exception as e:
                    logger.exception(
                        f"Env step error step {step_count}, action {action}: {e}"
                    )
                    truncated = True
                    break

                terminated = terminated or term_step
                is_max_steps = step_count + 1 >= max_episode_steps
                truncated = truncated or trunc_step or is_max_steps

                raw_obs_history.append(next_raw_obs)
                step_count += 1

                time.sleep(0.05)

            final_reason = (
                "Terminated"
                if terminated
                else ("Truncated" if truncated else "Max Steps Reached")
            )
            logger.info(
                f"  Eval Ep {i+1} ended. Steps:{step_count}, Reward:{episode_reward:.2f}, Reason:{final_reason}"
            )
            total_rewards.append(episode_reward)

        env_manager.close()
        logger.info("\n--- Evaluation Finished ---")
        if total_rewards:
            logger.info(
                f"Average reward over {len(total_rewards)} episodes: "
                f"{np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}"
            )
        else:
            logger.warning("No episodes completed successfully.")

    except Exception as e:
        logger.critical("An critical error occurred during evaluation setup or run.")
        logger.exception(e)
        if "env_manager" in locals() and env_manager:
            env_manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained MuZero model.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file used for training.",
    )
    parser.add_argument(
        "-s",
        "--checkpoint_step",
        type=int,
        required=True,
        help="The training step number of the checkpoint to load.",
    )
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at '{args.config}'")
        sys.exit(1)

    run_evaluation(args.config, args.checkpoint_step, args.num_episodes)
