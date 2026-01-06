# game_simulator.py
# This file defines a class to manage interactions with a Gymnasium environment.
# Its primary purpose is to provide a consistent interface for the MuZero agent
# to reset the environment, take steps, and retrieve basic information like
# legal actions and observation/action space specifications.
# Crucially, this manager deals with the *raw* environment interactions;
# observation preprocessing (like normalization, resizing, frame stacking)
# is handled elsewhere (e.g., in `utils.py` and the `EpisodeBuffer`).

import gymnasium as gym
import numpy as np
from typing import Tuple, List, Any, Optional, Dict, Union
import ale_py
import logging

Observation = Union[np.ndarray, Any]
Info = Dict[str, Any]
Action = int

logger = logging.getLogger(__name__)


class GymnasiumEnvManager:
    """
    Manages interaction with a standard Gymnasium environment.

    This class acts as a thin wrapper around a `gymnasium.Env` instance.
    It initializes the environment based on a given name and provides
    methods to reset the environment, execute steps, retrieve legal actions,
    and access observation/action space information.

    It separates the raw environment interaction logic from the agent's
    decision-making (MCTS) and learning (network updates, buffer sampling)
    processes, as well as observation preprocessing.
    """

    def __init__(self, game_name: str, render_mode: Optional[str] = None, **kwargs):
        """
        Initializes the Gymnasium environment manager.

        Args:
            game_name (str): The official name of the Gymnasium environment
                             (e.g., "ALE/Pong-v5", "CartPole-v1").
            render_mode (Optional[str]): The rendering mode for the environment.
                                         Common options:
                                         - None (default): No rendering (headless mode for training).
                                         - "human": Render to a window for visualization.
                                         - "rgb_array": Render to a NumPy array (less common for direct use here).
            **kwargs: Additional keyword arguments to pass to `gymnasium.make()`.

        Raises:
            Exception: If `gymnasium.make()` fails to create the environment.
        """
        self.env_name = game_name
        logger.info(
            f"Initializing Gymnasium environment: '{game_name}' with render_mode='{render_mode}'..."
        )
        try:
            self.env = gym.make(game_name, render_mode=render_mode, **kwargs)
            logger.info(
                f"Gymnasium environment '{game_name}' initialized successfully."
            )
            logger.info(f"  Raw Observation Space: {self.env.observation_space}")
            logger.info(f"  Raw Action Space: {self.env.action_space}")
        except Exception as e:
            logger.exception(
                f"Fatal Error: Failed to initialize Gymnasium environment '{game_name}': {e}"
            )
            raise

        self._action_space_size: Optional[int] = None
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self._action_space_size = self.env.action_space.n
            logger.info(
                f"  Detected Discrete action space with size: {self._action_space_size}"
            )
        else:
            logger.warning(
                f"Environment '{game_name}' has a non-Discrete action space "
                f"(type: {type(self.env.action_space)}). This MuZero implementation "
                f"is designed for Discrete action spaces. MCTS or action selection might fail."
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        """
        Resets the environment to its initial state.

        Args:
            seed (Optional[int]): An optional seed for the environment's random
                                  number generator to ensure reproducibility.

        Returns:
            Tuple[Observation, Info]: A tuple containing:
                - The initial raw observation from the environment.
                - An info dictionary with auxiliary information from the reset.
        """
        logger.debug(f"Resetting environment '{self.env_name}' with seed={seed}...")
        try:
            obs, info = self.env.reset(seed=seed)
            logger.debug("Environment reset complete.")
            return obs, info
        except Exception as e:
            logger.exception(f"Error during environment reset: {e}")
            raise

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Info]:
        """
        Executes a single time step in the environment using the provided action.

        Args:
            action (Action): The action selected by the agent (must be valid within
                             the environment's action space). For Discrete spaces,
                             this is typically an integer index.

        Returns:
            Tuple[Observation, float, bool, bool, Info]: A tuple containing:
                - next_obs (Observation): The raw observation after taking the action.
                - reward (float): The reward received from the environment for this step.
                - terminated (bool): True if the episode ended naturally (e.g., game over, goal reached).
                                     The final observation `next_obs` is valid in this case.
                - truncated (bool): True if the episode ended prematurely due to external factors
                                    (e.g., time limit reached, agent went out of bounds).
                                    The final observation `next_obs` is valid in this case.
                - info (Info): A dictionary containing auxiliary diagnostic information.
                               May be empty or contain environment-specific data.
        """
        logger.debug(f"Executing step with action: {action}")
        try:
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            reward = float(reward)
            logger.debug(
                f"  Step result: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
            )
            return next_obs, reward, terminated, truncated, info
        except Exception as e:
            logger.exception(f"Error during environment step with action {action}: {e}")
            raise

    def get_legal_actions(self) -> List[Action]:
        """
        Returns a list of all legal actions available in the current state.

        For standard `gymnasium.spaces.Discrete` environments, this typically
        returns a list of integers from 0 to n-1, where n is the total number
        of actions. It assumes all actions are always legal in Discrete spaces.

        Returns:
            List[Action]: A list of legal action identifiers (integers for Discrete space).

        Raises:
            TypeError: If the environment's action space is not `gym.spaces.Discrete`,
                       as determining legal actions for other spaces (e.g., continuous)
                       is not handled by this basic implementation.
        """
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            size = (
                self._action_space_size
                if self._action_space_size is not None
                else self.env.action_space.n
            )
            return list(range(size))

        err_msg = (
            f"Cannot determine legal actions: Environment action space must be "
            f"gym.spaces.Discrete for this method. Found type: {type(self.env.action_space)}"
        )
        logger.error(err_msg)
        raise TypeError(err_msg)

    @property
    def action_space_size(self) -> Optional[int]:
        """
        Returns the size (number of actions) of the action space if it's Discrete,
        otherwise returns None.
        """
        return self._action_space_size

    @property
    def observation_space(self) -> Optional[gym.Space]:
        """
        Returns the environment's raw observation space object (a `gym.Space` instance).
        This allows other components to inspect the structure, shape, and bounds
        of the raw observations.
        """
        return self.env.observation_space

    def close(self):
        """
        Closes the environment and releases any associated resources (e.g., rendering windows).
        Should be called when the environment is no longer needed.
        """
        logger.info(f"Closing Gymnasium environment '{self.env_name}'.")
        try:
            self.env.close()
        except Exception as e:
            logger.exception(f"Error during environment close: {e}")
