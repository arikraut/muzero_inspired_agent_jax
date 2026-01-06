# state_manager.py
# Interface layer between the environment’s concrete state/actions and MuZero’s learned model (h/g/f).

import jax
import jax.numpy as jnp
import logging
from typing import List, Tuple
from game_simulator import GymnasiumEnvManager
from nn import NeuralNetworkManager

Action = int

logger = logging.getLogger(__name__)


class AbstractStateManager:
    """
    Adapter between the environment and MuZero’s networks.

    Responsibilities:
      - Convert preprocessed observation history to a latent state (h / representation)
      - Produce policy + value from a latent state (f / prediction)
      - Predict next latent state + reward given (state, action) (g / dynamics)
      - Provide legal actions via the environment manager
    """

    def __init__(
        self, env_manager: "GymnasiumEnvManager", nn_manager: "NeuralNetworkManager"
    ):
        """Store environment and network managers used for model queries and legal actions."""
        self.env = env_manager
        self.NN = nn_manager

    def convert_game_states_to_abstract(self, nnr_input: jnp.ndarray) -> jnp.ndarray:
        """
        Run the representation network (h): observation history -> latent state.
        """
        logger.debug("Computing latent state via NNr (h)...")
        latent_state = self.NN.representation_forward(self.NN.params, nnr_input)
        logger.debug(f"Latent state shape: {latent_state.shape}")
        return latent_state

    def get_legal_actions(self) -> List[Action]:
        """Return legal actions from the environment manager."""
        return self.env.get_legal_actions()

    def get_policy_and_value(
        self, abstract_state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the prediction network (f): latent state -> (policy probabilities, value).
        """
        logger.debug("Computing policy/value via NNp (f)...")
        policy_logits, value = self.NN.prediction_forward(
            self.NN.params, abstract_state
        )
        policy_probs = jax.nn.softmax(policy_logits, axis=-1)
        logger.debug(
            f"Policy probs shape: {policy_probs.shape}, Value shape: {value.shape}"
        )
        return policy_probs, value

    def get_next_abstract_state_and_reward(
        self, abstract_state: jnp.ndarray, action: Action
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run the dynamics network (g): (latent state, action) -> (next latent state, reward).
        """
        logger.debug(f"Computing next state/reward via NNd (g), action={action}...")
        next_state, reward = self.NN.dynamics_forward(
            self.NN.params, abstract_state, action
        )
        logger.debug(
            f"Next state shape: {next_state.shape}, Reward shape: {reward.shape}"
        )
        return next_state, reward
