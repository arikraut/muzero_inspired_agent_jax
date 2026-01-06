# mcts.py
# MuZero-style Monte Carlo Tree Search (MCTS) implementation: Node + UMCTS.

import jax
import jax.numpy as jnp
import numpy as np
import time
import logging
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING

# Avoid circular imports at runtime while preserving type hints.
if TYPE_CHECKING:
    from state_manager import AbstractStateManager
    from nn import NeuralNetworkManager

Action = int

logger = logging.getLogger(__name__)

# ==============================================================================
# MCTS Node
# ==============================================================================


class Node:
    """A node in the MCTS tree, representing an abstract (latent) state."""

    def __init__(
        self,
        state: jnp.ndarray,
        parent: Optional["Node"],
        prior_prob: float,
        reward: float = 0.0,
    ):
        """
        Args:
            state: Abstract state represented by this node.
            parent: Parent node (None for root).
            prior_prob: Policy prior for the action that led here.
            reward: Predicted reward for the transition into this state.
        """
        self.state = state
        self.parent = parent
        self.children: Dict[Action, "Node"] = {}

        self.visit_count: int = 0
        self.q_value: float = 0.0

        self.reward: float = reward
        self.prior_prob: float = prior_prob

        self.action: Optional[Action] = None

    def update_q_value(self, new_return: float):
        """Update running average Q using the newly observed return."""
        self.q_value = (self.visit_count * self.q_value + new_return) / (
            self.visit_count + 1
        )

    def is_leaf(self) -> bool:
        """A node is a leaf if it has no children."""
        return not self.children

    def expand(
        self, legal_actions: List[Action], abstract_manager: "AbstractStateManager"
    ) -> float:
        """
        Expand a leaf node using the model:
          - Predict policy priors and value (NNp).
          - For each legal action, predict next state and reward (NNd) and create a child node.

        Returns:
            Value estimate V(s) for this node's state (0.0 on failure).
        """
        if not self.is_leaf():
            logger.warning("Attempted to expand a non-leaf node. Expansion skipped.")
            return 0.0

        logger.debug(
            f"Expanding leaf node (State: {self.state[:5]}... Prior: {self.prior_prob:.3f})"
        )

        try:
            policy_probs_jax, value_estimate_jax = (
                abstract_manager.get_policy_and_value(self.state)
            )

            value_estimate: float = float(jax.device_get(value_estimate_jax))
            policy_probs_cpu: np.ndarray = np.array(
                jax.device_get(policy_probs_jax), dtype=np.float32
            )

            if np.isnan(policy_probs_cpu).any() or np.isnan(value_estimate):
                logger.warning(
                    "NaN detected in prediction output during expansion. Child creation may fail."
                )

            if not legal_actions:
                logger.warning(
                    "Expansion called with no legal actions. Cannot create children."
                )

            for action in legal_actions:
                if action in self.children:
                    continue

                next_state, reward_jax = (
                    abstract_manager.get_next_abstract_state_and_reward(
                        self.state, action
                    )
                )
                reward_float: float = float(jax.device_get(reward_jax))

                if action < 0 or action >= len(policy_probs_cpu):
                    logger.error(
                        f"Action {action} out of bounds for policy array. Skipping child."
                    )
                    continue

                prior_p: float = float(policy_probs_cpu[action])
                if np.isnan(prior_p):
                    logger.warning(
                        f"NaN prior for action {action}. Setting prior to 0."
                    )
                    prior_p = 0.0

                child_node = Node(
                    state=next_state,
                    parent=self,
                    prior_prob=prior_p,
                    reward=reward_float,
                )
                child_node.action = action
                self.children[action] = child_node

                logger.debug(
                    f"  Created child for action {action}: Prior={prior_p:.3f}, Reward={reward_float:.3f}, State={next_state[:5]}..."
                )

            logger.debug(
                f"Node expansion complete. Predicted value V(s) = {value_estimate:.4f}"
            )
            if np.isnan(value_estimate):
                logger.warning(
                    "Expansion resulted in NaN value estimate. Returning 0.0."
                )
                return 0.0
            return value_estimate

        except Exception as e:
            logger.exception(f"Error during Node expansion: {e}")
            return 0.0

    def select_child(self, c_puct: float) -> Tuple[Action, "Node"]:
        """
        Select a child using PUCT: argmax_a [ Q + c_puct * P * sqrt(N_parent) / (1 + N_child) ].
        """
        if not self.children:
            raise ValueError(
                "Cannot select child: Node has no children (must be expanded first)."
            )

        best_action: Action = -1
        best_score: float = -float("inf")
        parent_visit_sqrt: float = np.sqrt(max(1, self.visit_count))

        for action, child in self.children.items():
            q_value = child.q_value
            u_score = (
                c_puct
                * child.prior_prob
                * (parent_visit_sqrt / (1 + child.visit_count))
            )
            puct_score = q_value + u_score

            if np.isnan(puct_score):
                logger.warning(f"NaN PUCT score for action {action}. Skipping.")
                continue

            if puct_score > best_score:
                best_score = puct_score
                best_action = action

        if best_action == -1:
            logger.error(
                "No best action found during PUCT selection. Scores might be invalid."
            )
            for action, child in self.children.items():
                logger.error(
                    f"  Action {action}: Q={child.q_value}, P={child.prior_prob}, N={child.visit_count}"
                )
            raise RuntimeError("Failed to select a valid child node using PUCT.")

        selected_child = self.children[best_action]
        logger.debug(
            f"Selected action {best_action} via PUCT (Score: {best_score:.4f})"
        )
        return best_action, selected_child


# ==============================================================================
# MuZero MCTS (UMCTS)
# ==============================================================================


class UMCTS:
    """
    MuZero-style MCTS operating in latent space via a learned model (h/g/f accessed through ASM).
    """

    def __init__(
        self,
        nn_manager: "NeuralNetworkManager",
        abstract_state_manager: "AbstractStateManager",
        config: Dict,
    ):
        self.nnm = nn_manager
        self.asm = abstract_state_manager
        self.config = config

        mcts_cfg = config.get("umcts", {})
        rlm_cfg = config.get("rlm", {})

        self.num_simulations: int = int(mcts_cfg.get("simulations", 50))
        self.c_puct: float = float(mcts_cfg.get("c_puct", 1.25))
        self.dmax: int = int(mcts_cfg.get("dmax", 10))
        self.discount_factor: float = float(rlm_cfg.get("discount_factor", 0.997))

        try:
            self.legal_actions: List[Action] = self.asm.get_legal_actions()
            if not self.legal_actions:
                logger.warning("UMCTS initialized with no legal actions!")
        except Exception:
            logger.exception("Failed to get legal actions during UMCTS init.")
            self.legal_actions = []

        self.root: Optional[Node] = None

        logger.info(
            f"UMCTS Initialized: Simulations/step={self.num_simulations}, "
            f"c_puct={self.c_puct}, Discount={self.discount_factor}, "
            f"dmax={self.dmax}, Num Legal Actions={len(self.legal_actions)}"
        )

    def _calculate_discounted_return(
        self, rewards_list: List[float], discount_factor: float
    ) -> float:
        """Compute discounted return from [r0, r1, ..., r_{d-1}, V_final]."""
        if not rewards_list:
            return 0.0

        current_return = rewards_list[-1]
        for i in range(len(rewards_list) - 2, -1, -1):
            current_return = rewards_list[i] + discount_factor * current_return
        return current_return

    def do_rollout(self, start_node: Node, max_rollout_depth: int) -> List[float]:
        """
        Model rollout from start_node for up to max_rollout_depth steps, returning
        [r0, r1, ..., r_{d-1}, V_final].
        """
        accumulated_rewards: List[float] = []
        current_abstract_state = start_node.state

        for _ in range(max_rollout_depth):
            try:
                policy_probs_jax, _ = self.asm.get_policy_and_value(
                    current_abstract_state
                )
                policy_probs_cpu = np.array(
                    jax.device_get(policy_probs_jax), dtype=np.float32
                )

                policy_sum = np.sum(policy_probs_cpu)
                if not np.isfinite(policy_sum) or policy_sum <= 1e-8:
                    logger.warning(
                        f"  Rollout: Invalid policy sum ({policy_sum:.3g}). Stopping rollout."
                    )
                    break
                if abs(policy_sum - 1.0) > 1e-5:
                    policy_probs_cpu /= policy_sum

                if not self.legal_actions:
                    logger.warning("  Rollout: No legal actions. Stopping rollout.")
                    break

                if len(policy_probs_cpu) != len(self.legal_actions):
                    logger.warning(
                        f"  Rollout: Policy length mismatch ({len(policy_probs_cpu)} vs {len(self.legal_actions)}). Choosing random action."
                    )
                    sampled_action = np.random.choice(self.legal_actions)
                else:
                    try:
                        sampled_action = np.random.choice(
                            self.legal_actions, p=policy_probs_cpu
                        )
                    except ValueError as e_choice:
                        logger.warning(
                            f"  Rollout: Error sampling action from policy: {e_choice}. Choosing random action."
                        )
                        sampled_action = np.random.choice(self.legal_actions)

                next_state_jax, reward_jax = (
                    self.asm.get_next_abstract_state_and_reward(
                        current_abstract_state, sampled_action
                    )
                )
                reward_cpu = float(jax.device_get(reward_jax))

                accumulated_rewards.append(reward_cpu)
                current_abstract_state = next_state_jax

            except Exception as e_rollout_step:
                logger.warning(
                    f"  Rollout: Exception during rollout step: {e_rollout_step}. Stopping rollout."
                )
                break

        try:
            _, final_value_jax = self.asm.get_policy_and_value(current_abstract_state)
            accumulated_rewards.append(float(jax.device_get(final_value_jax)))
        except Exception as e_final_val:
            logger.warning(
                f"  Rollout: Exception getting final value: {e_final_val}. Appending 0.0."
            )
            accumulated_rewards.append(0.0)

        logger.debug(
            f"    Rollout completed. Length={len(accumulated_rewards)-1} steps."
        )
        return accumulated_rewards

    def run_search(self, initial_abstract_state: jnp.ndarray):
        """
        Run MCTS for the given root abstract state.

        Each simulation performs:
          1) Selection (PUCT)
          2) Expansion (NN-guided)
          3) Rollout (model-based)
          4) Backpropagation (update N/Q)
        """
        start_time = time.time()
        logger.debug(f"Starting MCTS search ({self.num_simulations} simulations)...")

        current_legal_actions = self.legal_actions
        if not current_legal_actions:
            logger.error("MCTS search cannot run: No legal actions available.")
            self.root = Node(
                initial_abstract_state, parent=None, prior_prob=0.0, reward=0.0
            )
            self.root.visit_count = 1
            return

        self.root = Node(
            initial_abstract_state, parent=None, prior_prob=0.0, reward=0.0
        )

        try:
            logger.debug("Expanding root node...")
            _ = self.root.expand(current_legal_actions, self.asm)
            self.root.visit_count = 1
            logger.debug(
                f"Root expanded. Initial children created for actions: {list(self.root.children.keys())}"
            )
        except Exception as e:
            logger.exception(
                f"Critical Error: Failed to expand MCTS root node: {e}. Search aborted."
            )
            return

        if not self.root or not self.root.children:
            logger.warning(
                "MCTS root expansion resulted in no children. Search cannot proceed."
            )
            if self.root:
                self.root.visit_count = max(1, self.root.visit_count)
            return

        for sim in range(self.num_simulations):
            logger.debug(f"  Simulation {sim+1}/{self.num_simulations}...")
            current_node = self.root
            search_path: List[Node] = [current_node]

            # Selection
            logger.debug("    Phase 1: Selection")
            while not current_node.is_leaf():
                try:
                    action, current_node = current_node.select_child(self.c_puct)
                    logger.debug(
                        f"      Selected action {action} -> Node (State: {current_node.state[:5]}...)"
                    )
                    search_path.append(current_node)
                except (ValueError, RuntimeError) as e_select:
                    logger.error(
                        f"      Error during MCTS selection (Sim {sim+1}): {e_select}. Stopping simulation."
                    )
                    search_path = []
                    break
                except Exception as e_select_unexp:
                    logger.exception(
                        f"      Unexpected error during MCTS selection (Sim {sim+1}): {e_select_unexp}"
                    )
                    search_path = []
                    break

            if not search_path:
                logger.debug("    Simulation aborted during selection.")
                continue

            leaf_node = current_node
            logger.debug(f"    Reached leaf node (State: {leaf_node.state[:5]}...)")

            initial_backprop_value = 0.0
            backprop_path = search_path

            try:
                # Expansion
                logger.debug("    Phase 2: Expansion")
                leaf_value_estimate_fallback = leaf_node.expand(
                    current_legal_actions, self.asm
                )

                # Rollout
                logger.debug("    Phase 3: Rollout")
                if leaf_node.children:
                    child_actions = list(leaf_node.children.keys())
                    random_action = np.random.choice(child_actions)
                    random_child = leaf_node.children[random_action]
                    logger.debug(
                        f"      Rollout starts from random child (Action: {random_action})"
                    )

                    leaf_depth = len(search_path) - 1
                    max_rollout_depth = max(0, self.dmax - leaf_depth)
                    rollout_rewards_list = self.do_rollout(
                        random_child, max_rollout_depth
                    )

                    initial_backprop_value = self._calculate_discounted_return(
                        rollout_rewards_list, self.discount_factor
                    )
                    logger.debug(
                        f"      Rollout discounted value: {initial_backprop_value:.4f}"
                    )

                    backprop_path = search_path + [random_child]
                else:
                    initial_backprop_value = leaf_value_estimate_fallback

                # Backpropagation
                logger.debug("    Phase 4: Backpropagation")
                self.backpropagate(backprop_path, initial_backprop_value)
                logger.debug("    Backpropagation complete.")

            except Exception as e_expand_rollout:
                logger.exception(
                    f"    Error during MCTS expansion/rollout phase (Sim {sim+1}): {e_expand_rollout}. Stopping simulation."
                )
                continue

        logger.debug(f"MCTS search finished. Duration: {time.time() - start_time:.4f}s")

    def backpropagate(self, search_path: List[Node], value_estimate: float):
        """Backpropagate value up the path, updating visit counts and Q-values."""
        current_discounted_return: float = value_estimate
        logger.debug(
            f"    Backpropagating value: {value_estimate:.4f} starting from node {len(search_path)-1}"
        )

        for node in reversed(search_path):
            node.update_q_value(current_discounted_return)
            node.visit_count += 1

            action_taken = node.action if node.action is not None else "Root"
            logger.debug(
                f"      Updating Node (Reached via Action={action_taken}): N={node.visit_count}, New Q={node.q_value:.4f}"
            )

            current_discounted_return = (
                node.reward + self.discount_factor * current_discounted_return
            )

    def get_policy_and_value(
        self, temperature: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Return the action policy from root visit counts and the root value estimate (root Q).
        """
        if self.root is None:
            logger.error(
                "Cannot get policy/value: MCTS root node is None (search likely failed)."
            )
            num_actions = len(self.legal_actions)
            uniform_policy = np.ones(max(1, num_actions), dtype=np.float32) / max(
                1, num_actions
            )
            return uniform_policy, 0.0

        if not self.root.children:
            logger.warning(
                "MCTS root node has no children (expansion failed?). Returning uniform policy/zero value."
            )
            num_actions = len(self.legal_actions)
            uniform_policy = np.ones(max(1, num_actions), dtype=np.float32) / max(
                1, num_actions
            )
            root_value = self.root.q_value if hasattr(self.root, "q_value") else 0.0
            return uniform_policy, float(root_value)

        action_space_size = len(self.legal_actions)
        visit_counts = np.zeros(action_space_size, dtype=np.float32)

        for action_idx, action in enumerate(self.legal_actions):
            if action in self.root.children:
                visit_counts[action_idx] = self.root.children[action].visit_count

        logger.debug(f"Root children visit counts: {visit_counts}")

        if temperature == 0:
            policy = np.zeros_like(visit_counts)
            if np.sum(visit_counts) > 0:
                policy[int(np.argmax(visit_counts))] = 1.0
            else:
                logger.warning(
                    "All root visit counts are zero! Returning uniform policy."
                )
                policy = np.ones_like(visit_counts) / max(1, action_space_size)
            logger.debug(f"Greedy policy (temp=0): {policy}")
        else:
            if temperature <= 0:
                logger.warning(
                    f"Temperature ({temperature}) must be positive. Using temp=1.0."
                )
                temperature = 1.0

            temp_scaled_counts = np.power(visit_counts, 1.0 / temperature)
            sum_counts = np.sum(temp_scaled_counts)

            if sum_counts > 1e-8:
                policy = temp_scaled_counts / sum_counts
            else:
                logger.warning(
                    "Sum of temperature-scaled counts is near zero. Using uniform policy."
                )
                policy = np.ones_like(visit_counts) / max(1, action_space_size)

            logger.debug(f"Stochastic policy (temp={temperature}): {policy}")

        if np.isnan(policy).any():
            logger.error("NaN detected in final MCTS policy. Falling back to uniform.")
            policy = np.ones_like(visit_counts) / max(1, action_space_size)

        if abs(np.sum(policy) - 1.0) > 1e-5:
            logger.warning(f"Final MCTS policy sum is {np.sum(policy)}. Renormalizing.")
            policy = policy / np.maximum(np.sum(policy), 1e-8)

        root_q_value = self.root.q_value
        logger.debug(f"Root value estimate (Root Q-value): {root_q_value:.4f}")

        return policy, float(root_q_value)
