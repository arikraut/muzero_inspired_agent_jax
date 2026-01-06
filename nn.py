"""
nn.py
MuZero neural networks (JAX/Flax):
- Representation (h): observation -> latent state
- Dynamics (g): (latent state, action) -> (next latent state, reward)
- Prediction (f): latent state -> (policy logits, value)
Also includes NeuralNetworkManager for init/train/checkpointing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax import serialization
from functools import partial
from typing import (
    Sequence,
    Dict,
    Any,
    Tuple,
    List,
    Optional,
    Callable,
    Union,
)
import os
import traceback
import json
import yaml
import logging
import warnings
import copy

logger = logging.getLogger(__name__)


def compare_configs(saved_cfg_val: Any, current_cfg_val: Any) -> bool:
    """
    Recursively compares two configuration values (nested dicts/lists supported).
    Used during checkpoint loading to detect potential incompatibilities.
    """
    if type(saved_cfg_val) != type(current_cfg_val):
        if isinstance(saved_cfg_val, (list, tuple)) and isinstance(
            current_cfg_val, (list, tuple)
        ):
            return list(saved_cfg_val) == list(current_cfg_val)
        logger.debug(
            f"Config mismatch: Type difference - Saved: {type(saved_cfg_val)}, Current: {type(current_cfg_val)}"
        )
        return False

    if isinstance(saved_cfg_val, dict):
        if saved_cfg_val.keys() != current_cfg_val.keys():
            logger.debug(
                f"Config mismatch: Dictionary keys differ - Saved: {saved_cfg_val.keys()}, Current: {current_cfg_val.keys()}"
            )
            return False
        return all(
            compare_configs(saved_cfg_val[k], current_cfg_val[k]) for k in saved_cfg_val
        )
    elif isinstance(saved_cfg_val, (list, tuple)):
        return list(saved_cfg_val) == list(current_cfg_val)
    else:
        return saved_cfg_val == current_cfg_val


ACTIVATION_MAP: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "softmax": nn.softmax,
    "identity": lambda x: x,
}


class NeuralNetwork(nn.Module):
    """
    Generic MLP implemented in Flax.
    Used for MuZero dynamics/prediction and (optionally) representation for vector inputs.
    """

    layer_sizes: Sequence[int]
    activation_names: Sequence[str]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies MLP layers to input. Flattens inputs with ndim > 2. Adds batch dim if 1D.
        """
        if x.ndim > 2:
            logger.debug(f"MLP Input: Flattening {x.shape} -> ({x.shape[0]}, -1)")
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            logger.debug(
                f"MLP Input: Reshaping 1D input {x.shape} -> (1, {x.shape[0]})"
            )
            x = x.reshape(1, -1)

        if not self.layer_sizes:
            logger.warning(
                "MLP defined with no layers (layer_sizes is empty). Returning input directly."
            )
            return x
        if len(self.activation_names) != len(self.layer_sizes):
            raise ValueError(
                f"MLP Config Error: Number of activation functions ({len(self.activation_names)}) "
                f"must match the number of layer sizes ({len(self.layer_sizes)})."
            )

        for i, size in enumerate(self.layer_sizes):
            x = nn.Dense(features=size, name=f"dense_{i}")(x)
            activation_func = ACTIVATION_MAP.get(self.activation_names[i])
            if activation_func is None:
                raise ValueError(
                    f"Unknown activation function name '{self.activation_names[i]}' found in config."
                )
            if self.activation_names[i] != "identity":
                x = activation_func(x)

        return x


class CNNRepresentationNetwork(nn.Module):
    """
    CNN-based representation network for image-like observations.
    Conv stack -> flatten -> dense -> latent state.
    """

    latent_dim: int
    cnn_filters: Sequence[int]
    cnn_kernel_sizes: Sequence[int]
    cnn_strides: Sequence[int]
    cnn_activation: str
    dense_layers: Sequence[int]
    dense_activation: str
    output_activation: str

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies CNN representation to input and returns latent state (Batch, latent_dim)."""
        x = x.astype(jnp.float32)
        logger.debug(f"CNNRepresentation Input Shape: {x.shape}")

        if not (
            len(self.cnn_filters) == len(self.cnn_kernel_sizes) == len(self.cnn_strides)
        ):
            raise ValueError(
                "CNN Config Error: Lengths of cnn_filters, cnn_kernel_sizes, and cnn_strides must match."
            )

        try:
            cnn_act_fn = ACTIVATION_MAP[self.cnn_activation]
            dense_act_fn = ACTIVATION_MAP[self.dense_activation]
            output_act_fn = ACTIVATION_MAP[self.output_activation]
        except KeyError as e:
            raise ValueError(f"Unknown activation function name '{e}' in CNN config.")

        for i, (filters, kernel, stride) in enumerate(
            zip(self.cnn_filters, self.cnn_kernel_sizes, self.cnn_strides)
        ):
            x = nn.Conv(
                features=filters,
                kernel_size=(kernel, kernel),
                strides=(stride, stride),
                padding="VALID",
                name=f"conv_{i}",
            )(x)
            x = cnn_act_fn(x)
            logger.debug(
                f"  CNN Layer {i}: Conv({filters}, K={kernel}, S={stride}), Act: {self.cnn_activation}, Output Shape: {x.shape}"
            )

        if x.ndim < 2:
            raise ValueError(
                f"Unexpected shape after convolutional layers (expected at least 2D): {x.shape}."
            )

        original_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        logger.debug(f"  Flattened CNN output from {original_shape} to {x.shape}")

        for i, size in enumerate(self.dense_layers):
            x = nn.Dense(features=size, name=f"dense_{i}")(x)
            x = dense_act_fn(x)
            logger.debug(
                f"  Dense Layer {i}: Dense({size}), Act: {self.dense_activation}, Output Shape: {x.shape}"
            )

        x = nn.Dense(features=self.latent_dim, name="output_latent")(x)
        x = output_act_fn(x)
        logger.debug(
            f"  Output Layer: Dense({self.latent_dim}), Act: {self.output_activation}, Final Latent Shape: {x.shape}"
        )

        return x


class NeuralNetworkManager:
    """
    Manages MuZero networks (h/g/f), optimizer state, training, and checkpointing.
    """

    def __init__(self, merged_config: Dict[str, Any]):
        """Initializes modules, parameters, and optimizer from finalized config."""
        logger.info("Initializing NeuralNetworkManager...")
        self.merged_config = merged_config

        try:
            global_vars = self.merged_config["global_network_vars"]
            nn_config = self.merged_config["neural_network"]
            rep_config_base = nn_config["representation_network"]
            dyn_config_nn = nn_config["dynamics_network"]
            pred_config_nn = nn_config["prediction_network"]
            train_cfg = self.merged_config["training"]

            self.nnr_input_shape: Tuple = tuple(global_vars["nnr_input_shape"])
            self.latent_dim: int = global_vars["latent_dim"]
            self.action_dim: int = global_vars["action_dim"]
            self.representation_network_type: str = nn_config[
                "representation_network_type"
            ]
            self.unroll_steps: int = nn_config["unroll_steps"]

            self.policy_loss_weight: float = float(
                train_cfg.get("policy_loss_weight", 1.0)
            )
            self.value_loss_weight: float = float(
                train_cfg.get("value_loss_weight", 1.0)
            )
            self.reward_loss_weight: float = float(
                train_cfg.get("reward_loss_weight", 1.0)
            )

            self.one_hot_actions: bool = dyn_config_nn.get("one_hot_actions", False)
            self.dyn_action_dim: int = self._get_dyn_action_dim(dyn_config_nn)

            global_seed = global_vars.get("global_seed", np.random.randint(0, 2**31))
            self.key = jax.random.PRNGKey(global_seed)
            logger.info(f"Using global seed: {global_seed} for JAX PRNG key.")

            lr = float(global_vars.get("learning_rate", 1e-3))
            max_grad_norm = 1.0
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr),
            )
            logger.info(
                f"Optimizer: Adam with learning rate {lr} AND Gradient Clipping (Global Norm={max_grad_norm})"
            )

        except KeyError as e:
            logger.exception(f"NNM Init Error: Missing required configuration key: {e}")
            raise
        except Exception as e:
            logger.exception(
                f"NNM Init Error: Unexpected error reading configuration: {e}"
            )
            raise

        self.key, nnr_key, nnd_key, nnp_key = jax.random.split(self.key, 4)

        logger.info(
            f"Creating Representation Network (NNr 'h', type: {self.representation_network_type})..."
        )
        dummy_nnr_input = jnp.zeros((1,) + self.nnr_input_shape, dtype=jnp.float32)
        logger.debug(f"  Dummy NNr input shape for init: {dummy_nnr_input.shape}")

        if self.representation_network_type == "cnn":
            try:
                required_cnn_keys = [
                    "cnn_filters",
                    "cnn_kernel_sizes",
                    "cnn_strides",
                    "cnn_activation",
                    "dense_layers",
                    "dense_activation",
                    "output_activation",
                ]
                missing_keys = [
                    k for k in required_cnn_keys if k not in rep_config_base
                ]
                if missing_keys:
                    raise KeyError(
                        f"Missing required CNN keys in representation_network config: {missing_keys}"
                    )

                self.nnr_module = CNNRepresentationNetwork(
                    latent_dim=self.latent_dim,
                    cnn_filters=rep_config_base["cnn_filters"],
                    cnn_kernel_sizes=rep_config_base["cnn_kernel_sizes"],
                    cnn_strides=rep_config_base["cnn_strides"],
                    cnn_activation=rep_config_base["cnn_activation"],
                    dense_layers=rep_config_base["dense_layers"],
                    dense_activation=rep_config_base["dense_activation"],
                    output_activation=rep_config_base["output_activation"],
                )
                logger.info("  CNNRepresentationNetwork module created.")
            except Exception as e:
                logger.exception(f"Error creating CNN Representation Network: {e}")
                raise

        elif self.representation_network_type == "mlp":
            try:
                required_mlp_keys = ["hidden_layers", "activation_functions"]
                missing_keys = [
                    k for k in required_mlp_keys if k not in rep_config_base
                ]
                if missing_keys:
                    raise KeyError(
                        f"Missing required MLP keys in representation_network config: {missing_keys}"
                    )

                mlp_layers = list(rep_config_base["hidden_layers"]) + [self.latent_dim]
                mlp_activations = list(rep_config_base["activation_functions"])
                if len(mlp_layers) != len(mlp_activations):
                    raise ValueError(
                        f"MLP NNr Config Error: Number of layers ({len(mlp_layers)}) "
                        f"must match number of activations ({len(mlp_activations)})."
                    )

                self.nnr_module = NeuralNetwork(
                    layer_sizes=mlp_layers, activation_names=mlp_activations
                )
                logger.info("  NeuralNetwork (MLP) module created for NNr.")

            except Exception as e:
                logger.exception(f"Error creating MLP Representation Network: {e}")
                raise
        else:
            raise ValueError(
                f"Unsupported representation_network_type: '{self.representation_network_type}'. Must be 'cnn' or 'mlp'."
            )

        logger.info("Creating Dynamics Network (NNd 'g')...")
        dummy_latent = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
        dummy_dyn_action = jnp.zeros((1, self.dyn_action_dim), dtype=jnp.float32)
        dummy_dyn_input = jnp.concatenate([dummy_latent, dummy_dyn_action], axis=-1)
        nnd_output_dim = self.latent_dim + 1
        logger.debug(f"  Dummy NNd input shape for init: {dummy_dyn_input.shape}")
        logger.debug(f"  NNd output dimension: {nnd_output_dim}")
        self.nnd_module = self._create_module(dyn_config_nn, nnd_output_dim, "NNd")

        logger.info("Creating Prediction Network (NNp 'f')...")
        nnp_output_dim = self.action_dim + 1
        logger.debug(f"  Dummy NNp input shape for init: {dummy_latent.shape}")
        logger.debug(f"  NNp output dimension: {nnp_output_dim}")
        self.nnp_module = self._create_module(pred_config_nn, nnp_output_dim, "NNp")

        try:
            logger.info("Initializing network parameters using dummy inputs...")
            nnr_params = self.nnr_module.init(nnr_key, dummy_nnr_input)["params"]
            nnd_params = self.nnd_module.init(nnd_key, dummy_dyn_input)["params"]
            nnp_params = self.nnp_module.init(nnp_key, dummy_latent)["params"]

            self.params = {"nnr": nnr_params, "nnd": nnd_params, "nnp": nnp_params}
            self.opt_state = self.optimizer.init(self.params)
            logger.info("Optimizer state initialized.")

        except Exception as e:
            logger.exception(
                f"Error during parameter initialization or optimizer state creation: {e}"
            )
            logger.error(
                f"Shapes used for init: NNr={dummy_nnr_input.shape}, NNd={dummy_dyn_input.shape}, NNp={dummy_latent.shape}"
            )
            raise

        self.log_history = []
        logger.info("NeuralNetworkManager Initialization Complete.")

    def _get_dyn_action_dim(self, dyn_config: Dict[str, Any]) -> int:
        """Determine action input dimension for dynamics: one-hot (action_dim) or scalar (1)."""
        return self.action_dim if dyn_config.get("one_hot_actions", False) else 1

    def _create_module(
        self, network_config: Dict[str, Any], output_dim: int, name: str
    ) -> nn.Module:
        """
        Build an MLP module (NeuralNetwork) from config.
        Used for dynamics (g) and prediction (f).
        """
        logger.debug(f"Creating MLP module for {name}...")
        try:
            required_keys = ["hidden_layers", "activation_functions"]
            missing_keys = [k for k in required_keys if k not in network_config]
            if missing_keys:
                raise KeyError(
                    f"Missing required MLP keys for {name} config: {missing_keys}"
                )

            hidden_layers = network_config["hidden_layers"]
            activation_names = network_config["activation_functions"]
            layer_sizes = list(map(int, hidden_layers)) + [int(output_dim)]
            activation_names = list(map(str, activation_names))

            if len(activation_names) != len(layer_sizes):
                raise ValueError(
                    f"Config error for {name}: Number of activation functions ({len(activation_names)}) "
                    f"must match number of layers (hidden + output = {len(layer_sizes)})."
                )

            logger.info(
                f"  Creating {name} MLP with structure: Layers={layer_sizes}, Activations={activation_names}"
            )
            return NeuralNetwork(
                layer_sizes=layer_sizes, activation_names=activation_names
            )
        except Exception as e:
            logger.exception(f"Error creating MLP module '{name}': {e}")
            raise

    # JIT forward passes (static self) for representation/dynamics/prediction.

    @partial(jax.jit, static_argnums=(0,))
    def representation_forward(
        self, params: Dict, nnr_input: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Representation (h): preprocessed observation input -> latent state.
        Supports single-instance and batched input.
        """
        input_ndim_config = len(self.nnr_input_shape)
        was_unbatched = False
        if nnr_input.ndim == input_ndim_config:
            nnr_input = jnp.expand_dims(nnr_input, axis=0)
            was_unbatched = True
        elif nnr_input.ndim != input_ndim_config + 1:
            raise ValueError(
                f"Unexpected nnr_input dimensions: {nnr_input.ndim}. "
                f"Expected {input_ndim_config} (single) or {input_ndim_config + 1} (batched)."
            )

        latent_state = self.nnr_module.apply({"params": params["nnr"]}, nnr_input)

        if was_unbatched:
            latent_state = jnp.squeeze(latent_state, axis=0)

        return latent_state

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_forward(
        self,
        params: Dict,
        abstract_state: jnp.ndarray,
        action: Union[int, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Dynamics (g): (latent state, action) -> (next latent state, reward).
        Supports single-instance and batched latent state.
        """
        was_unbatched = abstract_state.ndim == 1
        if was_unbatched:
            abstract_state = jnp.expand_dims(abstract_state, axis=0)

        batch_size = abstract_state.shape[0]

        if self.one_hot_actions:
            action_array = jnp.asarray(action)
            batched_action_indices = (
                jnp.repeat(action_array, batch_size)
                if action_array.ndim == 0
                else action_array
            )
            if batched_action_indices.shape != (batch_size,):
                raise ValueError(
                    f"Action shape mismatch for one-hot encoding. Expected ({batch_size},), got {batched_action_indices.shape}"
                )

            dyn_action_input = jax.nn.one_hot(
                batched_action_indices,
                self.action_dim,
                dtype=abstract_state.dtype,
            )
        else:
            action_array = jnp.asarray(action, dtype=abstract_state.dtype)
            dyn_action_input = (
                jnp.repeat(action_array, batch_size).reshape(-1, 1)
                if action_array.ndim == 0
                else action_array.reshape(-1, 1)
            )
            if dyn_action_input.shape != (batch_size, 1):
                raise ValueError(
                    f"Action shape mismatch for scalar encoding. Expected ({batch_size}, 1), got {dyn_action_input.shape}"
                )

        dyn_input = jnp.concatenate([abstract_state, dyn_action_input], axis=-1)
        out = self.nnd_module.apply({"params": params["nnd"]}, dyn_input)

        next_latent = out[..., :-1]
        predicted_reward = out[..., -1]

        if was_unbatched:
            next_latent = jnp.squeeze(next_latent, axis=0)
            predicted_reward = jnp.squeeze(predicted_reward, axis=0)

        return next_latent, predicted_reward

    @partial(jax.jit, static_argnums=(0,))
    def prediction_forward(
        self, params: Dict, abstract_state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Prediction (f): latent state -> (policy logits, value).
        Supports single-instance and batched latent state.
        """
        was_unbatched = abstract_state.ndim == 1
        if was_unbatched:
            abstract_state = jnp.expand_dims(abstract_state, axis=0)

        out = self.nnp_module.apply({"params": params["nnp"]}, abstract_state)
        policy_logits = out[..., : self.action_dim]
        predicted_value = out[..., -1]

        if was_unbatched:
            policy_logits = jnp.squeeze(policy_logits, axis=0)
            predicted_value = jnp.squeeze(predicted_value, axis=0)

        return policy_logits, predicted_value

    # Loss / training

    def _calculate_losses(
        self,
        predicted_policy_logits: jnp.ndarray,
        target_policy: jnp.ndarray,
        predicted_value: jnp.ndarray,
        target_value: jnp.ndarray,
        predicted_reward: Optional[jnp.ndarray],
        target_reward: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (policy_loss, value_loss, reward_loss) for one unroll step."""
        log_pred_probs = jax.nn.log_softmax(predicted_policy_logits, axis=-1)
        policy_loss = -jnp.sum(target_policy * log_pred_probs, axis=-1)

        value_loss = 0.5 * jnp.square(predicted_value - target_value)

        if predicted_reward is not None and target_reward is not None:
            reward_loss = 0.5 * jnp.square(predicted_reward - target_reward)
        else:
            reward_loss = jnp.array(0.0)

        return jnp.mean(policy_loss), jnp.mean(value_loss), jnp.mean(reward_loss)

    def single_sequence_loss(
        self,
        params: Dict,
        initial_nnr_input: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        policies: jnp.ndarray,
        values: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes MuZero loss for a single unroll sequence.
        Returns (total_weighted_loss, avg_policy_loss, avg_value_loss, avg_reward_loss).
        """
        current_latent = self.representation_forward(params, initial_nnr_input)
        predicted_policy_logits_k0, predicted_value_k0 = self.prediction_forward(
            params, current_latent
        )

        policy_loss_k0, value_loss_k0, _ = self._calculate_losses(
            predicted_policy_logits=predicted_policy_logits_k0,
            target_policy=policies[0],
            predicted_value=predicted_value_k0,
            target_value=values[0],
            predicted_reward=None,
            target_reward=None,
        )

        total_policy_loss = policy_loss_k0
        total_value_loss = value_loss_k0
        total_reward_loss = jnp.array(0.0)

        for k in range(1, self.unroll_steps + 1):
            action_k = actions[k - 1]
            target_reward_k = rewards[k - 1]

            next_latent, predicted_reward_k = self.dynamics_forward(
                params, current_latent, action_k
            )

            if k < self.unroll_steps:
                predicted_policy_logits_k, predicted_value_k = self.prediction_forward(
                    params, next_latent
                )

                policy_loss_k, value_loss_k, reward_loss_k = self._calculate_losses(
                    predicted_policy_logits=predicted_policy_logits_k,
                    target_policy=policies[k],
                    predicted_value=predicted_value_k,
                    target_value=values[k],
                    predicted_reward=predicted_reward_k,
                    target_reward=target_reward_k,
                )
            else:
                _, _, reward_loss_k = self._calculate_losses(
                    predicted_policy_logits=jnp.zeros(self.action_dim),
                    target_policy=jnp.zeros(self.action_dim),
                    predicted_value=jnp.array(0.0),
                    target_value=jnp.array(0.0),
                    predicted_reward=predicted_reward_k,
                    target_reward=target_reward_k,
                )
                policy_loss_k = value_loss_k = jnp.array(0.0)

            total_policy_loss += policy_loss_k
            total_value_loss += value_loss_k
            total_reward_loss += reward_loss_k
            current_latent = next_latent

        num_policy_value_steps = float(self.unroll_steps)
        num_reward_steps = float(self.unroll_steps)
        norm_factor_pv = jnp.maximum(num_policy_value_steps, 1.0)
        norm_factor_r = jnp.maximum(num_reward_steps, 1.0)

        avg_policy_loss = total_policy_loss / norm_factor_pv
        avg_value_loss = total_value_loss / norm_factor_pv
        avg_reward_loss = total_reward_loss / norm_factor_r

        total_weighted_loss = (
            self.policy_loss_weight * avg_policy_loss
            + self.value_loss_weight * avg_value_loss
            + self.reward_loss_weight * avg_reward_loss
        )

        return total_weighted_loss, avg_policy_loss, avg_value_loss, avg_reward_loss

    @partial(jax.jit, static_argnums=(0,))
    def batch_loss_fn(
        self, params: Dict, batch: Dict
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Vectorizes single_sequence_loss over the batch dimension via vmap."""
        total_loss_batch, pol_loss_batch, val_loss_batch, rwd_loss_batch = jax.vmap(
            self.single_sequence_loss,
            in_axes=(None, 0, 0, 0, 0, 0),
            out_axes=0,
        )(
            params,
            batch["nnr_input"],
            batch["actions"],
            batch["rewards"],
            batch["policies"],
            batch["values"],
        )

        return (
            jnp.mean(total_loss_batch),
            jnp.mean(pol_loss_batch),
            jnp.mean(val_loss_batch),
            jnp.mean(rwd_loss_batch),
        )

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self, params: Dict, opt_state: optax.OptState, batch: Dict
    ) -> Tuple[Dict, optax.OptState, Dict[str, jnp.ndarray]]:
        """Runs one optimizer update step and returns updated state + loss dict."""

        def compute_loss_and_aux(p):
            total_loss, pol_loss, val_loss, rew_loss = self.batch_loss_fn(p, batch)
            return total_loss, (pol_loss, val_loss, rew_loss)

        (total_loss_val, aux_losses), grads = jax.value_and_grad(
            compute_loss_and_aux, has_aux=True
        )(params)

        pol_loss_val, val_loss_val, rew_loss_val = aux_losses
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        loss_dict = {
            "total_loss": total_loss_val,
            "policy_loss": pol_loss_val,
            "value_loss": val_loss_val,
            "reward_loss": rew_loss_val,
        }

        return new_params, new_opt_state, loss_dict

    def train_networks(
        self, episode_buffer, batch_size: int
    ) -> Optional[Dict[str, float]]:
        """
        Samples a batch from replay buffer and applies one training update.
        Returns scalar losses as floats on success.
        """
        logger.debug(f"Attempting to sample batch of size {batch_size}...")
        try:
            batch = episode_buffer.sample(batch_size)
        except Exception as e:
            logger.exception(f"Error sampling batch from episode buffer: {e}")
            return None

        if batch is None:
            logger.warning(
                "Sampling failed: Episode buffer returned None (likely empty or too few valid sequences). Skipping training step."
            )
            return None

        required_keys = {"nnr_input", "actions", "rewards", "policies", "values"}
        if not required_keys.issubset(batch.keys()):
            logger.error(
                f"Sampled batch is missing required keys: {required_keys - batch.keys()}. Skipping training step."
            )
            return None

        logger.debug(
            f"Batch sampled successfully. Shapes: NNR={batch['nnr_input'].shape}, A={batch['actions'].shape}, "
            f"R={batch['rewards'].shape}, P={batch['policies'].shape}, V={batch['values'].shape}"
        )

        try:
            if not isinstance(self.params, dict) or self.opt_state is None:
                logger.error(
                    "NNM state (params or opt_state) is invalid. Cannot train."
                )
                return None

            new_params, new_opt_state, loss_dict_jax = self.train_step(
                self.params, self.opt_state, batch
            )

            self.params = new_params
            self.opt_state = new_opt_state

            float_loss_dict = {k: float(v) for k, v in loss_dict_jax.items()}
            self.log_history.append(float_loss_dict)
            logger.debug(f"Training step successful. Losses: {float_loss_dict}")
            return float_loss_dict

        except Exception as e:
            logger.exception(f"Error during train_step execution: {e}")
            return None

    # Checkpointing

    def save_state(self, directory: str, step: int):
        """Saves params/optimizer state, training history, and config to disk."""
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(
                f"Saving checkpoint for training step {step} to directory: '{directory}'"
            )

            flax_state = {"params": self.params, "opt_state": self.opt_state}
            bytes_output = serialization.to_bytes(flax_state)
            flax_filepath = os.path.join(directory, f"muzero_state_{step}.flax")
            with open(flax_filepath, "wb") as f_flax:
                f_flax.write(bytes_output)
            logger.info(f"  Saved Flax state (params & optimizer) to: {flax_filepath}")

            history_filepath = os.path.join(directory, f"muzero_history_{step}.json")
            try:
                with open(history_filepath, "w") as f_hist:
                    json.dump(self.log_history, f_hist, indent=2)
                logger.info(
                    f"  Saved training log history ({len(self.log_history)} entries) to: {history_filepath}"
                )
            except Exception as e_hist:
                logger.warning(f"  Could not save training history: {e_hist}")

            config_save_path = os.path.join(directory, f"muzero_config_{step}.yaml")
            try:
                config_to_save = copy.deepcopy(self.merged_config)
                with open(config_save_path, "w") as f_cfg:
                    yaml.dump(
                        config_to_save,
                        f_cfg,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                logger.info(f"  Saved configuration to: {config_save_path}")
            except Exception as e_cfg:
                logger.warning(f"  Could not save configuration: {e_cfg}")

        except Exception as e:
            logger.exception(
                f"Error saving state for step {step} to '{directory}': {e}"
            )

    def load_state(self, directory: str, step: int) -> bool:
        """
        Loads params/optimizer state (and optionally history/config) from disk.
        Warns on likely config incompatibilities.
        """
        logger.info(
            f"\nAttempting to load checkpoint for step {step} from directory: '{directory}'..."
        )
        flax_filepath = os.path.join(directory, f"muzero_state_{step}.flax")
        config_load_path = os.path.join(directory, f"muzero_config_{step}.yaml")
        history_filepath = os.path.join(directory, f"muzero_history_{step}.json")

        if not os.path.exists(flax_filepath):
            logger.error(
                f"Checkpoint file not found: {flax_filepath}. Cannot load state."
            )
            return False

        saved_config = None
        config_compatible = True
        if os.path.exists(config_load_path):
            logger.info(f"Found saved configuration file: {config_load_path}")
            try:
                with open(config_load_path, "r") as f_cfg:
                    saved_config = yaml.safe_load(f_cfg)

                keys_to_compare = [
                    ("global_network_vars", "latent_dim"),
                    ("global_network_vars", "nnr_input_shape"),
                    ("global_network_vars", "action_dim"),
                    ("neural_network", "representation_network_type"),
                    ("neural_network", "representation_network"),
                    ("neural_network", "dynamics_network", "one_hot_actions"),
                    ("neural_network", "dynamics_network", "hidden_layers"),
                    ("neural_network", "prediction_network", "hidden_layers"),
                ]
                mismatched_keys_details = []
                current_config = self.merged_config

                for key_path in keys_to_compare:
                    saved_val, current_val = saved_config, current_config
                    valid_path = True
                    for key in key_path:
                        saved_val = (
                            saved_val.get(key, None)
                            if isinstance(saved_val, dict)
                            else None
                        )
                        current_val = (
                            current_val.get(key, None)
                            if isinstance(current_val, dict)
                            else None
                        )
                        if saved_val is None or current_val is None:
                            if saved_val is not None or current_val is not None:
                                mismatched_keys_details.append(
                                    f"Path '{'->'.join(key_path)}' missing in saved({saved_val is None}) or current({current_val is None})"
                                )
                            valid_path = False
                            break

                    if valid_path and not compare_configs(saved_val, current_val):
                        mismatched_keys_details.append(
                            f"Value mismatch at '{'->'.join(key_path)}': Saved='{saved_val}', Current='{current_val}'"
                        )

                if mismatched_keys_details:
                    config_compatible = False
                    warning_message = (
                        f"POTENTIAL CONFIG INCOMPATIBILITY DETECTED loading checkpoint step {step}.\n"
                        f"Differences found in critical keys between saved and current config:\n"
                        + "\n".join(
                            [f"  - {detail}" for detail in mismatched_keys_details]
                        )
                        + "\nLoading state into potentially incompatible network architecture. "
                        "This may cause errors or unexpected behavior."
                    )
                    warnings.warn(warning_message, UserWarning)
                    logger.warning(warning_message)
                else:
                    logger.info(
                        "Saved configuration appears compatible with current configuration."
                    )

            except Exception as e_cfg:
                logger.warning(
                    f"Error loading or comparing saved configuration file '{config_load_path}': {e_cfg}. Proceeding with state loading attempt."
                )
        else:
            warnings.warn(
                f"Saved configuration file not found ({config_load_path}). "
                f"Cannot verify compatibility. Assuming checkpoint is compatible.",
                UserWarning,
            )
            logger.warning(
                f"Saved configuration file not found ({config_load_path}). Assuming compatibility, but proceed with caution."
            )

        try:
            logger.info(f"Loading Flax state from {flax_filepath}...")
            with open(flax_filepath, "rb") as f:
                bytes_input = f.read()

            logger.info(
                "Creating structural template (target) for Flax deserialization..."
            )
            dummy_key = jax.random.PRNGKey(0)
            _, nnr_key, nnd_key, nnp_key = jax.random.split(dummy_key, 4)

            dummy_nnr_input = jnp.zeros((1,) + self.nnr_input_shape, dtype=jnp.float32)
            dummy_latent = jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
            dummy_dyn_action = jnp.zeros((1, self.dyn_action_dim), dtype=jnp.float32)
            dummy_dyn_input = jnp.concatenate([dummy_latent, dummy_dyn_action], axis=-1)

            try:
                if not all(
                    hasattr(self, m)
                    for m in ["nnr_module", "nnd_module", "nnp_module", "optimizer"]
                ):
                    raise RuntimeError(
                        "Internal error: Modules/optimizer not initialized before attempting load."
                    )

                dummy_nnr_params = self.nnr_module.init(nnr_key, dummy_nnr_input)[
                    "params"
                ]
                dummy_nnd_params = self.nnd_module.init(nnd_key, dummy_dyn_input)[
                    "params"
                ]
                dummy_nnp_params = self.nnp_module.init(nnp_key, dummy_latent)["params"]
                dummy_params = {
                    "nnr": dummy_nnr_params,
                    "nnd": dummy_nnd_params,
                    "nnp": dummy_nnp_params,
                }

                dummy_opt_state = self.optimizer.init(dummy_params)
                target_template = {"params": dummy_params, "opt_state": dummy_opt_state}
                logger.info("Structural template created successfully.")

            except Exception as e_template:
                logger.exception(
                    f"Error creating target template for deserialization: {e_template}"
                )
                return False

            logger.info("Attempting deserialization using target template...")
            loaded_flax_state = serialization.from_bytes(target_template, bytes_input)
            logger.info("Flax state deserialization successful.")

            self.params = loaded_flax_state["params"]
            self.opt_state = loaded_flax_state["opt_state"]
            logger.info(
                f"Successfully loaded network parameters and optimizer state for step {step}."
            )

            try:
                if os.path.exists(history_filepath):
                    with open(history_filepath, "r") as f_hist:
                        self.log_history = json.load(f_hist)
                    logger.info(
                        f"Loaded training history ({len(self.log_history)} entries) from: {history_filepath}"
                    )
                else:
                    logger.warning(
                        f"Training history file not found ({history_filepath}), resetting log history."
                    )
                    self.log_history = []
            except Exception as e_hist:
                logger.warning(
                    f"Error loading training history from {history_filepath}: {e_hist}. Resetting log history."
                )
                self.log_history = []

            logger.info(f"--- Checkpoint loading for step {step} successful. ---")
            if not config_compatible:
                logger.warning(
                    "--- Reminder: Loaded state with potential configuration incompatibilities noted above. ---"
                )
            return True

        except Exception as e:
            logger.exception(
                f"Fatal error loading Flax state for step {step} from {flax_filepath}: {e}"
            )
            return False
