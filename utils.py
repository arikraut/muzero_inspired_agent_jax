# utils.py
"""
Utilities supporting the MuZero implementation:

- Config loading + finalization/validation (including env-derived values)
- Observation preprocessing (one-hot / image transforms / normalization)
- History stacking to produce representation-network inputs
"""

import os
import yaml
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
import cv2
import copy
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    """
    if not os.path.exists(config_path):
        err_msg = f"Configuration file not found at: {config_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        err_msg = f"Error parsing YAML file {config_path}: {e}"
        logger.error(err_msg)
        raise ValueError(err_msg) from e
    except Exception as e:
        logger.exception(f"Unexpected error loading config {config_path}: {e}")
        raise


def _validate_config_sections(config: Dict[str, Any]):
    """Checks required top-level sections."""
    required_sections = [
        "global_network_vars",
        "neural_network",
        "game_settings",
        "rlm",
        "umcts",
        "episode_buffer",
        "training",
    ]
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(
            f"Config missing required sections: {', '.join(missing_sections)}."
        )
    logger.debug("All required config sections found.")


def _validate_network_config(config: Dict[str, Any]):
    """Validates presence and basic types of critical network-related keys."""
    if "global_network_vars" not in config:
        raise ValueError("Config missing section 'global_network_vars'.")
    global_vars_req = ["nnr_input_shape", "latent_dim"]
    missing_global = [
        k for k in global_vars_req if k not in config["global_network_vars"]
    ]
    if missing_global:
        raise ValueError(
            f"Config missing keys in 'global_network_vars': {', '.join(missing_global)}."
        )

    nnr_shape = config["global_network_vars"]["nnr_input_shape"]
    if not isinstance(nnr_shape, (list, tuple)):
        raise ValueError(
            f"'global_network_vars.nnr_input_shape' must be a list or tuple (e.g., [H, W, C]), got {type(nnr_shape)}"
        )

    if "neural_network" not in config:
        raise ValueError("Config missing section 'neural_network'.")
    nn_req = ["representation_network_type", "representation_input_states"]
    missing_nn = [k for k in nn_req if k not in config["neural_network"]]
    if missing_nn:
        raise ValueError(
            f"Config missing keys in 'neural_network': {', '.join(missing_nn)}."
        )

    logger.debug("Basic network config keys validated.")


def _validate_preprocessing_config(config: Dict[str, Any], raw_obs_space: gym.Space):
    """
    Validates preprocessing settings against the environment observation space.
    """
    if "game_settings" not in config or "preprocessing" not in config["game_settings"]:
        raise ValueError("Config missing 'game_settings.preprocessing' section.")

    prep_cfg = config["game_settings"]["preprocessing"]
    nnr_input_shape_cfg = tuple(config["global_network_vars"]["nnr_input_shape"])

    if prep_cfg.get("one_hot_encode_discrete_state", False):
        if not isinstance(raw_obs_space, gym.spaces.Discrete):
            raise ValueError(
                "Config requests 'one_hot_encode_discrete_state', but environment observation space "
                f"is not Discrete (found type: {type(raw_obs_space)})."
            )

        num_states_cfg = prep_cfg.get("num_discrete_states")
        if num_states_cfg is None:
            raise ValueError(
                "Config requests one-hot encoding, but 'game_settings.preprocessing.num_discrete_states' is not specified."
            )
        if not isinstance(num_states_cfg, int) or num_states_cfg <= 0:
            raise ValueError(
                f"'num_discrete_states' must be a positive integer, got {num_states_cfg} (type: {type(num_states_cfg)})."
            )

        env_n = getattr(raw_obs_space, "n", None)
        if env_n is None:
            raise ValueError(
                "Environment's Discrete observation space is missing the '.n' attribute."
            )

        if env_n != num_states_cfg:
            logger.warning(
                f"Config 'num_discrete_states' ({num_states_cfg}) does not match environment's observation space size ({env_n}). "
                f"Validation will use env size ({env_n}) for consistency checks; config value is used during preprocessing."
            )

        expected_one_hot_dim = (env_n,)
        if nnr_input_shape_cfg != expected_one_hot_dim:
            raise ValueError(
                f"Config requests one-hot encoding with {env_n} states, "
                f"but 'global_network_vars.nnr_input_shape' is {nnr_input_shape_cfg}. "
                f"Expected: {expected_one_hot_dim}."
            )

    if prep_cfg.get("resize_dims") is not None:
        resize_dims = prep_cfg["resize_dims"]
        if (
            not isinstance(resize_dims, list)
            or len(resize_dims) != 2
            or not all(isinstance(d, int) and d > 0 for d in resize_dims)
        ):
            raise ValueError(
                f"Config 'resize_dims' must be a list of two positive integers [Height, Width], got {resize_dims}"
            )

    logger.debug("Preprocessing config validated.")


def finalize_config(base_config: Dict[str, Any], env_manager: Any) -> Dict[str, Any]:
    """
    Completes and validates configuration after environment initialization.
    Injects env-derived values (e.g., action_dim) and performs consistency checks.
    """
    logger.info("Finalizing configuration...")

    if not hasattr(env_manager, "action_space_size"):
        raise AttributeError(
            "Environment manager missing 'action_space_size' attribute."
        )
    action_dim_value = env_manager.action_space_size

    if action_dim_value is None or not isinstance(action_dim_value, (int, np.integer)):
        action_space_type = getattr(
            getattr(env_manager, "env", None), "action_space", "Unknown"
        )
        err_msg = (
            f"Environment manager must provide an integer 'action_space_size' (Discrete action space required). "
            f"Got: {action_dim_value} (type: {type(action_dim_value)}). Env action space: {action_space_type}."
        )
        logger.error(err_msg)
        raise TypeError(err_msg)
    action_dim = int(action_dim_value)

    if (
        not hasattr(env_manager, "observation_space")
        or env_manager.observation_space is None
    ):
        raise AttributeError(
            "Environment manager must have a valid 'observation_space' attribute."
        )
    raw_obs_space = env_manager.observation_space

    final_config = copy.deepcopy(base_config)
    if "global_network_vars" not in final_config:
        final_config["global_network_vars"] = {}
    final_config["global_network_vars"]["action_dim"] = action_dim

    validation_passed = False
    try:
        logger.debug("Running validation checks...")
        _validate_config_sections(final_config)
        _validate_network_config(final_config)

        final_config["global_network_vars"]["nnr_input_shape"] = tuple(
            final_config["global_network_vars"]["nnr_input_shape"]
        )

        _validate_preprocessing_config(final_config, raw_obs_space)

        validation_passed = True
        logger.debug("All validation checks passed.")
    except (KeyError, ValueError, AttributeError, TypeError) as e:
        logger.error(f"Configuration Validation Error ({type(e).__name__}): {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected Configuration Error during finalization: {e}")
        raise

    if validation_passed:
        logger.info("-" * 40)
        logger.info("Final Configuration Summary:")
        logger.info(
            f"  Environment Name: {final_config.get('game_settings', {}).get('env_name', 'N/A')}"
        )
        logger.info(f"  Environment Action Dim: {action_dim}")
        logger.info(f"  Raw Observation Space: {raw_obs_space}")
        logger.info("  Preprocessing:")
        logger.info(
            f"    Frame Stacking: {final_config.get('game_settings', {}).get('preprocessing', {}).get('use_frame_stacking', 'N/A')}"
        )
        logger.info(
            f"    Num Stacked: {final_config.get('game_settings', {}).get('preprocessing', {}).get('num_stacked_frames', 'N/A')}"
        )
        logger.info(
            f"    Grayscale: {final_config.get('game_settings', {}).get('preprocessing', {}).get('grayscale', 'N/A')}"
        )
        logger.info(
            f"    Resize Dims: {final_config.get('game_settings', {}).get('preprocessing', {}).get('resize_dims', 'N/A')}"
        )
        logger.info(
            f"    One-Hot Encode: {final_config.get('game_settings', {}).get('preprocessing', {}).get('one_hot_encode_discrete_state', 'N/A')}"
        )
        logger.info(
            f"  Representation Network (NNr) Input Shape: {final_config['global_network_vars']['nnr_input_shape']}"
        )
        logger.info(
            f"  Representation Network Type: {final_config['neural_network']['representation_network_type']}"
        )
        logger.info(
            f"  Latent Dimension: {final_config['global_network_vars']['latent_dim']}"
        )
        logger.info(
            f"  Unroll Steps (K): {final_config['neural_network']['unroll_steps']}"
        )
        logger.info(
            f"  MCTS Simulations: {final_config.get('umcts', {}).get('simulations', 'N/A')}"
        )
        logger.info(
            f"  Batch Size: {final_config.get('training', {}).get('batch_size', 'N/A')}"
        )
        logger.info(
            f"  Learning Rate: {final_config['global_network_vars']['learning_rate']}"
        )
        logger.info("-" * 40)
    else:
        logger.critical(
            "Configuration finalization failed validation without raising. This indicates a validation flow bug."
        )
        raise RuntimeError("Configuration validation failed unexpectedly.")

    return final_config


# ==============================================================================
# Observation preprocessing and history stacking
# ==============================================================================


def preprocess_observation(obs: Any, config: Dict[str, Any]) -> np.ndarray:
    """
    Applies configured preprocessing to a single raw observation.

    Supports:
      - One-hot encoding for discrete observations
      - Image preprocessing (optional grayscale, resize, normalization)
      - Float32 casting and NaN checks
    """
    try:
        processed_obs = np.array(obs, copy=True)
    except Exception as e:
        logger.error(
            f"Failed to convert observation to numpy array. Obs type: {type(obs)}. Error: {e}"
        )
        raise ValueError(f"Cannot process observation of type {type(obs)}") from e

    prep_cfg = config["game_settings"]["preprocessing"]

    # Discrete -> one-hot
    if prep_cfg.get("one_hot_encode_discrete_state", False):
        num_states = prep_cfg.get("num_discrete_states")
        if num_states is None:
            raise ValueError(
                "Config inconsistency: 'num_discrete_states' is None while one-hot encoding is enabled."
            )

        if not (
            processed_obs.ndim == 0 and np.issubdtype(processed_obs.dtype, np.integer)
        ):
            raise ValueError(
                f"Cannot one-hot encode: obs must be a single integer scalar; got dtype={processed_obs.dtype}, shape={processed_obs.shape}."
            )

        state_index = int(processed_obs.item())
        if not 0 <= state_index < num_states:
            raise ValueError(
                f"Cannot one-hot encode: index {state_index} out of bounds for num_states={num_states}."
            )

        one_hot_vector = np.zeros(num_states, dtype=np.float32)
        one_hot_vector[state_index] = 1.0
        logger.debug(
            f"One-hot encoded state {state_index} into vector size {num_states}"
        )
        return one_hot_vector

    # Image transforms
    is_image = isinstance(processed_obs, np.ndarray) and processed_obs.ndim >= 2
    apply_grayscale = prep_cfg.get("grayscale", False)
    resize_dims = prep_cfg.get("resize_dims")
    apply_resize = resize_dims is not None

    if is_image and (apply_grayscale or apply_resize):
        logger.debug(
            f"Processing image observation (shape={processed_obs.shape}, dtype={processed_obs.dtype})"
        )

        if apply_grayscale and processed_obs.ndim == 3 and processed_obs.shape[-1] >= 3:
            if processed_obs.dtype != np.uint8:
                max_val = np.max(processed_obs)
                if 1.1 < max_val <= 255.1:
                    processed_obs = processed_obs.astype(np.uint8)
                elif max_val <= 1.1:
                    processed_obs = (processed_obs * 255).astype(np.uint8)
                else:
                    logger.warning(
                        f"Grayscale on non-uint8 image with max {max_val:.2f}; attempting uint8 cast."
                    )
                    processed_obs = processed_obs.astype(np.uint8)

            try:
                processed_obs = cv2.cvtColor(processed_obs, cv2.COLOR_RGB2GRAY)
                if processed_obs.ndim == 2:
                    processed_obs = np.expand_dims(processed_obs, axis=-1)
                logger.debug(f"Applied grayscale. New shape: {processed_obs.shape}")
            except cv2.error as cv_err:
                logger.error(
                    f"OpenCV error during grayscale conversion (shape={processed_obs.shape}, dtype={processed_obs.dtype}): {cv_err}"
                )
                raise
            except Exception as e:
                logger.exception(f"Unexpected error during cv2.cvtColor: {e}")
                raise

        if apply_resize:
            target_h, target_w = resize_dims[0], resize_dims[1]
            target_size_wh = (target_w, target_h)  # OpenCV expects (W, H)
            current_h, current_w = processed_obs.shape[0], processed_obs.shape[1]

            if current_h != target_h or current_w != target_w:
                try:
                    processed_obs = cv2.resize(
                        processed_obs, target_size_wh, interpolation=cv2.INTER_AREA
                    )
                    if processed_obs.ndim == 2:
                        processed_obs = np.expand_dims(processed_obs, axis=-1)
                    logger.debug(
                        f"Resized image ({current_h}, {current_w}) -> {target_size_wh}. New shape: {processed_obs.shape}"
                    )
                except cv2.error as cv_err:
                    logger.error(
                        f"OpenCV error during resize to {target_size_wh} (shape={processed_obs.shape}, dtype={processed_obs.dtype}): {cv_err}"
                    )
                    raise
                except Exception as e:
                    logger.exception(f"Unexpected error during cv2.resize: {e}")
                    raise

        # Normalize / cast
        if processed_obs.dtype == np.uint8:
            processed_obs = processed_obs.astype(np.float32) / 255.0
            logger.debug("Normalized uint8 image to float32 [0, 1].")
        elif np.issubdtype(processed_obs.dtype, np.floating):
            max_val = np.max(processed_obs)
            if 1.1 < max_val < 255.1:
                logger.warning(
                    f"Float image max {max_val:.2f} suggests 0-255 range; normalizing by 255.0."
                )
                processed_obs = processed_obs / 255.0
            elif max_val > 255.1:
                logger.warning(
                    f"Float image max {max_val:.2f} >> 255; normalizing and clipping to [0,1]."
                )
                processed_obs = np.clip(processed_obs / 255.0, 0.0, 1.0)
            if processed_obs.dtype != np.float32:
                processed_obs = processed_obs.astype(np.float32)

        if not np.issubdtype(processed_obs.dtype, np.floating):
            logger.warning(
                f"Image dtype after processing is {processed_obs.dtype}; casting to float32."
            )
            processed_obs = processed_obs.astype(np.float32)
        elif processed_obs.dtype != np.float32:
            processed_obs = processed_obs.astype(np.float32)

        if processed_obs.ndim == 2:
            logger.warning("Processed image is 2D; adding channel axis (C=1).")
            processed_obs = np.expand_dims(processed_obs, axis=-1)

    # Non-image numeric observations: ensure float32
    else:
        if not np.issubdtype(processed_obs.dtype, np.floating):
            logger.debug(
                f"Observation dtype {processed_obs.dtype} is not float; casting to float32."
            )
            processed_obs = processed_obs.astype(np.float32)
        elif processed_obs.dtype != np.float32:
            processed_obs = processed_obs.astype(np.float32)

    if np.isnan(processed_obs).any():
        logger.error(f"NaN value detected in processed observation: {processed_obs}")
        raise ValueError("NaN value encountered during observation preprocessing.")

    logger.debug(
        f"Final processed observation - shape={processed_obs.shape}, dtype={processed_obs.dtype}"
    )
    return processed_obs


def stack_and_preprocess_history(
    raw_obs_history: List[Any], config: Dict[str, Any]
) -> np.ndarray:
    """
    Preprocesses a recent history of observations and assembles the final NNr input.
    Validates that the output shape matches `global_network_vars.nnr_input_shape`.
    """
    nn_cfg = config["neural_network"]
    prep_cfg = config["game_settings"]["preprocessing"]
    nnr_input_shape_cfg = tuple(config["global_network_vars"]["nnr_input_shape"])
    num_required_history = nn_cfg["representation_input_states"]

    if not raw_obs_history:
        raise ValueError("Cannot process empty raw_obs_history.")

    corrected_history = list(raw_obs_history)
    if len(corrected_history) < num_required_history:
        padding_needed = num_required_history - len(corrected_history)
        first_obs = corrected_history[0]
        corrected_history = [first_obs] * padding_needed + corrected_history
        logger.debug(
            f"Padded history from {len(raw_obs_history)} to {len(corrected_history)} frames."
        )

    relevant_history = corrected_history[-num_required_history:]

    processed_frames: List[np.ndarray] = []
    try:
        for i, obs in enumerate(relevant_history):
            logger.debug(
                f"Preprocessing history frame {i}/{len(relevant_history)-1}..."
            )
            processed_frames.append(preprocess_observation(obs, config))
    except Exception as e:
        logger.exception(f"Error during preprocessing of frame {i} in history: {e}")
        raise ValueError(f"Failed to preprocess frame {i} in history.") from e

    final_nnr_input: Optional[np.ndarray] = None
    use_stacking = prep_cfg.get("use_frame_stacking", False)
    num_stacked_frames = prep_cfg.get("num_stacked_frames", 0)

    if use_stacking and num_stacked_frames != num_required_history:
        logger.warning(
            f"'use_frame_stacking' is True but num_stacked_frames ({num_stacked_frames}) != "
            f"representation_input_states ({num_required_history}). Stacking last num_stacked_frames."
        )
        if num_stacked_frames > len(processed_frames):
            raise ValueError(
                f"Cannot stack {num_stacked_frames} frames; only {len(processed_frames)} available."
            )
        frames_to_stack = processed_frames[-num_stacked_frames:]
    elif use_stacking:
        frames_to_stack = processed_frames
        if len(frames_to_stack) != num_stacked_frames:
            raise RuntimeError(
                f"Logic error: frames_to_stack ({len(frames_to_stack)}) != num_stacked_frames ({num_stacked_frames})."
            )
    else:
        frames_to_stack = []

    if use_stacking and frames_to_stack:
        logger.debug(f"Stacking {len(frames_to_stack)} processed frames...")
        first_frame_shape = frames_to_stack[0].shape
        for i, frame in enumerate(frames_to_stack[1:], 1):
            if frame.shape != first_frame_shape:
                raise ValueError(
                    f"Cannot stack frames: inconsistent shapes. Frame0={first_frame_shape}, Frame{i}={frame.shape}."
                )

        if len(first_frame_shape) == 3 and first_frame_shape[-1] in [1, 3]:
            try:
                final_nnr_input = np.concatenate(frames_to_stack, axis=-1).astype(
                    np.float32
                )
                logger.debug(
                    f"Concatenated {len(frames_to_stack)} frames along channel axis -> {final_nnr_input.shape}"
                )
            except ValueError as e:
                shapes = [f.shape for f in frames_to_stack]
                logger.error(
                    f"Error during frame concatenation (axis=-1): {e}. Shapes: {shapes}."
                )
                raise
        else:
            logger.warning(
                f"Non-standard frame shape {first_frame_shape}; stacking along axis=0."
            )
            try:
                final_nnr_input = np.stack(frames_to_stack, axis=0).astype(np.float32)
            except ValueError as e:
                shapes = [f.shape for f in frames_to_stack]
                logger.error(
                    f"Error during fallback stacking (axis=0): {e}. Shapes: {shapes}."
                )
                raise

    elif not use_stacking:
        if not processed_frames:
            raise ValueError("No processed frames available.")
        final_nnr_input = processed_frames[-1].astype(np.float32)
        logger.debug(
            f"Frame stacking disabled. Using most recent frame -> {final_nnr_input.shape}"
        )

    if final_nnr_input is None:
        raise RuntimeError("Failed to produce final NNr input tensor.")

    if final_nnr_input.shape != nnr_input_shape_cfg:
        msg = (
            f"CRITICAL SHAPE MISMATCH: final NNr input shape {final_nnr_input.shape} "
            f"does not match expected {nnr_input_shape_cfg}."
        )
        logger.error(msg)
        if processed_frames:
            logger.error(f"  First processed frame shape: {processed_frames[0].shape}")
            logger.error(f"  Last processed frame shape: {processed_frames[-1].shape}")
        raise ValueError(msg)

    if final_nnr_input.dtype != np.float32:
        logger.warning(
            f"Final NNr input dtype was {final_nnr_input.dtype}; casting to float32."
        )
        final_nnr_input = final_nnr_input.astype(np.float32)

    logger.debug(
        f"Created NNr input tensor shape={final_nnr_input.shape}, dtype={final_nnr_input.dtype}"
    )
    return final_nnr_input
