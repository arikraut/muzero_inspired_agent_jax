# plot_history.py
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_loss_history(history_file_path, output_file=None, window_size=50):
    """
    Loads loss history from a JSON file and plots the losses.

    Args:
        history_file_path (str): Path to the muzero_history_*.json file.
        output_file (str, optional): Path to save the plot image. If None, displays the plot.
        window_size (int): Window size for the moving average smoothing.
    """
    if not os.path.exists(history_file_path):
        logging.error(f"History file not found: {history_file_path}")
        return

    logging.info(f"Loading history from: {history_file_path}")
    try:
        with open(history_file_path, "r") as f:
            history_data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {history_file_path}: {e}")
        return
    except Exception as e:
        logging.error(f"Error reading history file {history_file_path}: {e}")
        return

    if not history_data:
        logging.warning("History data is empty.")
        return

    try:
        df = pd.DataFrame(history_data)
        if df.empty:
            logging.warning("Created DataFrame is empty.")
            return

        required_cols = ["total_loss", "policy_loss", "value_loss", "reward_loss"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"History data missing required columns: {missing_cols}")
            plot_cols = [col for col in required_cols if col in df.columns]
            if not plot_cols:
                return
        else:
            plot_cols = required_cols

        steps = np.arange(len(df)) + 1

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_cols)))
        for i, col in enumerate(plot_cols):
            if len(df[col]) >= window_size:
                smoothed = (
                    df[col]
                    .rolling(window=window_size, min_periods=1, center=True)
                    .mean()
                )
            else:
                smoothed = df[col]

            ax.plot(steps, df[col], alpha=0.3, color=colors[i])
            ax.plot(
                steps,
                smoothed,
                label=f"{col} (smoothed {window_size} steps)",
                color=colors[i],
                linewidth=2,
            )

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss Value")
        ax.set_title(
            f"MuZero Training Loss History\n(File: {os.path.basename(history_file_path)})"
        )
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        if output_file:
            logging.info(f"Saving plot to: {output_file}")
            plt.savefig(output_file, dpi=300)
            plt.close(fig)
        else:
            logging.info("Displaying plot...")
            plt.show()

    except Exception as e:
        logging.exception(f"An error occurred during plotting: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MuZero loss history from a JSON file."
    )
    parser.add_argument(
        "history_file", type=str, help="Path to the muzero_history_*.json file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot image (e.g., loss_plot.png).",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=50,
        help="Window size for moving average smoothing.",
    )
    args = parser.parse_args()

    plot_loss_history(args.history_file, args.output, args.window)
