# live_plot_history.py
import argparse
import json
import os
import glob
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_latest_history_file(checkpoint_dir, base_filename="muzero_history_"):
    files = glob.glob(os.path.join(checkpoint_dir, f"{base_filename}*.json"))
    if not files:
        return None, -1
    latest_file = None
    max_step = -1
    for f in files:
        try:
            step_str = (
                os.path.basename(f).replace(base_filename, "").replace(".json", "")
            )
            step = int(step_str)
            if step > max_step:
                max_step = step
                latest_file = f
        except ValueError:
            logging.warning(f"Could not parse step number from filename: {f}")
            continue
    return latest_file, max_step


def live_plot_loss(checkpoint_dir, poll_interval=30, window_size=50):
    if not os.path.isdir(checkpoint_dir):
        logging.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.ion()
    fig.show()
    fig.canvas.draw()

    last_plotted_step = -1
    plot_needs_update = True

    logging.info(
        f"Starting live plot. Polling '{checkpoint_dir}' every {poll_interval}s..."
    )

    try:
        while True:
            latest_file, latest_step = find_latest_history_file(checkpoint_dir)

            if latest_file and latest_step > last_plotted_step:
                logging.info(
                    f"Found newer history file: {os.path.basename(latest_file)} (step {latest_step})"
                )
                last_plotted_step = latest_step
                plot_needs_update = True

                try:
                    with open(latest_file, "r") as f:
                        history_data = json.load(f)
                    df = pd.DataFrame(history_data)
                    required_cols = [
                        "total_loss",
                        "policy_loss",
                        "value_loss",
                        "reward_loss",
                    ]
                    plot_cols = [col for col in required_cols if col in df.columns]

                    if df.empty or not plot_cols:
                        logging.warning(
                            f"No data or required columns found in {latest_file}"
                        )
                        plot_needs_update = False
                    else:
                        ax.cla()
                        steps = np.arange(len(df)) + 1
                        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_cols)))
                        for i, col in enumerate(plot_cols):
                            if len(df[col]) >= window_size:
                                smoothed = (
                                    df[col]
                                    .rolling(
                                        window=window_size, min_periods=1, center=True
                                    )
                                    .mean()
                                )
                            else:
                                smoothed = df[col]
                            ax.plot(steps, df[col], alpha=0.3, color=colors[i])
                            ax.plot(
                                steps,
                                smoothed,
                                label=f"{col} (smoothed {window_size})",
                                color=colors[i],
                                linewidth=2,
                            )

                        ax.set_xlabel("Training Steps")
                        ax.set_ylabel("Loss Value (log scale)")
                        ax.set_title(
                            f"MuZero Training Loss (Live) - Step: {latest_step}\nDirectory: {checkpoint_dir}"
                        )
                        ax.legend()
                        ax.set_yscale("log")
                        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

                        plt.tight_layout()

                except Exception as e:
                    logging.error(f"Error reading or processing {latest_file}: {e}")
                    plot_needs_update = False

            elif not latest_file and last_plotted_step == -1:
                logging.info("No history files found yet...")
                plot_needs_update = False

            if plot_needs_update:
                logging.info("Updating plot...")
                try:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plot_needs_update = False
                except Exception as e:
                    logging.error(f"Error updating plot window: {e}")
                    break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logging.info("Stopping live plot.")
    finally:
        plt.ioff()
        if plt.fignum_exists(fig.number):
            plt.close(fig)
        logging.info("Plot window closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live plot MuZero loss history by polling a directory."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory where checkpoints and history files are saved.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=30,
        help="Polling interval in seconds to check for new history files.",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=50,
        help="Window size for moving average smoothing.",
    )
    args = parser.parse_args()
    live_plot_loss(args.checkpoint_dir, args.interval, args.window)
