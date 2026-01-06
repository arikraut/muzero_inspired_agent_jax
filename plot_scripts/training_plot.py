import pandas as pd
import matplotlib.pyplot as plt
import os

csv_filename = "./muzero_checkpoints/CartPole_v1_baseline/training_stats.csv"

if not os.path.exists(csv_filename):
    print(f"Error: File not found at '{csv_filename}'")
    print("Please make sure the CSV file exists in the same directory as the script,")
    print("or provide the full path to the file.")
else:
    try:
        data = pd.read_csv(csv_filename)

        print("Data loaded successfully:")
        print(data.head())
        print("\nPlotting...")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            data["TrainingStep"],
            data["AvgEpisodeReward"],
            marker="o",
            linestyle="-",
            color="b",
            label="Avg Episode Reward",
        )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value")
        ax.set_title("MuZero CartPole Training Progress")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        plt.show()

        save_filename = "cartpole_training_plot.png"
        fig.savefig(save_filename)
        print(f"Plot saved as '{save_filename}'")

    except Exception as e:
        print(f"An error occurred while reading or plotting the data: {e}")
        print(
            "Please ensure the CSV file is formatted correctly with the expected headers:"
        )
        print("TrainingStep,AvgEpisodeSteps,AvgEpisodeReward")
