# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["figure.figsize"] = (10, 8)


def load_experiment_data(target_dir="results/EMA vs monotone"):
    # Create an empty list to store the data from each file
    all_data = []

    # Iterate over the files in the target directory
    for file_count, filename in enumerate(os.listdir(target_dir)):
        print(file_count)
        # if file_count == max_file_count:
        #     break

        # Construct the full file path
        file_path = os.path.join(target_dir, filename)

        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Open the file and read the data
            with open(file_path, "r") as f:
                # Create an empty list to store the data from the current file
                file_data = []
                for line in f:
                    # Split the line into columns
                    columns = line.strip().split()

                    # Convert the columns to integers and append them to the file data
                    file_data.append(np.array([int(c)
                                     for c in columns], dtype=int))

            # Convert the data from the current file to an ndarray
            file_data = np.array(file_data)

            # Append the data from the current file to the overall data list
            all_data.append(file_data)

    # Concatenate the data from all the files into a single ndarray
    data = np.stack(all_data)
    return data


def plot_average_reward_over_time(experiment_data, title, agent1_name="EMA", agent2_name="Monotone", steps=None):
    nb_experiments, max_steps, _ = experiment_data.shape
    # _: agent1_actions, agent2_actions, agent1_rewards, agent2_rewards
    if steps is None:
        steps = max_steps

    # mean of axis 0 -> wants seq of len t
    agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, 2], axis=0)  # EMA
    agent2_avg_rewards = np.mean(
        experiment_data[:, 0:steps:, 3], axis=0)  # Always 2

    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(steps)

    # moving average
    val1_mean = pd.Series(agent1_avg_rewards).rolling(100).mean()
    val2_mean = pd.Series(agent2_avg_rewards).rolling(100).mean()

    # Create the plot using matplotlib's plot function
    plt.plot(time, agent1_avg_rewards,
             label=f'{agent1_name} agent avg rewards')
    plt.plot(time, agent2_avg_rewards,
             label=f'{agent2_name} agent avg rewards')

    plt.plot(time, val1_mean,
             label=f'{agent1_name} agent rewards mean (window size = 100)')
    plt.plot(time, val2_mean,
             label=f'{agent2_name} agent rewards mean (window size = 100)')

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()

    # Plot the data as before
    plt.show()


# %%
# EMA VS MONOTONE
# Set the figure size to width=10 and height=8


experiment_data1 = load_experiment_data(target_dir="results/EMA vs monotone")

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=50)

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=6_000)

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=30_000)

plot_average_reward_over_time(
    experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone")


# %%

# OMNISCIENT VS MONOTONE
experiment_data2 = load_experiment_data(target_dir="results/Omniscient vs monotone")

plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
                              steps=50)

plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
                              steps=6_000)

plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
                              steps=30_000)

plot_average_reward_over_time(
    experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone")



# %%
