# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%

def load_experiment_data(target_dir="results/EMA vs monotone", max_file_count=20):
    # Create an empty list to store the data from each file
    all_data = []

    # Iterate over the files in the target directory
    for file_count, filename in enumerate(os.listdir(target_dir)):
        print(file_count)
        if file_count == max_file_count:
            break

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


experiment_data = load_experiment_data(target_dir="results/EMA vs monotone")


# %%

# # ChatGPT:
# # generate python code that given file output.txt with records of the following structure:
# # <val1>, <val2>, <val3>, <val4>
# # <val1>, <val2>, <val3>, <val4>
# # <val1>, <val2>, <val3>, <val4>

# # loads the data into numpy arrays, one for array for each column
# def load_file(path="results/output_EMA.txt"):
#     # Load the data from the file into a list of strings, each string being a line
#     with open(path) as f:
#         data = f.readlines()

#     # Split each line by the comma separator and store the values in a list of lists
#     values = [line.strip().split(' ') for line in data]

#     # Convert the lists of strings to arrays of floats
#     agent1_actions = np.array(values, dtype=float)[:, 0]
#     agent2_actions = np.array(values, dtype=float)[:, 1]
#     agent1_rewards = np.array(values, dtype=float)[:, 2]
#     agent2_rewards = np.array(values, dtype=float)[:, 3]

#     return agent1_actions, agent2_actions, agent1_rewards, agent2_rewards


# agent1_actions, agent2_actions, agent1_rewards, agent2_rewards = load_file("results/EMA VS monotone/experiment_0.txt")

# %%


# # Set the target directory
# target_dir = "./results/EMA vs monotone/"
# experiment_results = [load_file(os.path.join(target_dir, filename)) for filename in os.listdir(target_dir)]
# experiment_results = np.array(experiment_results)

# Iterate over the files in the target directory
# for filename in os.listdir(target_dir):
#     # Construct the full file path
#     file_path = os.path.join(target_dir, filename)

#     # Check if the file is a regular file (not a directory)
#     if os.path.isfile(file_path):
#         # Call the function to process the file
#         process_file(file_path)


# %%


def plot_rewards_over_time(agent1_rewards, agent2_rewards, title):
    # or the list of val1 and val2, generate a plot of the values over time. add labels, titles, legends
    val1_mean = pd.Series(agent1_rewards).rolling(1000).mean()
    val2_mean = pd.Series(agent2_rewards).rolling(1000).mean()

    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(len(agent1_rewards))

    # Create the plot using matplotlib's plot function
    plt.plot(time, agent1_rewards, label='agent1_actions')
    plt.plot(time, agent2_rewards, label='agent2_actions')

    plt.plot(time, val1_mean, label='agent1_rewards mean (window size = 100)')
    plt.plot(time, val2_mean, label='agent2_rewards mean (window size = 100)')

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()

    # Show the plot
    plt.show()


# plot_rewards_over_time(agent1_rewards, agent2_rewards, "EMA vs EMA")
# %%


def plot_average_reward_over_time(experiment_data, title):
    nb_experiments, steps, _ = experiment_data.shape
    steps = 10
    steps = 6_000
    # steps = 1_000_000
    # _: agent1_actions, agent2_actions, agent1_rewards, agent2_rewards

    # mean of axis 0 -> wants seq of len t
    agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, 2], axis=0) # EMA
    agent2_avg_rewards = np.mean(experiment_data[:, 0:steps:, 3], axis=0) # Always 2

    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(steps)

    val1_mean = pd.Series(agent1_avg_rewards).rolling(100).mean()
    val2_mean = pd.Series(agent2_avg_rewards).rolling(100).mean()


    # Create the plot using matplotlib's plot function
    plt.plot(time, agent1_avg_rewards, label='EMA avg rewards')
    plt.plot(time, agent2_avg_rewards, label='Monotone avg rewards')

    plt.plot(time, val1_mean, label='EMA agent rewards mean (window size = 100)')
    plt.plot(time, val2_mean, label='Monotone agent rewards mean (window size = 100)')

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()



    # Plot the data as before
    plot_average_reward_over_time(experiment_data, title)

    # Show the plot
    plt.show()


# Set the figure size to width=10 and height=8
plt.rcParams["figure.figsize"] = (10, 8)

plot_average_reward_over_time(experiment_data, title="EMA vs Monotone")