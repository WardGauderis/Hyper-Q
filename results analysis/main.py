# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%

# ChatGPT:
# generate python code that given file output.txt with records of the following structure:
# <val1>, <val2>, <val3>, <val4>
# <val1>, <val2>, <val3>, <val4>
# <val1>, <val2>, <val3>, <val4>

# loads the data into numpy arrays, one for array for ach column

# Load the data from the file into a list of strings, each string being a line
with open("results/output_EMA.txt") as f:
    data = f.readlines()

# Split each line by the comma separator and store the values in a list of lists
values = [line.strip().split(' ') for line in data]

# Convert the lists of strings to arrays of floats
agent1_actions = np.array(values, dtype=float)[:, 0]
agent2_actions = np.array(values, dtype=float)[:, 1]
agent1_rewards = np.array(values, dtype=float)[:, 2]
agent2_rewards = np.array(values, dtype=float)[:, 3]

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

plot_rewards_over_time(agent1_rewards, agent2_rewards, "EMA vs EMA")
# %%
