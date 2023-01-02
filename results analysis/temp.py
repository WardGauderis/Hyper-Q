# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["figure.figsize"] = (10, 8)

# %%


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
                    file_data.append(np.array([c
                                     for c in columns], dtype=float))

            # Convert the data from the current file to an ndarray
            file_data = np.array(file_data)

            # Append the data from the current file to the overall data list
            all_data.append(file_data)

    # Concatenate the data from all the files into a single ndarray
    data = np.stack(all_data)
    return data


# from wards assignment
def plot_returns(all_returns, all_evaluation_returns, window_size, evaluate_every, filename):
    """
    Plot the returns of all training episodes and the averaged evaluation return of each evaluation.

    :param all_returns: The returns of all training episodes per run.
    :param all_evaluation_returns: The averaged evaluation return of each evaluation per run.
    :param window_size: The size of the window for the moving average.
    :param evaluate_every: Frequency of evaluation.
    :param filename: The filename to save the plot to.
    :return: None
    """

    # Calculate mean and standard deviation of returns and evaluation returns.
    # Calculate a moving average of mean and standard deviation of the returns.
    all_returns_mean = np.mean(all_returns, axis=0)
    all_returns_std = np.std(all_returns, axis=0)

    all_returns_mean = np.convolve(all_returns_mean, np.ones(
        window_size) / window_size, mode="valid")
    all_returns_std = np.convolve(all_returns_std, np.ones(
        window_size) / window_size, mode="valid")
    all_evaluation_returns_mean = np.mean(all_evaluation_returns, axis=0)
    all_evaluation_returns_std = np.std(all_evaluation_returns, axis=0)

    # Plot the returns and evaluation returns with standard deviation
    plt.plot(all_returns_mean,
             label=f"Training (running average over {window_size} episodes)")
    plt.fill_between(np.arange(len(all_returns_mean)), all_returns_mean - all_returns_std,
                     all_returns_mean + all_returns_std, alpha=0.2)
    plt.plot(np.arange(evaluate_every, all_returns.shape[1] + 1, evaluate_every), all_evaluation_returns_mean,
             label="Evaluation")

    plt.fill_between(np.arange(evaluate_every, all_returns.shape[1] + 1, evaluate_every),
                     all_evaluation_returns_mean - all_evaluation_returns_std,
                     all_evaluation_returns_mean + all_evaluation_returns_std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

    # plt.gcf().set_size_inches(15, 4)
    # plt.savefig(filename, dpi=300, bbox_inches="tight")


def plot_average_reward_over_time(experiment_data, title,
                                  agent1_name="", agent2_name="",
                                  ma_window_size=2000,
                                  symmetrical_reward=True,
                                  steps=None, file_name=None):
    nb_experiments, max_steps, _ = experiment_data.shape

    if steps is None:
        steps = max_steps

    # mean of axis 0 -> wants seq of len t
    agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, 2], axis=0)
    std = np.std(experiment_data[:, 0:steps, 2], axis=0)

    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(steps)

    # moving average
    ma_rewards = pd.Series(agent1_avg_rewards).rolling(ma_window_size).mean()
    std_ma = pd.Series(std).rolling(ma_window_size).mean()

    # Create the plot using matplotlib's plot function
    plt.plot(time, ma_rewards, label=f'Rewards {agent1_name} MA (window size = {ma_window_size})')
    plt.fill_between(np.arange(len(ma_rewards)), ma_rewards - std_ma, ma_rewards + std_ma, alpha=0.2)

    if not symmetrical_reward:
        # mean of axis 0 -> wants seq of len t
        agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, 3], axis=0)
        std = np.std(experiment_data[:, 0:steps, 3], axis=0)

        # Generate a range of indices from 0 to the length of val1 - 1
        time = range(steps)

        # moving average
        ma_rewards = pd.Series(agent1_avg_rewards).rolling(ma_window_size).mean()
        std_ma = pd.Series(std).rolling(ma_window_size).mean()

        # Create the plot using matplotlib's plot function
        plt.plot(time, ma_rewards, label=f'Rewards {agent2_name} MA (window size = {ma_window_size})')
        plt.fill_between(np.arange(len(ma_rewards)), ma_rewards - std_ma, ma_rewards + std_ma, alpha=0.2)



    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    # Plot the data as before
    plt.show()


def plot_strategy_over_time_single_agent(strategies_over_time, title, steps=None, file_name=None,  ma_window_size=2000):
    nb_experiments, max_steps, _ = strategies_over_time.shape
    if steps is None:
        steps = max_steps

    agent_strategies = strategies_over_time[:, 0:steps]
    agent_strategy = np.mean(agent_strategies, axis=0)

    # std = np.std(agent_strategies, axis=0)

    time = range(steps)

    # moving average
    action0 = pd.Series(agent_strategy[:, 0]).rolling(ma_window_size).mean()
    action1 = pd.Series(agent_strategy[:, 1]).rolling(ma_window_size).mean()
    action2 = pd.Series(agent_strategy[:, 2]).rolling(ma_window_size).mean()

    # std_ma0 = pd.Series(std[:, 0]).rolling(ma_window_size).mean()
    # std_ma1 = pd.Series(std[:, 1]).rolling(ma_window_size).mean()
    # std_ma2 = pd.Series(std[:, 2]).rolling(ma_window_size).mean()

    # plt.fill_between(np.arange(len(agent_strategy[:, 0])),
    #                  agent_strategy[:, 0] - std_ma0,
    #                  agent_strategy[:, 0] + std_ma0,
    #                  alpha=0.2)

    # plt.fill_between(np.arange(len(agent_strategy[:, 1])),
    #                  agent_strategy[:, 1] - std_ma0,
    #                  agent_strategy[:, 1] + std_ma0,
    #                  alpha=0.2)

    # plt.fill_between(np.arange(len(agent_strategy[:, 2])),
    #                  agent_strategy[:, 2] - std_ma0,
    #                  agent_strategy[:, 2] + std_ma0,
    #                  alpha=0.2)

    plt.plot(time, action0,
             label=f'action: {0} probability MA (window size = {ma_window_size})')
    plt.plot(time, action1,
             label=f'action: {1} probability MA (window size = {ma_window_size})')
    plt.plot(time, action2,
             label=f'action: {2} probability MA (window size = {ma_window_size})')

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Strategy')
    plt.legend()

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    plt.show()


# %%
# **********************************************
# ************ Rock paper scisors  *************
# **********************************************
################ IGA VS MONOTONE ###############


experiment_data1 = load_experiment_data(target_dir="results/IGA vs monotone")

# %%

plot_average_reward_over_time(experiment_data1,
                              title="IGA vs Monotone",
                              agent1_name="IGA",
                              agent2_name="Monotone",
                              steps=600_000,
                              file_name='IGA vs monotone',
                              symmetrical_reward=False)


# %%

################ EMA VS MONOTONE ###############


experiment_data1 = load_experiment_data(target_dir="results/EMA vs monotone")

# %%

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=50)

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=6_000)

plot_average_reward_over_time(experiment_data1, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=30_000)

plot_average_reward_over_time(
    experiment_data1,
    title="EMA vs Monotone",
    agent1_name="EMA",
    agent2_name="Monotone")


# %%

################ OMNISCIENT VS MONOTONE ###############

# experiment_data2 = load_experiment_data(target_dir="results/Omniscient vs monotone")

# plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
#                               steps=50)

# plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
#                               steps=6_000)

# plot_average_reward_over_time(experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone",
#                               steps=30_000)

# plot_average_reward_over_time(
#     experiment_data2, title="Omniscient vs Monotone", agent1_name="Omniscient", agent2_name="Monotone")


# %%

################ BAYESIAN VS MONOTONE ###############


# %%
# **********************************************
# ************** COOPERATION GAME # ************
# **********************************************
################# OMNISCIENT ####################
experiment_coop_data = load_experiment_data(
    target_dir="results/cooperation/Omniscient vs Omniscient")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Cooperation game - Omniscient vs Omniscient - Reward over time",
                              agent1_name="EMA",
                              agent2_name="Monotone",
                              steps=600_000,
                              file_name='coop omniscient vs omniscient')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="Omniscient agent1 strategy evolution",
    steps=600_000,
    file_name="coop omniscient vs omniscient agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    title="Omniscient agent2 strategy evolution",
    file_name="coop omniscient vs omniscient agent2 actions",
    steps=600_000)

# %%


################# EMA ####################
experiment_coop_data = load_experiment_data(
    target_dir="results/cooperation/EMA vs EMA")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Cooperation game - EMA vs EMA - Reward over time",
                              agent1_name="EMA",
                              agent2_name="Monotone",
                              steps=600_000,
                              file_name='coop EMA vs EMA')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="EMA agent1 strategy evolution",
    steps=600_000,
    file_name="coop EMA vs EMA agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    title="EMA agent2 strategy evolution",
    file_name="coop EMA vs EMA agent2 actions",
    steps=600_000)

# %%


################# Bayesian ##################
experiment_coop_data = load_experiment_data(
    target_dir="results/cooperation/Bayesian vs Bayesian2")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Bayesian game - Bayesian vs Bayesian - Reward over time",
                              agent1_name="Bayesian",
                              agent2_name="Monotone",
                              ma_window_size=20_000,
                              steps=600_000,
                              file_name='coop Bayesian vs Bayesian')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="Bayesian agent1 strategy evolution",
    steps=600_000,
    ma_window_size=20_000,
    file_name="coop Bayesian vs Bayesian agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    ma_window_size=20_000,
    title="Bayesian agent2 strategy evolution",
    file_name="coop Bayesian vs Bayesian agent2 actions",
    steps=600_000)

# %%
