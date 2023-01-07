# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams["figure.figsize"] = (10, 8)

# %%


def load_experiment_data(target_dir="results3/EMA vs monotone"):
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
    plt.plot(time, ma_rewards,
             label=f'Rewards {agent1_name} MA (window size = {ma_window_size})')
    plt.fill_between(np.arange(len(ma_rewards)), ma_rewards -
                     std_ma, ma_rewards + std_ma, alpha=0.2)

    if not symmetrical_reward:
        # mean of axis 0 -> wants seq of len t
        agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, 3], axis=0)
        std = np.std(experiment_data[:, 0:steps, 3], axis=0)

        # Generate a range of indices from 0 to the length of val1 - 1
        time = range(steps)

        # moving average
        ma_rewards = pd.Series(agent1_avg_rewards).rolling(
            ma_window_size).mean()
        std_ma = pd.Series(std).rolling(ma_window_size).mean()

        # Create the plot using matplotlib's plot function
        plt.plot(time, ma_rewards,
                 label=f'Rewards {agent2_name} MA (window size = {ma_window_size})')
        plt.fill_between(np.arange(len(ma_rewards)), ma_rewards -
                         std_ma, ma_rewards + std_ma, alpha=0.2)

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")

    # Plot the data as before
    plt.show()


def plot_average_reward_hyperq_vs_other(hyper_q_experiments,
                                        title,
                                        agent_names=["Omniscient",
                                                     "EMA", "Bayesian"],
                                        ma_window_sizes=[2000, 2000, 20_000],
                                        agent_idx=3,
                                        symmetrical_reward=True,
                                        steps=None,
                                        file_name=None):

    if steps is None:
        steps = max_steps
    
    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(steps)

    for idx, experiment_data in enumerate(hyper_q_experiments):
        nb_experiments, max_steps, _ = experiment_data.shape

        # mean of axis 0 -> wants seq of len t
        agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, agent_idx], axis=0)



        # moving average
        ma_rewards = pd.Series(agent1_avg_rewards).rolling(
            ma_window_sizes[idx]).mean()

        # Create the plot using matplotlib's plot function
        plt.plot(time, ma_rewards,
                 label=f'{agent_names[idx]} MA (window size = {ma_window_sizes[idx]})')

        # Add a title, x and y labels, and a legend
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.legend()
    
    # Multiply the time values by 10 # beacuse c++ only saves every 10 steps
    tick_positions = range(0, steps, 5000)
    tick_labels = [x * 10 for x in tick_positions]
    plt.xticks(tick_positions, tick_labels)

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

plt.rcParams["figure.figsize"] = (10, 8)

REWARDS_PLAYER_X = 2
REWARDS_PLAYER_Y = 3

ACTIONS_PLAYER_X_STARTING_IDX = 4
ACTIONS_PLAYER_Y_STARTING_IDX = 7


# %%
# **********************************************
# ************ Rock paper scisors  *************
# **********************************************

################ IGA VS Omniscient, EMA, Bayesian  ###############
omniscient_experiment_data = load_experiment_data(target_dir="results3/IGA vs omniscient")
EMA_experiment_data = load_experiment_data(target_dir="results3/IGA vs EMA")
Bayesian_experiment_data = load_experiment_data(target_dir="results3/IGA vs Bayesian")
Bayesian_ultra_experiment_data = load_experiment_data(target_dir="results3/IGA vs Bayesian ultra")

# %%


################ IGA VS Omniscient, EMA, Bayesian  ###############
plot_average_reward_hyperq_vs_other([omniscient_experiment_data,
                                    EMA_experiment_data,
                                    Bayesian_experiment_data,
                                    Bayesian_ultra_experiment_data
                                    ],
                                    title="Hyper-Q vs. IGA: Avg. reward per time step",
                                    agent_names=["Omniscient",
                                                 "EMA", "Bayesian", "Bayesian ultra"],
                                    ma_window_sizes=[5000, 5000, 5000, 5000],
                                    symmetrical_reward=True,
                                    agent_idx=3,
                                    steps=120_000,
                                    file_name="HyperQ vs IGA Avg reward per time step")

# %%

omniscient_experiment_data = load_experiment_data(target_dir="results3/PHC vs omniscient")
EMA_experiment_data = load_experiment_data(target_dir="results3/PHC vs EMA")
Bayesian_experiment_data = load_experiment_data(target_dir="results3/PHC vs Bayesian")
Bayesian_ultra_experiment_data = load_experiment_data(target_dir="results3/PHC vs Bayesian ultra")

# %%

plot_average_reward_hyperq_vs_other([
                                    omniscient_experiment_data,
                                    EMA_experiment_data,
                                    Bayesian_experiment_data,
                                    Bayesian_ultra_experiment_data],
                                    title="Hyper-Q vs. PHC: Avg. reward per time step",
                                    agent_names=["Omniscient",
                                                 "EMA", "Bayesian", "Bayesian ultra"],
                                    ma_window_sizes=[5000, 5000, 5000, 5000],
                                    symmetrical_reward=True,
                                    agent_idx=REWARDS_PLAYER_Y,
                                    steps=120_000,
                                    file_name="HyperQ vs PHC Avg reward per time step")




# %%


################ PHC, IGA, Omniscient, EMA, Bayesian vs monotone ###############
experiment_vs_monotone1 = load_experiment_data(target_dir="results3/PHC vs monotone")
experiment_vs_monotone2 = load_experiment_data(target_dir="results3/IGA vs monotone")
experiment_vs_monotone3 = load_experiment_data(target_dir="results3/Omniscient vs monotone")
experiment_vs_monotone4 = load_experiment_data(target_dir="results3/EMA vs monotone")
experiment_vs_monotone5 = load_experiment_data(target_dir="results3/Bayesian vs monotone")
experiment_vs_monotone6 = load_experiment_data(target_dir="results3/Bayesian ultra vs monotone")

# %%

plot_average_reward_hyperq_vs_other([
                                    experiment_vs_monotone1,
                                    experiment_vs_monotone2,
                                    experiment_vs_monotone3,
                                    experiment_vs_monotone4,
                                    experiment_vs_monotone5,
                                    experiment_vs_monotone6
                                    ],
                                    title="PHC, IGA, Omniscient, EMA, Bayesian vs monotone: Avg. reward per time step",
                                    agent_names=[
                                        "PHC",
                                        "IGA",
                                        "Omniscient",
                                        "EMA",
                                        "Bayesian", "Bayesian ultra"],
                                    ma_window_sizes=[
                                        5000,
                                        5000,
                                        5000, 
                                        5000, 
                                        5000, 
                                        5000],
                                    symmetrical_reward=True,
                                    steps=120_000,
                                    agent_idx=REWARDS_PLAYER_X,
                                    file_name="HyperQ, IGA, PHC vs monotone")

# %%






# %%
# **********************************************
# ************** COOPERATION GAME # ************
# **********************************************


# %% Rewards

omniscient_experiment_data = load_experiment_data(target_dir="results3/cooperation/Omniscient vs Omniscient")
EMA_experiment_data = load_experiment_data(target_dir="results3/cooperation/EMA vs EMA")
Bayesian_experiment_data = load_experiment_data(target_dir="results3/cooperation/Bayesian vs Bayesian")
Bayesian_ultra_experiment_data = load_experiment_data(target_dir="results3/cooperation/Bayesian ultra vs Bayesian ultra")


# %%
plot_average_reward_hyperq_vs_other([
                                    omniscient_experiment_data,
                                    EMA_experiment_data,
                                    Bayesian_experiment_data,
                                    Bayesian_ultra_experiment_data],
                                    title="Hyper-Q cooperation game: Avg. reward per time step",
                                    agent_names=["Omniscient", "EMA", "Bayesian", "Bayesian ultra"],
                                    ma_window_sizes=[5000, 5000, 5000, 5000],
                                    symmetrical_reward=True,
                                    agent_idx=REWARDS_PLAYER_X,
                                    steps=60_000,
                                    file_name="HyperQ cooperation Avg reward per time step")

# %%
################# OMNISCIENT ####################
agent1_experiment1_strategies_over_time = omniscient_experiment_data[:, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
agent2_experiment1_strategies_over_time = omniscient_experiment_data[:, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]


plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Omniscient agent1 strategy evolution", file_name="coop omniscient vs omniscient agent1 actions", steps=60_000)
plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Omniscient agent2 strategy evolution", file_name="coop omniscient vs omniscient agent2 actions", steps=60_000)

################# EMA ####################
agent1_experiment1_strategies_over_time = EMA_experiment_data[:, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
agent2_experiment1_strategies_over_time = EMA_experiment_data[:, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: EMA agent1 strategy evolution", steps=60_000, file_name="coop EMA vs EMA agent1 actions")
plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: EMA agent2 strategy evolution", file_name="coop EMA vs EMA agent2 actions", steps=60_000)


################# Bayesian ##################
agent1_experiment1_strategies_over_time = Bayesian_experiment_data[:, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
agent2_experiment1_strategies_over_time = Bayesian_experiment_data[:, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Bayesian agent1 strategy evolution", steps=60_000, file_name="coop Bayesian vs Bayesian agent1 actions")
plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Bayesian agent2 strategy evolution", file_name="coop Bayesian vs Bayesian agent2 actions", steps=60_000)


################# Bayesian ##################
agent1_experiment1_strategies_over_time = Bayesian_ultra_experiment_data[:, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
agent2_experiment1_strategies_over_time = Bayesian_ultra_experiment_data[:, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Bayesian ultra agent1 strategy evolution", steps=60_000, file_name="coop Bayesian ultra vs Bayesian ultra agent1 actions")
plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Bayesian ultra agent2 strategy evolution", file_name="coop Bayesian vs Bayesian agent2 actions", steps=60_000)

# %%



















# %%

assert False


################ PHC VS MONOTONE ###############


experiment_data = load_experiment_data(target_dir="results3/PHC vs monotone")

plot_average_reward_over_time(experiment_data,
                              title="PHC vs Monotone",
                              agent1_name="PHC",
                              agent2_name="Monotone",
                              steps=120_000,
                              file_name='PHC vs monotone',
                              symmetrical_reward=False)


agent1_experiment1_strategies_over_time = experiment_data[:, :, 7:10]
plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="Monotone agent1 strategy evolution",
    steps=60_000,
    file_name="coop omniscient vs omniscient agent1 actions")


## %%

################ IGA VS MONOTONE ###############


experiment_data = load_experiment_data(target_dir="results3/IGA vs monotone")

# %%

plot_average_reward_over_time(experiment_data,
                              title="Rewards IGA vs Monotone",
                              agent1_name="IGA",
                              agent2_name="Monotone",
                              steps=120_000,
                              file_name='IGA vs monotone',
                              symmetrical_reward=False)


# %%

################ EMA VS MONOTONE ###############


experiment_data = load_experiment_data(target_dir="results3/EMA vs monotone")

# %%

plot_average_reward_over_time(experiment_data, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=50)

plot_average_reward_over_time(experiment_data, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=6_000)

plot_average_reward_over_time(experiment_data, title="EMA vs Monotone", agent1_name="EMA", agent2_name="Monotone",
                              steps=30_000)

plot_average_reward_over_time(
    experiment_data,
    title="EMA vs Monotone",
    agent1_name="EMA",
    agent2_name="Monotone")


# %%

################ OMNISCIENT VS MONOTONE ###############

# experiment_data2 = load_experiment_data(target_dir="results3/Omniscient vs monotone")

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
    target_dir="results3/cooperation/Omniscient vs Omniscient")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Cooperation game - Omniscient vs Omniscient - Reward over time",
                              agent1_name="EMA",
                              agent2_name="Monotone",
                              steps=60_000,
                              file_name='coop omniscient vs omniscient')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="Omniscient agent1 strategy evolution",
    steps=60_000,
    file_name="coop omniscient vs omniscient agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    title="Omniscient agent2 strategy evolution",
    file_name="coop omniscient vs omniscient agent2 actions",
    steps=60_000)

# %%


################# EMA ####################
experiment_coop_data = load_experiment_data(
    target_dir="results3/cooperation/EMA vs EMA")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Cooperation game - EMA vs EMA - Reward over time",
                              agent1_name="EMA",
                              agent2_name="Monotone",
                              steps=60_000,
                              file_name='coop EMA vs EMA')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="EMA agent1 strategy evolution",
    steps=60_000,
    file_name="coop EMA vs EMA agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    title="EMA agent2 strategy evolution",
    file_name="coop EMA vs EMA agent2 actions",
    steps=60_000)

# %%


################# Bayesian ##################
experiment_coop_data = load_experiment_data(
    target_dir="results3/cooperation/Bayesian vs Bayesian2")

# %%

plot_average_reward_over_time(experiment_coop_data,
                              title="Bayesian game - Bayesian vs Bayesian - Reward over time",
                              agent1_name="Bayesian",
                              agent2_name="Monotone",
                              ma_window_size=20_000,
                              steps=60_000,
                              file_name='coop Bayesian vs Bayesian')

# %%


agent1_experiment1_strategies_over_time = experiment_coop_data[:, :, 4:7]
agent2_experiment1_strategies_over_time = experiment_coop_data[:, :, 7:10]


plot_strategy_over_time_single_agent(
    agent1_experiment1_strategies_over_time,
    title="Bayesian agent1 strategy evolution",
    steps=60_000,
    ma_window_size=20_000,
    file_name="coop Bayesian vs Bayesian agent1 actions")

plot_strategy_over_time_single_agent(
    agent2_experiment1_strategies_over_time,
    ma_window_size=20_000,
    title="Bayesian agent2 strategy evolution",
    file_name="coop Bayesian vs Bayesian agent2 actions",
    steps=60_000)

# %%
