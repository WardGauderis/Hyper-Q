# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import platform
import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 8)
REWARDS_PLAYER_X = 2
REWARDS_PLAYER_Y = 3

ACTIONS_PLAYER_X_STARTING_IDX = 4
ACTIONS_PLAYER_Y_STARTING_IDX = 7

DATA_ROOT = "example_data/results_test/"


def load_experiment_data(target_dir):
    # Create an empty list to store the data from each file
    all_data = []
    # Iterate over the files in the target directory
    for _file_count, filename in enumerate(os.listdir(target_dir + "\\")):
        if filename == "config.json":
            continue

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
    try:
        data = np.stack(all_data)
    except:
        print("No data found in directory: ", target_dir)
        return None
    return data


def plot_average_reward_hyperq_vs_other(hyper_q_experiments,
                                        title,
                                        agent_names=["Omniscient",
                                                     "EMA", "Bayesian"],
                                        ma_window_sizes=[2000, 2000, 20_000],
                                        agent_idx=3,
                                        steps=None,
                                        file_name=None):

    if steps is None:
        nb_experiments, max_steps, _ = hyper_q_experiments[0].shape
        steps = max_steps

    # Generate a range of indices from 0 to the length of val1 - 1
    time = range(steps)

    marker_styles = [
        '.',
        ',',
        'o',
        'v',
        '^',
        '<',
        '>',
        '1',
        '2',
        '3',
        '4',
        's',
        'p',
        '*'
    ]

    subsampling_rate = 2000  # plot every 10th data point

    for idx, experiment_data in enumerate(hyper_q_experiments):
        nb_experiments, max_steps, _ = experiment_data.shape

        # mean of axis 0 -> wants seq of len t
        agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, agent_idx], axis=0)

        # moving average
        ma_rewards = pd.Series(agent1_avg_rewards).rolling(ma_window_sizes[idx]).mean()


        # Subsample the data by only plotting every nth point
        time_subsampled = time[::subsampling_rate]
        ma_rewards_subsampled = ma_rewards[::subsampling_rate]

        # Create the plot using matplotlib's plot function
        plt.plot(time_subsampled, ma_rewards_subsampled,
                 linestyle='solid',   # set the line style to solid
                 linewidth=1,         # set the line width to 1
                 # set the marker style to circles
                 marker=marker_styles[idx],
                 markersize=5,        # set the marker size to 3
                 label=f'{agent_names[idx]} MA (window size = {ma_window_sizes[idx]})')

        # Add a title, x and y labels, and a legend
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Avg. Reward')
        plt.legend()

    # Multiply the time values by 10 # beacuse c++ only saves every 10 steps
    if (steps == 120_000):
        tick_positions = range(0, steps, 20000)
        tick_labels = [x * 10 for x in tick_positions]
        plt.xticks(tick_positions, tick_labels)
    else:
        tick_positions = range(0, steps, 5000)
        tick_labels = [x * 10 for x in tick_positions]
        plt.xticks(tick_positions, tick_labels)

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.clf()

    # Plot the data as before
    # plt.show()

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

    subsampling_rate = 2000  # plot every 10th data point

    # Subsample the data by only plotting every nth point
    time_subsampled = time[::subsampling_rate]
    action0_subsampled = action0[::subsampling_rate]
    action1_subsampled = action1[::subsampling_rate]
    action2_subsampled = action2[::subsampling_rate]

    plt.plot(time_subsampled, action0_subsampled,
             label=f'action: {0} probability MA (window size = {ma_window_size})',
             linestyle='solid',
             linewidth=1,
             marker="o",
             markersize=5)

    plt.plot(time_subsampled, action1_subsampled,
             label=f'action: {1} probability MA (window size = {ma_window_size})',
             linestyle='solid',
             linewidth=1,
             marker="*",
             markersize=5)
    plt.plot(time_subsampled, action2_subsampled,
             label=f'action: {2} probability MA (window size = {ma_window_size})',
             linestyle='solid',
             linewidth=1,
             marker="v",
             markersize=5)
    # plt.plot(time_subsampled, ma_rewards_subsampled, linestyle='solid', linewidth=1, marker=marker_styles[idx], markersize=5)

    # Add a title, x and y labels, and a legend
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Strategy')
    plt.legend()

    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.clf()

    # plt.show()

def read_json_file(file_path):
    with open(file_path, 'r') as stream:
        # read content and return a dict
        return json.load(stream)


def get_directories_in_dir(target_dir):
    directories = [f.path for f in os.scandir(os.path.realpath(target_dir)) if f.is_dir()]
    # TODO: FOR LINUX/MAC: game_directory.split("/")[-1]
    if platform.system() == 'Windows':
        names = [directory.split("\\")[-1] for directory in directories]
    else:
        names = [directory.split("/")[-1] for directory in directories]
    return directories, names


def graph_per_agent():
    game_directories, game_names = get_directories_in_dir(DATA_ROOT)
    for game_directory, game_name in zip(game_directories, game_names):
        first_agents_directories, first_agents = get_directories_in_dir(game_directory)
        
        for first_agent_dir, first_agent in zip(first_agents_directories, first_agents):
            second_agents_directories, second_agents = get_directories_in_dir(first_agent_dir)
        
            runs_for_single_configuration = []
            for second_agent_dir, second_agent in zip(second_agents_directories, second_agents):
                configuration_directories, configurations = get_directories_in_dir(second_agent_dir)

                for configuration_dir, configuration in zip(configuration_directories, configurations):

                    config = read_json_file(os.path.join(configuration_dir, "config.json"))

                    nb_runs = config["runs"]
                    nb_iterations = config["iterations"]
                    exploration_type = config["exploration"]["type"] # eg rnd_restart
                    agent_x_type = config["agent_x"]["type"] # eg monotone
                    agent_y_type = config["agent_y"]["type"] # eg bayesian_hyper_q
                    game = config["game"] # eg rock_paper_scissors

                    runs = load_experiment_data(configuration_dir)
                    if runs is None:
                        print(f"Could not load data for {game_name}, {first_agent}, {second_agent}, {configuration}")
                        continue






                    plot_average_reward_hyperq_vs_other([ runs ],
                                        title=f"Multiple agents vs. {first_agent}: Avg. reward per time step",
                                        agent_names=[agent_x_type, agent_y_type],
                                        ma_window_sizes=[5000],
                                        agent_idx=REWARDS_PLAYER_X, #todo: maybe diff number, old was 3
                                        file_name=f"{DATA_ROOT}/{game_name}/{first_agent}/{second_agent}/{agent_x_type}_{agent_y_type}_{game}_{configuration}.png")
                    

                    agent_x_experiment_strategies_over_time = runs[:, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
                    agent_y_experiment_strategies_over_time = runs[:, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

                    plot_strategy_over_time_single_agent(agent_x_experiment_strategies_over_time, title=f"{game_name}: {agent_x_type} strategy evolution", ma_window_size=5000,
                                                        file_name=f"{DATA_ROOT}/{game_name}/{first_agent}/{second_agent}/strat_evol_{agent_x_type}__{agent_x_type}vs{agent_y_type}_{game}_{configuration}.png")
                    plot_strategy_over_time_single_agent(agent_y_experiment_strategies_over_time, title=f"{game_name}: {agent_y_type} strategy evolution", ma_window_size=5000,
                                                        file_name=f"{DATA_ROOT}/{game_name}/{first_agent}/{second_agent}/strat_evol_{agent_y_type}__{agent_x_type}vs{agent_y_type}_{game}_{configuration}.png")

                    avg_reward_X = np.mean(runs[:, :, REWARDS_PLAYER_X])
                    avg_reward_Y = np.mean(runs[:, :, REWARDS_PLAYER_Y])
                    print(f"G: {game_name}, first agent: {first_agent}, second agent: {second_agent}, x_t: {agent_x_type}, y_t: {agent_y_type}, config: {configuration} - AVG X: { format(avg_reward_X,'.4f') } AVG Y: {format(avg_reward_Y,'.4f')}")


def average_reward_hyperq_vs_other(hyper_q_experiments, ma_window_sizes, subsampling_rate, agent_idx, steps=None, filename=None):
    if steps is None:
        nb_experiments, max_steps, _ = hyper_q_experiments[0].shape
        steps = max_steps

    for idx, experiment_data in enumerate(hyper_q_experiments):
        nb_experiments, max_steps, _ = experiment_data.shape

        # mean of axis 0 -> wants seq of len t
        agent1_avg_rewards = np.mean(experiment_data[:, 0:steps, agent_idx], axis=0)

        # compute moving average using np 
        ma_rewards = np.convolve(agent1_avg_rewards, np.ones((ma_window_sizes,))/ma_window_sizes, mode='valid')

    subsampled = ma_rewards[::subsampling_rate]

    if filename is not None:
        # save via np csv
        np.savetxt(filename, subsampled, delimiter=",")

    return subsampled


def save_rewards_file():
    game_directories, game_names = get_directories_in_dir(DATA_ROOT)
    for game_directory, game_name in zip(game_directories, game_names):
        first_agents_directories, first_agents = get_directories_in_dir(game_directory)
        data_series = None # (nb_players, nb_rewards)
        
        for first_agent_dir, first_agent in zip(first_agents_directories, first_agents):
            second_agents_directories, second_agents = get_directories_in_dir(first_agent_dir)
        
            runs_for_single_configuration = []
            for second_agent_dir, second_agent in zip(second_agents_directories, second_agents):
                configuration_directories, configurations = get_directories_in_dir(second_agent_dir)

                for configuration_dir, configuration in zip(configuration_directories, configurations):

                    config = read_json_file(os.path.join(configuration_dir, "config.json"))

                    nb_runs = config["runs"]
                    nb_iterations = config["iterations"]
                    exploration_type = config["exploration"]["type"] # eg rnd_restart
                    agent_x_type = config["agent_x"]["type"] # eg monotone
                    agent_y_type = config["agent_y"]["type"] # eg bayesian_hyper_q
                    game = config["game"] # eg rock_paper_scissors

                    runs = load_experiment_data(configuration_dir)
                    if runs is None:
                        print(f"Could not load data for {game_name}, {first_agent}, {second_agent}, {configuration}")
                        continue


                    data = average_reward_hyperq_vs_other([ runs ],
                                        ma_window_sizes=100_000,
                                        subsampling_rate=200,
                                        agent_idx=REWARDS_PLAYER_X, 
                                        filename=f"{DATA_ROOT}/{game_name}/ag_x_rewards_{agent_x_type}_{agent_y_type}_{game}_{configuration}.csv"
                                        )
                    
                    print(data.shape) #(251,)
                
                    avg_reward_X = np.mean(runs[:, :, REWARDS_PLAYER_X])
                    avg_reward_Y = np.mean(runs[:, :, REWARDS_PLAYER_Y])
                    print(f"G: {game_name}, first agent: {first_agent}, second agent: {second_agent}, x_t: {agent_x_type}, y_t: {agent_y_type}, config: {configuration} - AVG X: { format(avg_reward_X,'.4f') } AVG Y: {format(avg_reward_Y,'.4f')}")


# def plot_average_rewards(directory):
#     # load in csv files from directory and plot them
#     files = os.listdir(directory)
#     csv_files = [file for file in files if file.endswith(".csv")]
#     csv_files = sorted(csv_files)

#     # plot all csv files
#     # csv_file = csv_files[0]
#     # for csv_file in csv_files:
#         # if csv_flie contains "phc"
#         # if "phc" in csv_file:
#     data1 = np.loadtxt(os.path.join(directory, csv_files[0]), delimiter=",")
#     data2 = np.loadtxt(os.path.join(directory, csv_files[1]), delimiter=",")
#     plt.plot(data1)
#     plt.plot(data2)

#     data3 = np.loadtxt(os.path.join(directory, csv_files[2]), delimiter=",")
#     data4 = np.loadtxt(os.path.join(directory, csv_files[3]), delimiter=",")


#     # create a secondary y-axis
#     ax2 = plt.gca().twinx()

#     # plot the second line on the secondary y-axis
#     ax2.plot(data3, color='r')
#     ax2.plot(data4, color='r')

#     # show the plot
#     plt.show()





def plot_average_rewards(directory):
    # load in csv files from directory and plot them
    files = os.listdir(directory)
    csv_files = [file for file in files if file.endswith(".csv")]
    csv_files = sorted(csv_files)

    data1 = np.loadtxt(os.path.join(directory, csv_files[0]), delimiter=",")
    data2 = np.loadtxt(os.path.join(directory, csv_files[1]), delimiter=",")
    data3 = np.loadtxt(os.path.join(directory, csv_files[2]), delimiter=",")
    data4 = np.loadtxt(os.path.join(directory, csv_files[3]), delimiter=",")

    # use a different color palette
    colors = sns.color_palette('muted')

    # plot the first line on the primary y-axis
    fig, ax1 = plt.subplots()
    ax1.plot(data1, color=colors[0], linewidth=2, label='Line 1 (Y1 axis)')
    ax1.plot(data2, color=colors[1], linewidth=2, label='Line 2 (Y1 axis)')
    ax1.set_xlabel('X axis', fontsize=14)
    ax1.set_ylabel('Y1 axis', color=colors[0], fontsize=14)
    ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=12)

    # create a secondary y-axis with a different scale
    ax2 = ax1.twinx()
    ax2.plot(data3, color=colors[2], linewidth=2, label='Line 3 (Y2 axis)')
    ax2.plot(data4, color=colors[3], linewidth=2, label='Line 4 (Y2 axis)')
    ax2.set_ylabel('Y2 axis', color=colors[2], fontsize=14)
    ax2.tick_params(axis='y', labelcolor=colors[2], labelsize=12)

    # adjust the scale of the secondary y-axis
    ax2.set_ylim(min(data3.min(), data4.min()), max(data3.max(), data4.max()) * 1.1)

    # set some space between the two y-axes
    plt.subplots_adjust(right=0.8)
    ax2.spines['right'].set_position(('axes', 1.1))

    # add gridlines
    ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)
    ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # add legend to indicate which line corresponds to which axis
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=12)

    plt.show()





if __name__=="__main__":
    # plot_average_rewards(r"C:\GitHub\Hyper-Q\results analysis\example_data\results_test\rock_paper_scissors")





    # %%
    data



    # %%


# plt.rcParams["figure.figsize"] = (10, 8)




# # %%
# # **********************************************
# # ************ Rock paper scisors  *************
# # **********************************************

# ################ IGA VS Omniscient, EMA, Bayesian  ###############
# omniscient_experiment_data = load_experiment_data(
#     target_dir="results4/IGA vs omniscient")
# EMA_experiment_data = load_experiment_data(target_dir="results4/IGA vs EMA")
# Bayesian_experiment_data = load_experiment_data(
#     target_dir="results4/IGA vs Bayesian")
# Bayesian_ultra_experiment_data = load_experiment_data(
#     target_dir="results4/IGA vs Bayesian ultra")

# # %%


# ################ IGA VS Omniscient, EMA, Bayesian  ###############
# plot_average_reward_hyperq_vs_other([omniscient_experiment_data,
#                                     EMA_experiment_data,
#                                     Bayesian_experiment_data,
#                                     Bayesian_ultra_experiment_data
#                                      ],
#                                     title="Hyper-Q vs. IGA: Avg. reward per time step",
#                                     agent_names=["Omniscient",
#                                                  "EMA", "Bayesian", "Bayesian ultra"],
#                                     ma_window_sizes=[5000, 5000, 5000, 5000],
#                                     symmetrical_reward=True,
#                                     agent_idx=3,
#                                     steps=120_000,
#                                     file_name="HyperQ vs IGA Avg reward per time step")

# # %%

# omniscient_experiment_data = load_experiment_data(
#     target_dir="results4/PHC vs omniscient")
# EMA_experiment_data = load_experiment_data(target_dir="results4/PHC vs EMA")
# Bayesian_experiment_data = load_experiment_data(
#     target_dir="results4/PHC vs Bayesian")
# Bayesian_ultra_experiment_data = load_experiment_data(
#     target_dir="results4/PHC vs Bayesian ultra")

# # %%

# plot_average_reward_hyperq_vs_other([
#                                     omniscient_experiment_data,
#                                     EMA_experiment_data,
#                                     Bayesian_experiment_data,
#                                     Bayesian_ultra_experiment_data],
#                                     title="Hyper-Q vs. PHC: Avg. reward per time step",
#                                     agent_names=["Omniscient",
#                                                  "EMA", "Bayesian", "Bayesian ultra"],
#                                     ma_window_sizes=[5000, 5000, 5000, 5000],
#                                     symmetrical_reward=True,
#                                     agent_idx=REWARDS_PLAYER_Y,
#                                     steps=120_000,
#                                     file_name="HyperQ vs PHC Avg reward per time step")


# # %%


# ################ PHC, IGA, Omniscient, EMA, Bayesian vs monotone ###############
# experiment_vs_monotone1 = load_experiment_data(
#     target_dir="results4/PHC vs monotone")
# experiment_vs_monotone2 = load_experiment_data(
#     target_dir="results4/IGA vs monotone")
# experiment_vs_monotone3 = load_experiment_data(
#     target_dir="results4/Omniscient vs monotone")
# experiment_vs_monotone4 = load_experiment_data(
#     target_dir="results4/EMA vs monotone")
# experiment_vs_monotone5 = load_experiment_data(
#     target_dir="results4/Bayesian vs monotone")
# experiment_vs_monotone6 = load_experiment_data(
#     target_dir="results4/Bayesian ultra vs monotone")

# # %%

# plot_average_reward_hyperq_vs_other([
#                                     experiment_vs_monotone1,
#                                     experiment_vs_monotone2,
#                                     experiment_vs_monotone3,
#                                     experiment_vs_monotone4,
#                                     experiment_vs_monotone5,
#                                     experiment_vs_monotone6
#                                     ],
#                                     title="PHC, IGA, Omniscient, EMA, Bayesian vs monotone: Avg. reward per time step",
#                                     agent_names=[
#                                         "PHC",
#                                         "IGA",
#                                         "Omniscient",
#                                         "EMA",
#                                         "Bayesian", "Bayesian ultra"],
#                                     ma_window_sizes=[
#                                         5000,
#                                         5000,
#                                         5000,
#                                         5000,
#                                         5000,
#                                         5000],
#                                     symmetrical_reward=True,
#                                     steps=120_000,
#                                     agent_idx=REWARDS_PLAYER_X,
#                                     file_name="HyperQ, IGA, PHC vs monotone")

# # %%


# # %%
# # **********************************************
# # ************** COOPERATION GAME # ************
# # **********************************************


# # %% Rewards

# omniscient_experiment_data = load_experiment_data(
#     target_dir="results4/cooperation/Omniscient vs Omniscient")
# EMA_experiment_data = load_experiment_data(
#     target_dir="results4/cooperation/EMA vs EMA")
# Bayesian_experiment_data = load_experiment_data(
#     target_dir="results4/cooperation/Bayesian vs Bayesian")
# Bayesian_ultra_experiment_data = load_experiment_data(
#     target_dir="results4/cooperation/Bayesian ultra vs Bayesian ultra")


# # %%
# plot_average_reward_hyperq_vs_other([
#                                     omniscient_experiment_data,
#                                     EMA_experiment_data,
#                                     Bayesian_experiment_data,
#                                     Bayesian_ultra_experiment_data],
#                                     title="Hyper-Q cooperation game: Avg. reward per time step",
#                                     agent_names=["Omniscient", "EMA",
#                                                  "Bayesian", "Bayesian ultra"],
#                                     ma_window_sizes=[5000, 5000, 5000, 5000],
#                                     symmetrical_reward=True,
#                                     agent_idx=REWARDS_PLAYER_X,
#                                     steps=60_000,
#                                     file_name="HyperQ cooperation Avg reward per time step")

# # %%
# ################# OMNISCIENT ####################
# agent1_experiment1_strategies_over_time = omniscient_experiment_data[:,
#                                                                      :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
# agent2_experiment1_strategies_over_time = omniscient_experiment_data[:,
#                                                                      :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]


# plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Omniscient agent1 strategy evolution",
#                                      file_name="coop omniscient vs omniscient agent1 actions", steps=60_000)
# plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Omniscient agent2 strategy evolution",
#                                      file_name="coop omniscient vs omniscient agent2 actions", steps=60_000)

# ################# EMA ####################
# agent1_experiment1_strategies_over_time = EMA_experiment_data[:,
#                                                               :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
# agent2_experiment1_strategies_over_time = EMA_experiment_data[:,
#                                                               :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

# plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time,
#                                      title="Cooperation game: EMA agent1 strategy evolution", steps=60_000, file_name="coop EMA vs EMA agent1 actions")
# plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time,
#                                      title="Cooperation game: EMA agent2 strategy evolution", file_name="coop EMA vs EMA agent2 actions", steps=60_000)


# ################# Bayesian ##################
# agent1_experiment1_strategies_over_time = Bayesian_experiment_data[:,
#                                                                    :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
# agent2_experiment1_strategies_over_time = Bayesian_experiment_data[:,
#                                                                    :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

# plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Bayesian agent1 strategy evolution",
#                                      steps=60_000, file_name="coop Bayesian vs Bayesian agent1 actions")
# plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Bayesian agent2 strategy evolution",
#                                      file_name="coop Bayesian vs Bayesian agent2 actions", steps=60_000)


# ################# Bayesian ##################
# agent1_experiment1_strategies_over_time = Bayesian_ultra_experiment_data[
#     :, :, ACTIONS_PLAYER_X_STARTING_IDX:ACTIONS_PLAYER_X_STARTING_IDX+3]
# agent2_experiment1_strategies_over_time = Bayesian_ultra_experiment_data[
#     :, :, ACTIONS_PLAYER_Y_STARTING_IDX:ACTIONS_PLAYER_Y_STARTING_IDX+3]

# plot_strategy_over_time_single_agent(agent1_experiment1_strategies_over_time, title="Cooperation game: Bayesian ultra agent1 strategy evolution",
#                                      steps=60_000, file_name="coop Bayesian ultra vs Bayesian ultra agent1 actions")
# plot_strategy_over_time_single_agent(agent2_experiment1_strategies_over_time, title="Cooperation game: Bayesian ultra agent2 strategy evolution",
#                                      file_name="coop Bayesian ultra vs Bayesian ultra agent2 actions", steps=60_000)

# # %%

