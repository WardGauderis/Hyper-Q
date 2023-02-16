import itertools
import json
import os
import subprocess

# Should be multiple of #CPU cores
runs = 20
iterations = 1_500_000

# game_domains = {
#     "game": ["rock_paper_scissors"],
#     "exploration": ["epsilon_greedy"],
#     "epsilon": [None, 0.001, 0.01],  # only epsilon_greedy
#     "every": [None],  # only random_restart
# }

# agent_domains = {
#     "type": ["bayesian_ultra_q", "iga", "phc"],
#     "strategy": [None, [1 / 2, 1 / 2, 0]],  # only monotone
#     "mu": [None, 0.005],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
#     "alpha": [None, 0.01, 0.05],  # not iga or monotone
#     "gamma": [None, 0.9],  # not iga or monotone
#     "similarity": [None, "likelihood", "likelihood_scaled", "cosine"],  # only ultra_q
#     "step_size": [None, 0.01],  # only iga
#     "delta": [None, 0.01],  # only phc
#     "init": [None, "random", 0, 5]
# }

game_domains = {
    "game": ["hill_climbing", "rock_paper_scissors"],
    "exploration": ["epsilon_greedy"],
    "epsilon": [None, 0.01],  # only epsilon_greedy
    "every": [None],  # only random_restart
}

agent_domains = {
    "type": ["ema_hyper_q", "monotone", "bayesian_hyper_q", "omniscient_hyper_q", "iga", "phc", "bayesian_ultra_q"],
    "strategy": [None, [1 / 2, 1 / 2, 0]],  # only monotone
    "mu": [None, 0.005],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
    "alpha": [None, 0.01],  # not iga or monotone
    "gamma": [None, 0.9],  # not iga or monotone
    "similarity": [None, "likelihood_scaled"], # only ultra_q
    "step_size": [None, 0.01],  # only iga
    "delta": [None, 0.01],  # only phc
    "init": [None, "random"]
}

# We hebben al:
# rock paper scissors:
#   Ultra vs IGA
#   Ultra vs PHC
# hill climbing:
#   ultra vs ultra




def remove_none(d):
    """
    Remove all keys with value None
    :param d:
    :return:
    """
    if isinstance(d, dict):
        return {k: remove_none(v) for k, v in d.items() if v is not None}
    else:
        return d


def game_configs(domains):
    """
    Generate all possible game configurations
    :param domains:
    :return:
    """
    for combination in itertools.product(*domains.values()):
        d = dict(zip(domains.keys(), combination))

        greedy = d["exploration"] == "epsilon_greedy"
        epsilon = d["epsilon"] is not None
        if greedy ^ epsilon:
            continue

        random_restart = d["exploration"] == "random_restart"
        every = d["every"] is not None
        if random_restart ^ every:
            continue

        d["exploration"] = {
            "type": d["exploration"],
            "epsilon": d["epsilon"],
            "every": d["every"],
        }
        del d["epsilon"]
        del d["every"]

        yield remove_none(d)


def agent_configs(domains):
    """
    Generate all possible agent configurations
    :param domains:
    :return:
    """
    for combination in itertools.product(*domains.values()):
        d = dict(zip(domains.keys(), combination))

        strategy = d["strategy"] is not None
        monotone = d["type"] == "monotone"
        if strategy ^ monotone:
            continue

        mu = d["mu"] is not None
        estimator = d["type"] in ["ema_hyper_q", "bayesian_hyper_q", "bayesian_ultra_q"]
        if mu ^ estimator:
            continue

        alpha = d["alpha"] is not None
        not_iga_or_monotone = d["type"] not in ["iga", "monotone"]
        if alpha ^ not_iga_or_monotone:
            continue

        gamma = d["gamma"] is not None
        if gamma ^ not_iga_or_monotone:
            continue

        similarity = d["similarity"] is not None
        ultra_q = d["type"] == "bayesian_ultra_q"
        if similarity ^ ultra_q:
            continue

        step_size = d["step_size"] is not None
        iga = d["type"] == "iga"
        if step_size ^ iga:
            continue

        delta = d["delta"] is not None
        phc = d["type"] == "phc"
        if delta ^ phc:
            continue

        init = d["init"] is not None
        if init ^ not_iga_or_monotone:
            continue

        yield remove_none(d)


game_configs = list(game_configs(game_domains))
agent_configs = list(agent_configs(agent_domains))

print(f"game configs: {len(game_configs)}")
print(f"agent configs: {len(agent_configs)}")


# print(game_configs)
# print(agent_configs)
# [{'game': 'hill_climbing', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}}, 
#  {'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}}]

# [{'type': 'ema_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#  {'type': 'monotone', 'strategy': [0.5, 0.5, 0]}, 
#  {'type': 'bayesian_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#  {'type': 'omniscient_hyper_q', 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#  {'type': 'iga', 'step_size': 0.01}, 
#  {'type': 'phc', 'alpha': 0.01, 'gamma': 0.9, 'delta': 0.01, 'init': 'random'}, 
#  {'type': 'bayesian_ultra_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'similarity': 'likelihood_scaled', 'init': 'random'}]

# Wat willen we:
# rock
#   Omniscient vs IGA, PHC
#   EMA vs IGA, PHC
#   Bayesian vs IGA, PHC

# hill:
#   Omniscient vs Omniscient
#   EMA vs EMA
#   Bayesian vs Bayesian

def test_configs(game_configs, agent_configs):
    """
    Test all possible configurations that are interesting
    :param game_configs:
    :param agent_configs:
    :return:
    """
    for game_config in game_configs:
        for agent_config_x in agent_configs:
            for agent_config_y in agent_configs:

                if( 
                    # hill
                    (game_config["game"] == "hill_climbing" and agent_config_x["type"] == "omniscient_hyper_q" and agent_config_y["type"] == "omniscient_hyper_q") or
                    (game_config["game"] == "hill_climbing" and agent_config_x["type"] == "ema_hyper_q" and agent_config_y["type"] == "ema_hyper_q") or
                    (game_config["game"] == "hill_climbing" and agent_config_x["type"] == "bayesian_hyper_q" and agent_config_y["type"] == "bayesian_hyper_q") or
                    
                    # rock
                    (game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == "omniscient_hyper_q" and (agent_config_y["type"] == "iga" or agent_config_y["type"] == "phc")) or
                    (game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == "ema_hyper_q" and (agent_config_y["type"] == "iga" or agent_config_y["type"] == "phc")) or
                    (game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == "bayesian_hyper_q" and (agent_config_y["type"] == "iga" or agent_config_y["type"] == "phc"))
                     
                ):
                    pass
                else:
                    continue

                # Only do hill climbing with agents of the same type
                # if game_config["game"] == "hill_climbing" and agent_config_x["type"] != agent_config_y["type"]:
                #     continue

                # Only do rock paper scissors with different agents
                # if game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == agent_config_y["type"]:
                #     continue

                # Don't let test agents play against themselves
                # test_agents = ["monotone", "iga", "phc"]
                # if agent_config_x["type"] in test_agents and agent_config_y["type"] in test_agents:
                #     continue

                # Only play agents opponent with same parameters
                temp_x = {k for k, v in agent_config_x.items() if k not in ["type"]}
                temp_y = {k for k, v in agent_config_y.items() if k not in ["type"]}
                intersection = temp_x.intersection(temp_y)
                if any(agent_config_x[k] != agent_config_y[k] for k in intersection):
                    continue

                # Only Bayesian ultra
                # if agent_config_x["type"] != "bayesian_ultra_q" and agent_config_y["type"] != "bayesian_ultra_q":
                #     continue

                # if sorted(agent_config_x) < sorted(agent_config_y):
                #     continue

                d = {
                    "runs": runs,
                    "iterations": iterations,
                    **game_config,
                    "agent_x": agent_config_x,
                    "agent_y": agent_config_y,
                }
                yield d


configs = list(test_configs(game_configs, agent_configs))
print(f"configs: {len(configs)}")

print(configs)


# [
#     {'runs': 20, 'iterations': 1500000, 'game': 'hill_climbing', 
#      'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'ema_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'ema_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}}, 

#      {'runs': 20, 'iterations': 1500000, 'game': 'hill_climbing', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'bayesian_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'bayesian_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}}, 
     
#      {'runs': 20, 'iterations': 1500000, 'game': 'hill_climbing', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'omniscient_hyper_q', 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'omniscient_hyper_q', 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}}, 
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'ema_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'iga', 'step_size': 0.01}}, 
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'ema_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'phc', 'alpha': 0.01, 'gamma': 0.9, 'delta': 0.01, 'init': 'random'}}, 
     
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'bayesian_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'iga', 'step_size': 0.01}}, 
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'bayesian_hyper_q', 'mu': 0.005, 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'phc', 'alpha': 0.01, 'gamma': 0.9, 'delta': 0.01, 'init': 'random'}}, 
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'omniscient_hyper_q', 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'iga', 'step_size': 0.01}}, 
#      {'runs': 20, 'iterations': 1500000, 'game': 'rock_paper_scissors', 'exploration': {'type': 'epsilon_greedy', 'epsilon': 0.01}, 
#      'agent_x': {'type': 'omniscient_hyper_q', 'alpha': 0.01, 'gamma': 0.9, 'init': 'random'}, 
#      'agent_y': {'type': 'phc', 'alpha': 0.01, 'gamma': 0.9, 'delta': 0.01, 'init': 'random'}}]

# input("Press enter to start...")

subprocess.check_output(["cmake", "-DCMAKE_BUILD_TYPE=Release", "."])
subprocess.check_output(["make", "-j"])

counters = {}
for i, config in enumerate(configs):
    types = [config["agent_x"]["type"], config["agent_y"]["type"]]
    types.sort()
    directory = f"results/{config['game']}/{types[0]}/{types[1]}"

    counters[directory] = counters.get(directory, 0) + 1
    directory = directory + f"/{counters[directory]}"
    filename = "config.json"

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(config, f, indent=4)

    top = os.getcwd()
    os.chdir(directory)

    print(f"Running {i + 1}/{len(configs)}")
    print(f"Directory: {directory}")

    subprocess.check_output([f"{top}/Hyper_Q"])

    os.chdir(top)


# # try:
# import itertools
# import json
# import os
# import subprocess

# # Should be multiple of #CPU cores
# runs = 20
# iterations = 1_500_000


# # {
# #     "runs": 20,
# #     "iterations": 1500000,
# #     "game": "rock_paper_scissors",
# #     "exploration": {
# #         "type": "epsilon_greedy",
# #         "epsilon": 0.01
# #     },
# #     "agent_x": {
# #         "type": "bayesian_ultra_q",
# #         "mu": 0.005,
# #         "alpha": 0.01,
# #         "gamma": 0.9,
# #         "similarity": "likelihood_scaled",
# #         "init": "random"
# #     },
# #     "agent_y": {
# #         "type": "phc",
# #         "alpha": 0.01,
# #         "gamma": 0.9,
# #         "delta": 0.01,
# #         "init": "random"
# #     }
# # }


# game_domains = {
#     "game": ["hill_climbing", "rock_paper_scissors"],
#     "exploration": ["epsilon_greedy"],
#     "epsilon": [None, 0.01],  # only epsilon_greedy
#     "every": [None],  # only random_restart
# }

# agent_domains = {
#     "type": ["ema_hyper_q", "monotone", "bayesian_hyper_q", "omniscient_hyper_q" "iga", "phc"],
#     "strategy": [None, [1 / 2, 1 / 2, 0]],  # only monotone
#     "mu": [None, 0.005],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
#     "alpha": [None, 0.01],  # not iga or monotone
#     "gamma": [None, 0.9],  # not iga or monotone
#     "similarity": [None, "likelihood_scaled"], # only ultra_q
#     "step_size": [None, 0.01],  # only iga
#     "delta": [None, 0.01],  # only phc
#     "init": [None, "random"]
# }

# # "mu": [None, 0.005, 0.01],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
# # "every": [None, 1000, 5000],  # only random_restart

# # runs = 32
# # iterations = 2_000_000
# #
# # game_domains = {
# #     "game": ["rock_paper_scissors", "hill_climbing"],
# #     "exploration": ["random_restart", "epsilon_greedy", "epsilon_greedy_decay"],
# #     "epsilon": [None, 0.0001, 0.001, 0.01, 0.1],  # only epsilon_greedy
# #     "every": [None, 100, 1000, 10000],  # only random_restart
# # }
# #
# # agent_domains = {
# #     "type": ["monotone", "omniscient_hyper_q", "ema_hyper_q", "bayesian_hyper_q", "bayesian_ultra_q", "iga", "phc"],
# #     "strategy": [None, [1, 0, 0], [1 / 2, 1 / 2, 0], [1 / 3, 1 / 3, 1 / 3]],  # only monotone
# #     "mu": [None, 0.001, 0.005, 0.01],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
# #     "alpha": [None, 0.005, 0.01, 0.05, 0.1],  # not iga or monotone
# #     "gamma": [None, 0.8, 0.9, 0.99],  # not iga or monotone
# #     "similarity": [None, "likelihood", "likelihood_scaled", "cosine"],  # only ultra_q
# #     "step_size": [None, 0.0001, 0.001, 0.01, 0.1],  # only iga
# #     "delta": [None, 0.001, 0.01, 0.1],  # only phc
# #     "init": [None, "random", -1, 0, 1, 10]
# #     # not iga or monotone
# # }

# def remove_none(d):
#     """
#     Remove all keys with value None
#     :param d:
#     :return:
#     """
#     if isinstance(d, dict):
#         return {k: remove_none(v) for k, v in d.items() if v is not None}
#     else:
#         return d


# def game_configs(domains):
#     """
#     Generate all possible game configurations
#     :param domains:
#     :return:
#     """
#     for combination in itertools.product(*domains.values()):
#         d = dict(zip(domains.keys(), combination))

#         greedy = d["exploration"] == "epsilon_greedy"
#         epsilon = d["epsilon"] is not None
#         if greedy ^ epsilon:
#             continue

#         random_restart = d["exploration"] == "random_restart"
#         every = d["every"] is not None
#         if random_restart ^ every:
#             continue

#         d["exploration"] = {
#             "type": d["exploration"],
#             "epsilon": d["epsilon"],
#             "every": d["every"],
#         }
#         del d["epsilon"]
#         del d["every"]

#         yield remove_none(d)


# def agent_configs(domains):
#     """
#     Generate all possible agent configurations
#     :param domains:
#     :return:
#     """
#     for combination in itertools.product(*domains.values()):
#         d = dict(zip(domains.keys(), combination))

#         strategy = d["strategy"] is not None
#         monotone = d["type"] == "monotone"
#         if strategy ^ monotone:
#             continue

#         mu = d["mu"] is not None
#         estimator = d["type"] in ["ema_hyper_q", "bayesian_hyper_q", "bayesian_ultra_q"]
#         if mu ^ estimator:
#             continue

#         alpha = d["alpha"] is not None
#         not_iga_or_monotone = d["type"] not in ["iga", "monotone"]
#         if alpha ^ not_iga_or_monotone:
#             continue

#         gamma = d["gamma"] is not None
#         if gamma ^ not_iga_or_monotone:
#             continue

#         similarity = d["similarity"] is not None
#         ultra_q = d["type"] == "bayesian_ultra_q"
#         if similarity ^ ultra_q:
#             continue

#         step_size = d["step_size"] is not None
#         iga = d["type"] == "iga"
#         if step_size ^ iga:
#             continue

#         delta = d["delta"] is not None
#         phc = d["type"] == "phc"
#         if delta ^ phc:
#             continue

#         init = d["init"] is not None
#         if init ^ not_iga_or_monotone:
#             continue

#         yield remove_none(d)


# game_configs = list(game_configs(game_domains))
# agent_configs = list(agent_configs(agent_domains))

# print(f"game configs: {len(game_configs)}")
# print(f"agent configs: {len(agent_configs)}")


# def test_configs(game_configs, agent_configs):
#     """
#     Test all possible configurations that are interesting
#     :param game_configs:
#     :param agent_configs:
#     :return:
#     """
#     for game_config in game_configs:
#         for agent_config_x in agent_configs:
#             for agent_config_y in agent_configs:

#                 # Only do hill climbing with agents of the same type
#                 if game_config["game"] == "hill_climbing" and agent_config_x["type"] != agent_config_y["type"]:
#                     continue

#                 # Only do rock paper scissors with different agents
#                 if game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == agent_config_y["type"]:
#                     continue

#                 # Don't let test agents play against themselves
#                 test_agents = ["monotone", "iga", "phc"]
#                 if agent_config_x["type"] in test_agents and agent_config_y["type"] in test_agents:
#                     continue

#                 # Only play agents opponent with same parameters
#                 temp_x = {k for k, v in agent_config_x.items() if k not in ["type"]}
#                 temp_y = {k for k, v in agent_config_y.items() if k not in ["type"]}
#                 intersection = temp_x.intersection(temp_y)
#                 if any(agent_config_x[k] != agent_config_y[k] for k in intersection):
#                     continue

#                 # Only Bayesian ultra
#                 if agent_config_x["type"] != "bayesian_ultra_q" and agent_config_y["type"] != "bayesian_ultra_q":
#                     continue

#                 if sorted(agent_config_x) < sorted(agent_config_y):
#                     continue

#                 d = {
#                     "runs": runs,
#                     "iterations": iterations,
#                     **game_config,
#                     "agent_x": agent_config_x,
#                     "agent_y": agent_config_y,
#                 }
#                 yield d


# configs = list(test_configs(game_configs, agent_configs))
# print(f"configs: {len(configs)}")

# # input("Press enter to start...")
# # C:\Program Files\CMake\
# # subprocess.check_output(["cmake", "-DCMAKE_BUILD_TYPE=Release", "."])
# # subprocess.check_output(["make", "-j"])

# subprocess.check_output([r"C:\Program Files\CMake\bin\cmake.exe", "-DCMAKE_BUILD_TYPE=Release", "."])
# subprocess.check_output([r"C:\Program Files\CMake\bin\cmake.exe", "--build", ".", "--config", "Release"])


# counters = {}
# for i, config in enumerate(configs):
#     types = [config["agent_x"]["type"], config["agent_y"]["type"]]
#     types.sort()
#     directory = f"results/{config['game']}/{types[0]}/{types[1]}"

#     counters[directory] = counters.get(directory, 0) + 1
#     directory = directory + f"/{counters[directory]}"
#     filename = "config.json"

#     os.makedirs(directory, exist_ok=True)
#     with open(os.path.join(directory, filename), "w") as f:
#         json.dump(config, f, indent=4)

#     top = os.getcwd()
#     os.chdir(directory)

#     print(f"Running {i + 1}/{len(configs)}")
#     print(f"Directory: {directory}")

#     subprocess.check_output([f"{top}/Hyper_Q"])

#     os.chdir(top)
# # except Exception as e:
# #     # save exception message to temp.txt
# #     with open("temp.txt", "w") as f:
# #         f.write(str(e))
        

# #     print(e)