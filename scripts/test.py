import itertools
import json
import os
import subprocess

# Should be multiple of #CPU cores
runs = 32
iterations = 2_000_000

game_domains = {
    "game": ["rock_paper_scissors", "hill_climbing"],
    "exploration": ["random_restart", "epsilon_greedy", "epsilon_greedy_decay"],
    "epsilon": [None, 0.0001, 0.001, 0.01, 0.1],  # only epsilon_greedy
    "every": [None, 100, 1000, 10000],  # only random_restart
}

agent_domains = {
    "type": ["monotone", "omniscient_hyper_q", "ema_hyper_q", "bayesian_hyper_q", "bayesian_ultra_q", "iga", "phc"],
    "strategy": [None, [1, 0, 0], [1 / 2, 1 / 2, 0], [1 / 3, 1 / 3, 1 / 3]],  # only monotone
    "mu": [None, 0.001, 0.005, 0.01],  # only ema_hyper_q, bayesian_hyper_q or bayesian_ultra_q
    "alpha": [None, 0.005, 0.01, 0.05, 0.1],  # not iga or monotone
    "gamma": [None, 0.8, 0.9, 0.99],  # not iga or monotone
    "similarity": [None, "likelihood", "likelihood_scaled", "cosine"],  # only ultra_q
    "step_size": [None, 0.0001, 0.001, 0.01, 0.1],  # only iga
    "delta": [None, 0.001, 0.01, 0.1],  # only phc
    "init": [None, "random", -1, 0, 1, 10]
    # not iga or monotone
}


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

                # Only do hill climbing with agents of the same type
                if game_config["game"] == "hill_climbing" and agent_config_x["type"] != agent_config_y["type"]:
                    continue

                # Only do rock paper scissors with different agents
                if game_config["game"] == "rock_paper_scissors" and agent_config_x["type"] == agent_config_y["type"]:
                    continue

                # Don't let test agents play against themselves
                test_agents = ["monotone", "iga", "phc"]
                if agent_config_x["type"] in test_agents and agent_config_y["type"] in test_agents:
                    continue

                # Only play agents opponent with same parameters
                temp_x = {k for k, v in agent_config_x.items() if k not in ["type"]}
                temp_y = {k for k, v in agent_config_y.items() if k not in ["type"]}
                intersection = temp_x.intersection(temp_y)
                if any(agent_config_x[k] != agent_config_y[k] for k in intersection):
                    continue

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

input("Press enter to start...")

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

    subprocess.run([f"{top}/Hyper_Q"])

    os.chdir(top)
