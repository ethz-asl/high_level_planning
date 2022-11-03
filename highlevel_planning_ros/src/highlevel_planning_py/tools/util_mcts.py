import os
import json
import pickle
import matplotlib.pyplot as plt

from highlevel_planning_py.exploration import mcts


def mcts_exit_handler(
    node,
    time_string,
    config,
    metrics,
    knowledge_base,
    paths: dict,
    plot_graph: bool = True,
):
    savedir = os.path.join(paths["data_dir"], "mcts")
    os.makedirs(savedir, exist_ok=True)

    if plot_graph:
        figure, ax = plt.subplots()
        mcts.plot_graph(node.graph, node, figure, ax, explorer=None)
        filename = "{}_mcts_tree.png".format(time_string)
        figure.savefig(os.path.join(savedir, filename))

    data = dict()
    data["tree"] = node
    data["config"] = config._cfg
    data["metrics"] = metrics
    data["knowledge_base"] = knowledge_base

    filename = "{}_data.pkl".format(time_string)
    with open(os.path.join(savedir, filename), "wb") as f:
        pickle.dump(data, f)

    filename = "{}_metrics.txt".format(time_string)
    with open(os.path.join(savedir, filename), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key:42}: {value}\n")

    filename = "{}_config.txt".format(time_string)
    with open(os.path.join(savedir, filename), "w") as f:
        json.dump(config._cfg, f)

    print(f"Saved everything. Time string: {time_string}")
