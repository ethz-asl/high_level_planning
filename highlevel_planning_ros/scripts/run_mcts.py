import numpy as np
import os
import json
import pickle
import ast
from datetime import datetime
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

# Simulation
# from highlevel_planning_py.sim.scene_planning_1 import ScenePlanning1
from highlevel_planning_py.sim.scene_planning_2 import ScenePlanning2

# Skills
from highlevel_planning_py.skills.navigate import SkillNavigate
from highlevel_planning_py.skills.grasping import SkillGrasping
from highlevel_planning_py.skills.placing import SkillPlacing

# Learning
from highlevel_planning_py.exploration.explorer import Explorer
from highlevel_planning_py.exploration.pddl_extender import PDDLExtender
from highlevel_planning_py.exploration import mcts
from highlevel_planning_py.exploration.logic_tools import determine_relevant_predicates
from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase

# Other
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools import run_util

from highlevel_planning_py.exploration.exploration_tools import get_items_closeby

# mpl.use("TkAgg")

# ----------------------------------------------------------------------

SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "src_root_dir": SRCROOT,
    "asset_dir": os.path.join(SRCROOT, "models"),
    "bin_dir": os.path.join(SRCROOT, "bin"),
}


def mcts_exit_handler(node, time_string, config, metrics, knowledge_base):
    savedir = os.path.join(PATHS["data_dir"], "mcts")
    os.makedirs(savedir, exist_ok=True)

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


def main():
    # Command line arguments
    args = run_util.parse_arguments()

    # Seed RNGs
    if not args.no_seed:
        print("Seeding RNG")
        np.random.seed(0)

    if args.method == "direct" and args.reuse_objects:
        raise RuntimeError("Cannot reload objects when in direct mode.")

    # Load existing simulation data if desired
    savedir = os.path.join(PATHS["data_dir"], "simulator")
    objects, robot_mdl = run_util.restore_pybullet_sim(savedir, args)

    # Load config file
    if len(args.config_file_path) == 0:
        config_file_path = os.path.join(SRCROOT, "config", "main.yaml")
    else:
        config_file_path = args.config_file_path
    cfg = ConfigYaml(config_file_path)

    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")
    # rep = Reporter(DATADIR, cfg, time_string, args.noninteractive)
    # atexit.register(exit_handler, rep)

    # Populate simulation
    scene, world = run_util.setup_pybullet_world(
        ScenePlanning2, PATHS["asset_dir"], args, savedir, objects
    )
    robot = run_util.setup_robot(world, cfg, PATHS["asset_dir"], robot_mdl)

    # Save state
    run_util.save_pybullet_sim(args, savedir, scene, robot)

    # -----------------------------------

    # Set goal
    # goals = [("in-hand", True, ("duck", "robot1"))]
    # goals = [("at", True, ("container1", "robot1"))]
    # goals = [("at", True, ("cupboard", "robot1"))]
    # goals = [("on", True, ("cupboard", "cube1"))]
    # goals = [("on", True, ("cupboard", "duck"))]
    # goals = [
    #     ("on", True, ("cupboard", "cube1")),
    #     ("at", True, ("container1", "robot1")),
    # ]
    # goals = [("on", True, ("container2", "cube1"))]
    # goals = [("on", True, ("container2", "lego"))]
    # goals = [("inside", True, ("container2", "cube1"))]
    # goals = [("inside", True, ("container1", "cube1"))]
    # goals = [("inside", True, ("container1", "lego"))]
    # goals = [("inside", True, ("container1", "duck"))]
    # goals = [("inside", True, ("shelf", "tall_box"))]
    # goals = [("inside", True, ("container2", "cube2"))]
    goals = ast.literal_eval(cfg.getparam(["user_input", "goals"]))

    # -----------------------------------

    kb, preds = run_util.setup_knowledge_base(
        PATHS, scene, robot, cfg, time_string, goals, world.client_id
    )

    # Set up skills
    sk_grasp = SkillGrasping(scene, robot, cfg)
    sk_place = SkillPlacing(scene, robot, cfg)
    sk_nav = SkillNavigate(scene, robot)
    skill_set = {"grasp": sk_grasp, "nav": sk_nav, "place": sk_place}

    # PDDL extender
    pddl_ex = PDDLExtender(kb, preds)

    # Set up exploration
    xplorer = Explorer(skill_set, robot, scene.objects, pddl_ex, kb, cfg, world)
    goal_objects = xplorer._get_items_goal()
    closeby_objects = get_items_closeby(
        goal_objects,
        scene.objects,
        world.client_id,
        robot.model.uid,
        distance_limit=cfg.getparam(["mcts", "closeby_objects_distance_threshold"]),
    )
    relevant_objects = goal_objects + closeby_objects
    action_list = [act for act in kb.actions if act not in kb.meta_actions]
    relevant_predicates = determine_relevant_predicates(relevant_objects, kb)

    # Set up MCTS
    graph = nx.DiGraph()
    max_depth = cfg.getparam(["mcts", "max_depth"], default_value=10)
    mcts_state = mcts.HLPState(
        True, 0, world.client_id, xplorer, relevant_predicates, max_depth
    )
    mcts_root_node = mcts.HLPTreeNode(
        mcts_state,
        action_list,
        graph,
        relevant_objects=relevant_objects,
        explorer=xplorer,
    )
    mcts_search = mcts.HLPTreeSearch(mcts_root_node, xplorer, cfg)

    # ---------------------------------------------------------------

    metrics = mcts_search.tree_search()
    metrics["goals"] = kb.goals

    kb_clone = KnowledgeBase(PATHS, domain_name=scene.__class__.__name__)
    kb_clone.duplicate(kb)
    mcts_exit_handler(mcts_root_node, time_string, cfg, metrics, kb_clone)


if __name__ == "__main__":
    main()
