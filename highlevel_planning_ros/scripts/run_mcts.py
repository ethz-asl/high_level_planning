import numpy as np
import os
import ast
from datetime import datetime
import networkx as nx
import matplotlib as mpl

# Simulation
# from highlevel_planning_py.sim.scene_planning_1 import ScenePlanning1
from highlevel_planning_py.sim.scene_planning_2 import ScenePlanning2

# Skills
from highlevel_planning_py.skills.navigate import SkillNavigate
from highlevel_planning_py.skills.grasping import SkillGrasping
from highlevel_planning_py.skills.placing import SkillPlacing
from highlevel_planning_py.execution.es_sequential_execution import SequentialExecution

# Learning
from highlevel_planning_py.exploration.explorer import Explorer
from highlevel_planning_py.exploration.pddl_extender import PDDLExtender
from highlevel_planning_py.exploration import mcts
from highlevel_planning_py.exploration.logic_tools import determine_relevant_predicates
from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase

# Other
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools import run_util, util, util_mcts
from highlevel_planning_py.tools import util_pybullet

from highlevel_planning_py.exploration.exploration_tools import get_items_closeby

# mpl.use("TkAgg")

# ----------------------------------------------------------------------

SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "": "",
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "src_root_dir": SRCROOT,
    "asset_dir": os.path.join(SRCROOT, "models"),
    "bin_dir": os.path.join(SRCROOT, "bin"),
}


def main():
    # Command line arguments
    args = util.parse_arguments()

    # Seed RNGs
    if not args.no_seed:
        print("Seeding RNG")
        np.random.seed(0)

    if args.method == "direct" and args.reuse_objects:
        raise RuntimeError("Cannot reload objects when in direct mode.")

    # Load existing simulation data if desired
    savedir = os.path.join(PATHS["data_dir"], "simulator")
    objects, robot_mdl = util_pybullet.restore_pybullet_sim(savedir, args)

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
    scene, world = util_pybullet.setup_pybullet_world(
        ScenePlanning2, PATHS, args, savedir, objects
    )
    robot = run_util.setup_robot(world, cfg, PATHS["asset_dir"], robot_mdl)

    # Save state
    util_pybullet.save_pybullet_sim(args, savedir, scene, robot)

    # -----------------------------------

    # Set goal
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

    get_execution_system = lambda seq, params: SequentialExecution(
        skill_set, seq, params, kb
    )

    # Set up exploration
    xplorer = Explorer(
        skill_set, robot, scene.objects, pddl_ex, kb, cfg, world, get_execution_system
    )
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
    mcts_state = mcts.HLPState(True, 0, xplorer, relevant_predicates, max_depth)
    mcts_root_node = mcts.HLPTreeNode(
        mcts_state,
        action_list,
        graph,
        avoid_double_nav=cfg.getparam(["mcts", "avoid_double_nav"]),
        relevant_objects=relevant_objects,
        explorer=xplorer,
        virtual_objects=["origin", "grasp0", "grasp1"],
        pybullet_domain=True,
    )
    mcts_search = mcts.HLPTreeSearch(mcts_root_node, xplorer, cfg)

    # ---------------------------------------------------------------

    metrics = mcts_search.tree_search()
    metrics["goals"] = kb.goals

    kb_clone = KnowledgeBase(PATHS, domain_name=scene.__class__.__name__)
    kb_clone.duplicate(kb)
    util_mcts.mcts_exit_handler(
        mcts_root_node, time_string, cfg, metrics, kb_clone, PATHS
    )


if __name__ == "__main__":
    main()
