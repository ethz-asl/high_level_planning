import numpy as np
import os
import ast
from datetime import datetime
import networkx as nx
import matplotlib as mpl

# Simulation
from highlevel_planning_py.pddl_interface.custom_pddl_parser import CustomPDDLParser
from highlevel_planning_py.sim_pddl.explorer import PDDLExplorer
from highlevel_planning_py.sim_pddl.sequential_execution import SequentialExecutionPDDL
from highlevel_planning_py.sim_pddl.world import PDDLSimWorld

# Learning
from highlevel_planning_py.exploration.pddl_extender import PDDLExtender
from highlevel_planning_py.exploration import mcts
from highlevel_planning_py.exploration.logic_tools import determine_relevant_predicates
from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase

# Other
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools import util, util_mcts, run_util_pddl_bm

# mpl.use("TkAgg")

# ----------------------------------------------------------------------

SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "src_root_dir": SRCROOT,
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "bin_dir": os.path.join(SRCROOT, "bin"),
}


def main():
    # Command line arguments
    args = util.parse_arguments_pddl_bm()

    # Seed RNGs
    if not args.no_seed:
        print("Seeding RNG")
        np.random.seed(0)

    # Load config file
    if len(args.config_file_path) == 0:
        config_file_path = os.path.join(SRCROOT, "config", "main_pddl_bm.yaml")
    else:
        config_file_path = args.config_file_path
    cfg = ConfigYaml(config_file_path)

    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")

    gt_domain_file = os.path.join(
        os.path.expanduser("~"),
        "ipc2020-domains",
        "Rover-GTOHP",
        "domain.hddl",
    )
    assert os.path.isfile(gt_domain_file), f"{gt_domain_file} does not exist"
    init_dir = os.path.join(
        PATHS["src_root_dir"], "data", "hierarchical_domains", "rover"
    )
    init_domain_file = os.path.join(
        init_dir, cfg.getparam(["user_input", "init_domain"])
    )
    init_problem_file = os.path.join(
        init_dir, cfg.getparam(["user_input", "init_problem"])
    )
    parser_init = CustomPDDLParser()
    parser_init.parse_domain(init_domain_file)
    parser_init.parse_problem(init_problem_file)

    # Set goal
    goals = ast.literal_eval(cfg.getparam(["user_input", "goals"]))

    world = PDDLSimWorld(gt_domain_file, parser_init)
    kb, preds = run_util_pddl_bm.setup_knowledge_base_pddl_bm(
        PATHS, parser_init, args, goals, world
    )

    # PDDL extender
    pddl_ex = PDDLExtender(kb, preds)

    # Set up exploration
    get_execution_system = lambda seq, parames: SequentialExecutionPDDL(
        seq, parames, world, kb
    )
    xplorer = PDDLExplorer(kb.objects, pddl_ex, kb, cfg, world, get_execution_system)
    goal_objects = xplorer._get_items_goal()
    closeby_objects = list(kb.objects.keys())
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
        cfg.getparam(["mcts", "avoid_double_nav"]),
        relevant_objects=relevant_objects,
        explorer=xplorer,
    )
    mcts_search = mcts.HLPTreeSearch(mcts_root_node, xplorer, cfg)

    # ---------------------------------------------------------------

    metrics = mcts_search.tree_search()
    metrics["goals"] = kb.goals

    kb_clone = KnowledgeBase(PATHS, domain_name=parser_init.domain_name)
    kb_clone.duplicate(kb)
    util_mcts.mcts_exit_handler(
        mcts_root_node, time_string, cfg, metrics, kb_clone, PATHS, plot_graph=False
    )


if __name__ == "__main__":
    main()
