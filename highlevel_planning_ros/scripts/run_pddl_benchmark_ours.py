#!/usr/bin/env python3
import os
import ast
import atexit
from datetime import datetime
import numpy as np

from highlevel_planning_py.pddl_interface.custom_pddl_parser import CustomPDDLParser
from highlevel_planning_py.tools import util, run_util_pddl_bm
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools.reporter import Reporter
from highlevel_planning_py.tools.exploration_management import run_exploration
from highlevel_planning_py.exploration.pddl_extender import PDDLExtender

from highlevel_planning_py.sim_pddl.explorer import PDDLExplorer
from highlevel_planning_py.sim_pddl.world import PDDLSimWorld
from highlevel_planning_py.sim_pddl.sequential_execution import SequentialExecutionPDDL


SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "src_root_dir": SRCROOT,
    "asset_dir": os.path.join(SRCROOT, "data", "models"),
    "bin_dir": os.path.join(SRCROOT, "bin"),
}


def main():
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

    domain_dir = os.path.join(PATHS["data_dir"], "knowledge", parser_init.domain_name)
    util.check_path_exists(domain_dir)
    assert cfg.getparam(["domain_name"]) == parser_init.domain_name

    # Set up reporter
    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")
    rep = Reporter(
        PATHS,
        cfg,
        domain_name=parser_init.domain_name,
        time_string=time_string,
        domain_file=args.domain_file,
    )
    atexit.register(util.exit_handler, rep)

    goals = ast.literal_eval(cfg.getparam(["user_input", "goals"]))
    demo_sequence = ast.literal_eval(cfg.getparam(["user_input", "demo_sequence"]))
    demo_parameters = ast.literal_eval(cfg.getparam(["user_input", "demo_parameters"]))
    if len(demo_sequence) == 0:
        print("No demonstration given")
        demo_sequence, demo_parameters = None, None

    world = PDDLSimWorld(gt_domain_file, parser_init)
    kb, preds = run_util_pddl_bm.setup_knowledge_base_pddl_bm(
        PATHS, parser_init, args, goals, world
    )

    pddl_ex = PDDLExtender(kb, preds)
    get_execution_system = lambda seq, parames: SequentialExecutionPDDL(
        seq, parames, world, kb
    )
    xplorer = PDDLExplorer(kb.objects, pddl_ex, kb, cfg, world, get_execution_system)
    run_exploration(
        world,
        kb,
        rep,
        args,
        get_execution_system,
        xplorer,
        demo_sequence,
        demo_parameters,
    )
    print("done")


if __name__ == "__main__":
    main()
