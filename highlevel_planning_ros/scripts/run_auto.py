import numpy as np
import os
import atexit
import ast
from datetime import datetime

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

# Other
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools import run_util, util
from highlevel_planning_py.tools import util_pybullet
from highlevel_planning_py.tools.reporter import Reporter
from highlevel_planning_py.tools.exploration_management import run_exploration

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

    # Populate simulation
    scene, world = util_pybullet.setup_pybullet_world(
        ScenePlanning2, PATHS, args, savedir, objects
    )
    robot = run_util.setup_robot(world, cfg, PATHS["asset_dir"], robot_mdl)

    # Set up reporter
    time_now = datetime.now()
    time_string = time_now.strftime("%y%m%d-%H%M%S")
    rep = Reporter(
        PATHS,
        cfg,
        domain_name=scene.__class__.__name__,
        time_string=time_string,
        domain_file=args.domain_file,
    )
    atexit.register(util.exit_handler, rep)

    # Save state
    util_pybullet.save_pybullet_sim(args, savedir, scene, robot)

    # ---------------------------------------------------

    # User input

    # Set goal
    goals = ast.literal_eval(cfg.getparam(["user_input", "goals"]))

    # Define a demonstration to guide exploration
    demo_sequence = ast.literal_eval(cfg.getparam(["user_input", "demo_sequence"]))
    demo_parameters = ast.literal_eval(cfg.getparam(["user_input", "demo_parameters"]))
    if len(demo_sequence) == 0:
        print("No demonstration given")
        demo_sequence, demo_parameters = None, None

    # -----------------------------------

    kb, preds = run_util.setup_knowledge_base(
        PATHS,
        scene,
        robot,
        cfg,
        time_string,
        goals,
        world.client_id,
        domain_file=args.domain_file,
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

    # ---------------------------------------------------------------

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


if __name__ == "__main__":
    main()
