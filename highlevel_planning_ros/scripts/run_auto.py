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
from highlevel_planning_py.execution.es_sequential_execution import (
    execute_plan_sequentially,
)

# Learning
from highlevel_planning_py.exploration.explorer import Explorer
from highlevel_planning_py.exploration.pddl_extender import PDDLExtender

# Other
from highlevel_planning_py.tools.config import ConfigYaml
from highlevel_planning_py.tools import run_util
from highlevel_planning_py.tools.reporter import Reporter

# ----------------------------------------------------------------------

SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATHS = {
    "data_dir": os.path.join(os.path.expanduser("~"), "Data", "highlevel_planning"),
    "src_root_dir": SRCROOT,
    "asset_dir": os.path.join(SRCROOT, "models"),
    "bin_dir": os.path.join(SRCROOT, "bin"),
}


def exit_handler(rep: Reporter):
    rep.write_result_file()


def print_plan(sequence, parameters):
    print("---------------------------------------------------")
    print("Found plan:")
    for idx, seq_item in enumerate(sequence):
        print(f"{seq_item} {parameters[idx]}")
    print("---------------------------------------------------")


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

    # Populate simulation
    scene, world = run_util.setup_pybullet_world(
        ScenePlanning2, PATHS["asset_dir"], args, savedir, objects
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
    atexit.register(exit_handler, rep)

    # Save state
    run_util.save_pybullet_sim(args, savedir, scene, robot)

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

    # Set up exploration
    xplorer = Explorer(skill_set, robot, scene.objects, pddl_ex, kb, cfg, world)

    # ---------------------------------------------------------------

    # Store initial state
    initial_state_id = world.save_state()

    explored = False
    while True:
        # Plan
        plan = kb.solve()
        rep.report_after_planning(plan, kb)

        # Execute
        if plan is False:
            planning_failed = True
            print("No plan found.")
        else:
            planning_failed = False
            sequence, parameters = plan
            print_plan(sequence, parameters)
            if not args.noninteractive:
                input("Press enter to run...")
            res = execute_plan_sequentially(
                sequence, parameters, skill_set, kb, verbose=True
            )
            rep.report_after_execution(res)
            if res:
                print("Reached goal successfully. Exiting.")
                break
            else:
                print("Failure during plan execution.")

        if explored:
            print("Already explored once, aborting.")
            break

        # Decide what happens next
        if not args.noninteractive:
            choice = input(f"Choose next action: (a)bort, (e)xplore\nYour choice: ")
        else:
            choice = "e"
        if choice == "e":
            # Exploration
            explored = True
            rep.report_before_exploration(kb, plan)
            success, metrics = xplorer.exploration(
                planning_failed,
                demo_sequence,
                demo_parameters,
                state_id=initial_state_id,
                no_seed=args.no_seed,
            )
            rep.report_after_exploration(kb, metrics)
            if not success:
                print("Exploration was not successful")
                break
        else:
            if choice != "a":
                print("Invalid choice, aborting.")
            break


if __name__ == "__main__":
    main()
