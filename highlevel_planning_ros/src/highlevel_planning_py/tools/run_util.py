import argparse
import os
import pybullet as p
import pickle
import numpy as np

from highlevel_planning_py.sim.world import WorldPybullet

from highlevel_planning_py.sim.robot_arm import RobotArmPybullet
from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase
from highlevel_planning_py.skills import pddl_descriptions
from highlevel_planning_py.knowledge.predicates import Predicates


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reuse-objects",
        action="store_true",
        help="if given, the simulation does not reload objects. Objects must already be present.",
    )
    parser.add_argument(
        "-s",
        "--sleep",
        action="store_true",
        help="if given, the simulation will sleep for each update step, to mimic real time execution.",
    )
    parser.add_argument(
        "-m",
        "--method",
        action="store",
        help="determines in which mode to connect to pybullet. Can be 'gui', 'direct' or 'shared'.",
        default="gui",
    )
    parser.add_argument(
        "-n", "--noninteractive", action="store_true", help="skip user prompts."
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="if given, RNGs are not initialized with a random seed.",
    )
    parser.add_argument(
        "-d",
        "--domain-file",
        action="store",
        default="_domain.pkl",
        help="The file name of the domain file loaded/stored by the knowledge base.",
    )
    parser.add_argument(
        "-c",
        "--config-file-path",
        action="store",
        help="Absolute path to the config file to use.",
        default="",
    )
    args = parser.parse_args()
    return args


def save_pybullet_sim(args, savedir, scene, robot=None):
    robot_mdl = robot.model if robot is not None else None
    if not args.reuse_objects:
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        with open(os.path.join(savedir, "objects.pkl"), "wb") as output:
            pickle.dump((scene.objects, robot_mdl), output)
        p.saveBullet(os.path.join(savedir, "state.bullet"))


def restore_pybullet_sim(savedir, args):
    objects = None
    robot_mdl = None
    if args.reuse_objects:
        with open(os.path.join(savedir, "objects.pkl"), "rb") as pkl_file:
            objects, robot_mdl = pickle.load(pkl_file)
    return objects, robot_mdl


def setup_pybullet_world(scene_object, assets_dir, args, savedir=None, objects=None):
    # Create world
    world = WorldPybullet(
        style=args.method,
        sleep=args.sleep,
        load_objects=not args.reuse_objects,
        savedir=savedir,
    )
    p.setAdditionalSearchPath(assets_dir, physicsClientId=world.client_id)

    scene = scene_object(world, assets_dir, restored_objects=objects)

    return scene, world


def setup_robot(world, cfg, asset_dir, robot_mdl):
    # Spawn robot
    robot = RobotArmPybullet(world, cfg, asset_dir, robot_mdl)
    robot.reset()
    robot.to_start()
    world.step_seconds(0.5)
    return robot


def setup_knowledge_base(
    paths,
    scene,
    robot,
    cfg,
    time_string,
    goals,
    pb_client_id,
    domain_file="_domain.pkl",
):
    # Set up planner interface and domain representation
    kb = KnowledgeBase(
        paths,
        domain_name=scene.__class__.__name__,
        time_string=time_string,
        domain_file=domain_file,
    )
    kb.set_goals(goals)

    # Add basic skill descriptions
    skill_descriptions = pddl_descriptions.get_action_descriptions()
    for skill_name, description in skill_descriptions.items():
        kb.add_action(
            action_name=skill_name, action_definition=description, overwrite=True
        )

    # Add required types
    kb.add_type("robot")
    kb.add_type("navgoal")  # Anything we can navigate to
    kb.add_type("position", "navgoal")  # Pure positions
    kb.add_type("item", "navgoal")
    kb.add_type("item-graspable", "item")  # Anything we can grasp
    kb.add_type("grasp_id")

    # Add origin
    kb.add_object("origin", "position", np.array([0.0, 0.0, 0.0]))
    kb.add_object("robot1", "robot")
    kb.add_object("grasp0", "grasp_id", (0, 0))
    kb.add_object("grasp1", "grasp_id", (0, 1))

    # Set up predicates
    preds = Predicates(scene, robot, kb, cfg, pb_client_id)
    kb.set_predicate_funcs(preds)

    for descr in preds.descriptions.items():
        kb.add_predicate(
            predicate_name=descr[0], predicate_definition=descr[1], overwrite=True
        )

    # Planning problem
    kb.populate_visible_objects(scene)
    kb.check_predicates(scene.objects, robot.model.uid, pb_client_id)

    return kb, preds
