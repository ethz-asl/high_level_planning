import numpy as np

from highlevel_planning_py.sim.robot_arm import RobotArmPybullet
from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase
from highlevel_planning_py.skills import pddl_descriptions
from highlevel_planning_py.knowledge.predicates import Predicates


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
