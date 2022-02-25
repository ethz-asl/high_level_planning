import py_trees
from highlevel_planning.skills.grasping import ActionGrasping
from highlevel_planning.skills.navigate import ActionNavigate
from highlevel_planning.skills.placing import ActionPlacing
from highlevel_planning.skills.pddl_descriptions import get_action_descriptions
from highlevel_planning.execution.condition_check import ConditionChecker_Predicate
from highlevel_planning.execution.custom_chooser import CustomChooser
from highlevel_planning.execution.es import ExecutionSystem
import copy


class AutoBehaviourTree(ExecutionSystem):
    def __init__(self, robot, predicates, plan, goals, pipes):
        self._robot = robot
        self._predicates = predicates
        self.tree = None

        self._pipes = pipes

        # py_trees.logging.level = py_trees.logging.Level.DEBUG

        self._create_tree_from_plan(plan, goals)

        self.ticking = True

    def setup(self):
        self.tree.setup(timeout=15)

    def step(self):
        self.tree.tick()
        return False

    def print_status(self):
        py_trees.display.print_ascii_tree(self.tree.root, show_status=True)

    # ---- Private member functions ----

    def _create_tree_from_plan(self, plan, goals):
        N = len(plan)

        action_descriptions = get_action_descriptions()

        # Compute all actions' preconditions and effects
        all_preconds = [None] * N
        all_effects = [None] * N
        all_action_descriptions = []
        all_action_arg_dicts = []
        for k in range(N):
            plan_item = plan[k]
            plan_item_list = plan_item.split(" ")
            action_name = plan_item_list[1]
            descr_action = action_descriptions[action_name]
            all_action_descriptions.append(descr_action)
            action_arg_dict = {}
            if len(plan_item_list) > 2:
                for i in range(len(plan_item_list[2:])):
                    action_arg_dict[descr_action[1]["params"][i][0]] = plan_item_list[
                        2 + i
                    ]
            all_action_arg_dicts.append(action_arg_dict)

            # Establish pre-conditions
            preconds = []
            for precond in descr_action[1]["preconds"]:
                precond_processed = (
                    precond[0],
                    precond[1],
                    self._process_pred_args(precond[2], action_arg_dict),
                )
                preconds.append(precond_processed)
            all_preconds[k] = preconds

            # Establish effects
            effects = []
            for effect in descr_action[1]["effects"]:
                effect_processed = (
                    effect[0],
                    effect[1],
                    self._process_pred_args(effect[2], action_arg_dict),
                )
                effects.append(effect_processed)
            all_effects[k] = effects

        # Goals
        modified_goals = copy.deepcopy(goals)

        goal_nodes = []
        for goal in modified_goals:
            goal_check = ConditionChecker_Predicate(
                self._predicates.call[goal[0]],
                self._process_pred_args(goal[2], {}, ignore_keyerror=True),
                invert=goal[1],
            )
            goal_nodes.append(goal_check)
        if len(goal_nodes) > 0:
            final_goal_node = py_trees.composites.Sequence(
                name="Final goal reached?", children=goal_nodes
            )
        else:
            final_goal_node = py_trees.behaviours.Success(
                name="Final goal reached? [succ]"
            )

        # Compute all subgoals
        all_goals = [None] * N
        for k in reversed(range(N)):
            goals = []

            # Remove goals that are reached through effects of action k
            for effect in all_action_descriptions[k][1]["effects"]:
                # Replace the parameters with their values
                effect_params = self._process_pred_args(
                    effect[2], all_action_arg_dicts[k], substitute_robot=False
                )
                modified_effect = (effect[0], effect[1], tuple(effect_params))

                # Check if this effect is part of the goal
                goals_to_remove = []
                for goal in modified_goals:
                    if goal == modified_effect:
                        goals_to_remove.append(goal)
                for goal in goals_to_remove:
                    modified_goals.remove(goal)

            # Set up predicate checkers
            goal_nodes = []
            for goal in modified_goals:
                goal_check = ConditionChecker_Predicate(
                    self._predicates.call[goal[0]],
                    self._process_pred_args(goal[2], {}, ignore_keyerror=True),
                    invert=goal[1],
                )
                goal_nodes.append(goal_check)
            if len(goal_nodes) > 0:
                goals_node = py_trees.composites.Sequence(
                    name="Goals action {}".format(k + 1), children=goal_nodes
                )
            else:
                goals_node = py_trees.behaviours.Success(
                    name="Goals action {} [succ]".format(k + 1)
                )

            all_goals[k] = goals_node

        # Set up actual tree
        next_lower_root = None
        for k in range(N):
            plan_item = plan[k]
            plan_item_list = plan_item.split(" ")
            action_name = plan_item_list[1]

            # Need run part
            effect_node = py_trees.composites.Sequence(
                children=self._convert_to_nodes(all_effects[k])
            )
            if len(all_preconds[k + 1 :]) > 0:
                temp = []
                for precond_temp in all_preconds[k + 1 :]:
                    precond_node_temp = py_trees.composites.Sequence(
                        children=self._convert_to_nodes(precond_temp)
                    )
                    temp.append(precond_node_temp)
                precond_node = py_trees.composites.Sequence(children=temp)
                need_run_root = py_trees.composites.Sequence(
                    name="Need run {}".format(k + 1),
                    children=[effect_node, precond_node],
                )
            else:
                need_run_root = effect_node

            # Do run part
            if action_name == "grasp":
                action_node = ActionGrasping(
                    process_pipe=self._pipes["grasp"],
                    target=(plan_item_list[2], None, 0),
                )
            elif action_name == "nav":
                action_node = ActionNavigate(
                    process_pipe=self._pipes["nav"], target_name=plan_item_list[2]
                )
            elif action_name == "place":
                action_node = ActionPlacing(
                    process_pipe=self._pipes["place"], target_pos=plan_item_list[3]
                )
            do_run_root = CustomChooser(
                name="Do run {}".format(k + 1), children=[need_run_root, action_node]
            )

            # Can run part
            precond_node_temp = py_trees.composites.Sequence(
                children=self._convert_to_nodes(all_preconds[k])
            )
            can_run_root = py_trees.composites.Sequence(
                name="Can run {}".format(k + 1),
                children=[all_goals[k], precond_node_temp],
            )

            # Precheck part
            if next_lower_root is not None:
                precheck_root = CustomChooser(
                    name="Precheck {}".format(k + 1),
                    children=[can_run_root, next_lower_root],
                )
            else:
                precheck_root = can_run_root

            # Local root
            local_root = py_trees.composites.Sequence(
                name="Root {}".format(k + 1), children=[precheck_root, do_run_root]
            )
            next_lower_root = local_root

        root = CustomChooser(name="Plan runner", children=[final_goal_node, local_root])
        self.tree = py_trees.trees.BehaviourTree(root)
        self._show_tree()

    def _show_tree(self):
        print("=" * 20)
        print("Behavior tree:")
        print("-" * 20)
        py_trees.display.print_ascii_tree(self.tree.root)
        print("=" * 20)

    def _process_pred_args(
        self, pred_args, arg_dict, substitute_robot=True, ignore_keyerror=False
    ):
        pred_args_processed = list(copy.deepcopy(pred_args))
        for i in range(len(pred_args)):
            try:
                pred_args_processed[i] = arg_dict[pred_args[i]]
            except KeyError as e:
                if not ignore_keyerror:
                    raise e
            if pred_args_processed[i] == "robot1" and substitute_robot:
                pred_args_processed[i] = self._robot
        return pred_args_processed

    def _convert_to_nodes(self, specs):
        nodes = []
        for spec in specs:
            node = ConditionChecker_Predicate(
                self._predicates.call[spec[0]], spec[2], invert=spec[1]
            )
            nodes.append(node)
        return nodes
