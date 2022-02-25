from highlevel_planning_py.execution.es import ExecutionSystem
from highlevel_planning_py.tools.util import SkillExecutionError
from highlevel_planning_py.exploration.logic_tools import parametrize_predicate


def execute_plan_sequentially(
    sequence, parameters, skill_set, knowledge_base, verbose=False
):
    if len(sequence) == 0:
        print("Nothing to do.")
        return True
    es = SequentialExecution(skill_set, sequence, parameters, knowledge_base)
    es.setup()
    index = 1
    while True:
        if verbose:
            print("------------- Iteration {} ---------------".format(index))
            es.print_status()
            index += 1
        success, plan_finished, msgs = es.step()
        if not success and verbose:
            print("Error messages:")
            for msg in msgs:
                print(msg)
        if plan_finished or not success:
            break
    return success


class SequentialExecution(ExecutionSystem):
    def __init__(self, skill_set, sequence, parameters, knowledge_base):
        self.ticking = False

        self.skill_set_ = skill_set
        self.sequence = sequence
        self.parameters = parameters

        self.knowledge_base = knowledge_base

        self.current_idx_ = 0
        if len(sequence) == 0:
            self.finished_plan = True
        else:
            self.finished_plan = False

        # Define ignore effects
        self.ignore_effects = {
            "nav-in-reach": [
                ("in-reach", False, ["current_pos", "rob"]),
                ("at", False, ["current_pos", "rob"]),
            ],
            "nav-at": [
                ("at", False, ["current_pos", "rob"]),
                ("in-reach", False, ["current_pos", "rob"]),
            ],
        }

    def step(self, index=None):
        idx = self.current_idx_ if index is None else index

        success = True
        msgs = []
        if not self.finished_plan:
            action_name = self.sequence[idx]
            action_name = action_name.split("_")[0]
            action_parameters = self.parameters[idx]
            success, msgs = self.execute_action(action_name, action_parameters)

            if success and index is None:
                self.current_idx_ += 1
                if self.current_idx_ == len(self.sequence):
                    self.finished_plan = True
        return success, self.finished_plan, msgs

    def execute_action(self, action_name, action_parameters):
        msgs = []
        success = True
        if action_name in self.knowledge_base.meta_actions:
            expanded = self.knowledge_base.expand_step(action_name, action_parameters)
            for sub_step in expanded:
                ret_success, ret_msgs = self.execute_action(sub_step[0], sub_step[1])
                success &= ret_success
                msgs.extend(ret_msgs)
        else:
            try:
                if action_name == "grasp":
                    target_name = action_parameters["obj"]
                    grasp_spec = action_parameters["gid"]
                    grasp_spec = self.knowledge_base.lookup_table[grasp_spec]
                    target_link_idx = grasp_spec[0]
                    target_grasp_idx = grasp_spec[1]
                    res = self.skill_set_["grasp"].grasp_object(
                        target_name, target_link_idx, target_grasp_idx
                    )
                    if not res:
                        raise SkillExecutionError
                elif action_name == "nav-in-reach" or action_name == "nav-at":
                    target_name = action_parameters["goal_pos"]
                    if self.knowledge_base.is_type(target_name, type_query="position"):
                        position = self.knowledge_base.lookup_table[target_name]
                        self.skill_set_["nav"].move_to_pos(position, nav_min_dist=0.2)
                    else:
                        self.skill_set_["nav"].move_to_object(target_name)
                elif action_name == "place":
                    target_pos_name = action_parameters["pos"]
                    target_pos = self.knowledge_base.lookup_table[target_pos_name]
                    self.skill_set_["place"].place_object(target_pos)
                else:
                    raise (
                        NotImplementedError,
                        "SequentialExecution script cannot deal with action "
                        + action_name
                        + " yet.",
                    )
            except SkillExecutionError:
                success = False
                msgs.append("Failure during execution of {}".format(action_name))

        # Check if the effects were reached successfully
        action_description = self.knowledge_base.actions[action_name]
        for effect in action_description["effects"]:
            skip_effect = False
            for ignore_effect in action_description["exec_ignore_effects"]:
                if ignore_effect == effect:
                    skip_effect = True
                    break
            if skip_effect:
                continue

            parameterized_effect = parametrize_predicate(effect, action_parameters)
            res = self.knowledge_base.predicate_funcs.call[effect[0]](
                *parameterized_effect[2]
            )
            if not res == effect[1]:
                # Try it a second time because especially grasp detection is wonky at times.
                res = self.knowledge_base.predicate_funcs.call[effect[0]](
                    *parameterized_effect[2]
                )
            if not res == effect[1]:
                success = False
                msgs.append(
                    "Failed to reach effect {} during action {}".format(
                        parameterized_effect, action_name
                    )
                )

        return success, msgs

    def print_status(self):
        if not self.finished_plan:
            print("Plan item up next: " + self.sequence[self.current_idx_])
        else:
            print("Finished plan")
