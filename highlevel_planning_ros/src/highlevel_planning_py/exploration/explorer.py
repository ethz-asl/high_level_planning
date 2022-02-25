# Imports
import numpy as np
from collections import OrderedDict, defaultdict
from copy import deepcopy
import time

from highlevel_planning_py.tools.util import get_combined_aabb
from highlevel_planning_py.exploration import logic_tools
from highlevel_planning_py.exploration.sequence_completion import complete_sequence
from highlevel_planning_py.exploration.precondition_discovery import (
    precondition_discovery,
)
from highlevel_planning_py.exploration.exploration_tools import get_items_closeby
from highlevel_planning_py.execution.es_sequential_execution import SequentialExecution


class Explorer:
    def __init__(
        self,
        skill_set,
        robot,
        scene_objects,
        pddl_extender,
        knowledge_base,
        config,
        world,
    ):
        self.config_params = config.getparam(["explorer"])

        self.action_list = [
            act
            for act in knowledge_base.actions
            if act not in knowledge_base.meta_actions
        ]
        for rm_action in self.config_params["action_denylist"]:
            self.action_list.remove(rm_action)

        self.skill_set = skill_set
        self.robot = robot
        self.robot_uid_ = robot.model.uid
        self.scene_objects = scene_objects
        self.pddl_extender = pddl_extender
        self.knowledge_base = knowledge_base
        self.world = world

        self.current_state_id = None
        self.metrics = None
        self.metrics_prefix = ""
        self.generalized_objects = dict()

    def set_metrics_prefix(self, prefix: str):
        self.metrics_prefix = prefix
        self.metrics[prefix] = OrderedDict()

    def add_metric(self, key: str, value):
        self.metrics[self.metrics_prefix][key] = value

    def exploration(
        self,
        planning_failed: bool,
        demo_sequence=None,
        demo_parameters=None,
        state_id=None,
        no_seed: bool = False,
    ):
        exploration_start_time = time.time()
        total_time_budget = self.config_params["search_budget_sec"]
        total_time_budget_exceeded = False
        self.metrics = OrderedDict()
        self.knowledge_base.clear_temp()

        self.metrics["config"] = self.config_params

        if not no_seed:
            np.random.seed(0)
        sequences_tried = set()

        # Save the state the robot is currently in
        if state_id is None:
            self.current_state_id = self.world.save_state()
        else:
            self.current_state_id = state_id
            self.world.restore_state(state_id)

        # Identify objects that are involved in reaching the goal
        goal_objects = self._get_items_goal()
        radii = self.config_params["radii"]

        # Default closeby objects for demo, prepending and generalization exploration
        closeby_objects = get_items_closeby(
            goal_objects,
            self.scene_objects,
            self.world.client_id,
            self.robot_uid_,
            distance_limit=radii[0],
        )

        special_objects = {"goal": goal_objects, "closeby": closeby_objects}

        res = False
        if not planning_failed and not res and not total_time_budget_exceeded:
            self.set_metrics_prefix("02_prepend")
            tic = time.time()
            time_budget = self.config_params["time_proportion_prepend"] * (
                total_time_budget - (tic - exploration_start_time)
            )
            self.add_metric("time_budget", time_budget)
            res = self._explore_prepending_sequence(
                special_objects, sequences_tried, time_budget, tic
            )
            total_time = time.time() - tic
            self.add_metric("total_time", total_time)
            total_time_budget_exceeded = (
                tic + total_time - exploration_start_time > total_time_budget
            )
        if not res and not total_time_budget_exceeded:
            self.set_metrics_prefix("03_generalize")
            tic = time.time()
            time_budget = self.config_params["time_proportion_generalize"] * (
                total_time_budget - (tic - exploration_start_time)
            )
            self.add_metric("time_budget", time_budget)
            res = self._explore_generalized_action(
                special_objects, sequences_tried, time_budget, tic
            )
            total_time = time.time() - tic
            self.add_metric("total_time", total_time)
            total_time_budget_exceeded = (
                tic + total_time - exploration_start_time > total_time_budget
            )
        if (
            not res
            and demo_sequence is not None
            and demo_parameters is not None
            and not total_time_budget_exceeded
        ):
            self.set_metrics_prefix("01_demo")
            self.add_metric("demo_sequence", demo_sequence)
            self.add_metric("demo_parameters", demo_parameters)
            tic = time.time()
            time_budget = self.config_params["time_proportion_demo"] * (
                total_time_budget - (tic - exploration_start_time)
            )
            self.add_metric("time_budget", time_budget)
            res = self._explore_demonstration(
                demo_sequence,
                demo_parameters,
                special_objects,
                sequences_tried,
                time_budget,
            )
            total_time = time.time() - tic
            self.add_metric("total_time", total_time)
            total_time_budget_exceeded = (
                tic + total_time - exploration_start_time > total_time_budget
            )
        time_per_radius = (
            total_time_budget - (time.time() - exploration_start_time)
        ) / len(radii)
        for radius in radii:
            if not res and not total_time_budget_exceeded:
                self.set_metrics_prefix(f"04_rad{radius}")
                tic = time.time()
                closeby_objects = get_items_closeby(
                    goal_objects,
                    self.scene_objects,
                    self.world.client_id,
                    self.robot_uid_,
                    distance_limit=radius,
                )
                special_objects = {"goal": goal_objects, "closeby": closeby_objects}
                time_left = total_time_budget - (tic - exploration_start_time)
                time_budget = np.min([time_left, time_per_radius])
                self.add_metric("time_budget", time_budget)
                self.add_metric("special_objects", special_objects)
                res = self._explore_goal_objects(
                    sequences_tried, special_objects, time_budget
                )
                total_time = time.time() - tic
                self.add_metric("total_time", total_time)
                total_time_budget_exceeded = (
                    tic + total_time - exploration_start_time > total_time_budget
                )
        return res, self.metrics

    # ----- Different sampling strategies ------------------------------------

    @staticmethod
    def get_sampling_counters_dict():
        return OrderedDict(
            [
                (key, defaultdict(int))
                for key in (
                    "valid_sequences",
                    "preplan_success",
                    "plan_success",
                    "goal_reached",
                )
            ]
        )

    @staticmethod
    def get_timing_counters_dict():
        return OrderedDict(
            [
                (key, defaultdict(float))
                for key in (
                    "sampling",
                    "sequence_completion",
                    "execution",
                    "goal_testing",
                    "sequence_refinement",
                    "domain_extension",
                )
            ]
        )

    def _explore_demonstration(
        self,
        demo_sequence,
        demo_parameters,
        special_objects,
        sequences_tried,
        time_budget,
    ):
        print(f"Exploring demonstration (budget: {time_budget}) ...")
        found_plan = self._sampling_loops_caller(
            special_objects,
            len(demo_sequence),
            len(demo_sequence),
            sequences_tried,
            time_budget,
            given_sequence=demo_sequence,
            given_parameters=demo_parameters,
        )
        return found_plan

    def _explore_prepending_sequence(
        self, special_objects, sequences_tried, time_budget, start_time
    ):
        print(f"Exploring prepending sequence (budget: {time_budget}) ...")
        plan = self.knowledge_base.solve()
        if not plan:
            return False
        sequence, parameters = plan
        relevant_sequence, relevant_parameters = self._extract_goal_relevant_sequence(
            sequence, parameters, fix_all_params=True
        )

        min_sequence_length = len(relevant_sequence)
        # Removed the +1 from min_sequence_length, for the case that the sequence is sufficient but the
        # parameterization leads to unreliable execution.
        max_sequence_length = np.max(
            (self.config_params["max_sequence_length"], len(relevant_sequence) + 1)
        )
        found_plan = self._sampling_loops_caller(
            special_objects,
            min_sequence_length,
            max_sequence_length,
            sequences_tried,
            time_budget=time_budget - (time.time() - start_time),
            given_sequence=relevant_sequence,
            given_parameters=relevant_parameters,
            planning_failed=False,
        )
        return found_plan

    def _explore_generalized_action(
        self, special_objects, sequences_tried, time_budget, start_time
    ):
        print(f"Exploring generalizing action (budget: {time_budget}) ...")

        if len(self.knowledge_base.goals) > 1:
            print("Generalization cannot deal with more than 1 goal right now.")
            return False

        # Check if an action with a similar effect already exists
        self.knowledge_base.clear_temp()
        self.generalized_objects.clear()
        plan = False
        goal = self.knowledge_base.goals[0]
        for act, act_spec in self.knowledge_base.actions.items():
            for effect in act_spec["effects"]:
                if goal[0] == effect[0] and goal[1] == effect[1]:
                    # Give the goal parameter the new types
                    param_dict = {param[0]: param[1] for param in act_spec["params"]}
                    goal_param_names = set()
                    for goal_param_i, goal_param in enumerate(goal[2]):
                        param_name = effect[2][goal_param_i]
                        goal_param_names.add(param_name)
                        new_type = param_dict[param_name]
                        self.knowledge_base.add_temp_object(
                            new_type, object_name=goal_param
                        )
                        if new_type not in self.generalized_objects:
                            self.generalized_objects[new_type] = set()
                        self.generalized_objects[new_type].add(goal_param)

                    # Make the relevant objects fit other parameters of the action found
                    for param_name, param_type in param_dict.items():
                        # If this parameter is filled by a goal object, ignore it for the other objects
                        if param_name in goal_param_names:
                            continue
                        for close_object in special_objects["closeby"]:
                            close_object_type = self.knowledge_base.objects[
                                close_object
                            ][0]
                            if self.knowledge_base.type_x_child_of_y(
                                param_type, close_object_type
                            ):
                                self.knowledge_base.add_temp_object(
                                    param_type, object_name=close_object
                                )
                                if param_type not in self.generalized_objects:
                                    self.generalized_objects[param_type] = set()
                                self.generalized_objects[param_type].add(close_object)

                    plan = self.knowledge_base.solve_temp(
                        self.knowledge_base.goals,
                        specific_generalized_objects=self.generalized_objects,
                    )
                    if plan is not False:
                        # This only tests the first match.
                        break
            if plan is not False:
                break

        if plan is False:
            return False
        sequence, parameters = plan

        relevant_sequence, relevant_parameters = self._extract_goal_relevant_sequence(
            sequence, parameters
        )

        min_sequence_length = len(relevant_sequence)
        max_sequence_length = len(relevant_sequence)
        found_plan = self._sampling_loops_caller(
            special_objects,
            min_sequence_length,
            max_sequence_length,
            sequences_tried,
            time_budget=time_budget - (time.time() - start_time),
            given_sequence=relevant_sequence,
            given_parameters=relevant_parameters,
            planning_failed=False,
        )
        self.generalized_objects.clear()
        return found_plan

    def _explore_goal_objects(self, sequences_tried, special_objects, time_budget):
        print(f"Exploring goal objects (budget: {time_budget}) ...")
        min_sequence_length = self.config_params["min_sequence_length"]
        max_sequence_length = self.config_params["max_sequence_length"]
        self.knowledge_base.clear_temp()
        found_plan = self._sampling_loops_caller(
            special_objects,
            min_sequence_length,
            max_sequence_length,
            sequences_tried,
            time_budget,
        )
        return found_plan

    # ----- Tools for sampling ------------------------------------

    def _sampling_loops_caller(
        self,
        special_objects,
        min_sequence_length,
        max_sequence_length,
        sequences_tried,
        time_budget,
        given_sequence=None,
        given_parameters=None,
        planning_failed=True,
    ):
        found_plan = False
        budget_exceeded = False
        sampling_counters = self.get_sampling_counters_dict()
        sampling_timers = self.get_timing_counters_dict()
        sequence_lengths = list(range(min_sequence_length, max_sequence_length + 1))
        time_per_sequence_length = time_budget / len(sequence_lengths)

        local_start_time = time.time()

        sample_idx = 0
        while True:
            # Time keeping
            iteration_start_time = time.time()
            run_time = iteration_start_time - local_start_time
            if run_time >= time_budget:
                budget_exceeded = True
                break

            # Determine sequence length
            if self.config_params["alternating_sequence_length"]:
                seq_len_idx = sample_idx % len(sequence_lengths)
                sample_idx += 1
            else:
                seq_len_idx = int(np.floor(run_time / time_per_sequence_length))
            seq_len = sequence_lengths[seq_len_idx]

            # Run one loop iteration
            found_plan = self._sampling_loop(
                sequences_tried,
                sampling_counters,
                sampling_timers,
                seq_len,
                given_seq=given_sequence,
                given_params=given_parameters,
                special_objects=special_objects,
                planning_failed=planning_failed,
            )
            if found_plan:
                self.add_metric("successful_seq_len", seq_len)
                break
            else:
                # Avoid large numbers of samples piling up. Keeps symbolic description light.
                self.knowledge_base.clear_temp_samples()

        # Store counters
        for counter in sampling_counters:
            self.add_metric(f"#_{counter}", sampling_counters[counter])
        for timer in sampling_timers:
            self.add_metric(f"t_{timer}", sampling_timers[timer])
        self.add_metric("found_plan", found_plan)
        self.add_metric("time_budget_exceeded", budget_exceeded)

        # Restore initial state
        self.world.restore_state(self.current_state_id)
        self.knowledge_base.clear_temp()
        return found_plan

    def _sampling_loop(
        self,
        sequences_tried,
        counters,
        sampling_timers,
        seq_len: int,
        given_seq=None,
        given_params=None,
        special_objects=None,
        planning_failed=True,
    ):
        found_plan = False
        relevant_objects = special_objects["goal"] + special_objects["closeby"]

        # Restore initial state
        self.world.restore_state(self.current_state_id)

        # Sample sequences until an abstractly feasible one was found
        (success, completion_result) = self._sample_feasible_sequence(
            sequences_tried,
            seq_len,
            sampling_timers,
            given_seq=given_seq,
            given_params=given_params,
            special_objects=special_objects,
        )
        if not success:
            print("Sampling failed. Abort searching in this sequence length.")
            return found_plan
        counters["valid_sequences"][seq_len] += 1

        test_success, test_success_idx, test_timing = self._test_completed_sequence(
            completion_result
        )
        sampling_timers["execution"][seq_len] += test_timing[0]
        sampling_timers["goal_testing"][seq_len] += test_timing[1]
        counters["preplan_success"][seq_len] += test_success[0]
        counters["plan_success"][seq_len] += test_success[1]
        counters["goal_reached"][seq_len] += test_success[2]
        if test_success[2] == 0:
            return found_plan
        print("SUCCESS. Achieved goal, now extending symbolic description.")

        # -----------------------------------------------
        # Extend the symbolic description appropriately

        tic = time.time()

        # Trim sequence if not the whole sequence was needed for success
        assert test_success_idx[0] == "pre" or test_success_idx[0] == "main"
        if test_success_idx[0] == "pre":
            short_seq = completion_result[2][: test_success_idx[1] + 1]
            short_params = completion_result[3][: test_success_idx[1] + 1]
            completion_result = complete_sequence(
                short_seq, short_params, relevant_objects, self
            )
        elif (
            test_success_idx[0] == "main"
            and test_success_idx[1] < len(completion_result[0]) - 1
        ):
            short_seq = completion_result[0][: test_success_idx[1] + 1]
            short_params = completion_result[1][: test_success_idx[1] + 1]
            completion_result = complete_sequence(
                short_seq, short_params, relevant_objects, self
            )

        # Try to find actual key actions (only if there is more than 1 key action)
        last_working_completion_result = completion_result
        key_actions = completion_result[4]
        if len(key_actions) > 1:
            completed_sequence = completion_result[0]
            completed_parameters = completion_result[1]
            maximum_pushback = [0] * len(key_actions)
            for key_action_idx in range(len(key_actions) - 2, -1, -1):
                pushback = 0
                while True:
                    pushback += 1
                    if (
                        key_actions[key_action_idx] + pushback
                        > key_actions[key_action_idx + 1]
                    ):
                        break
                    elif (
                        key_actions[key_action_idx] + pushback
                        < key_actions[key_action_idx + 1]
                    ):
                        modified_key_actions = deepcopy(key_actions)
                        modified_key_actions[key_action_idx] += pushback
                    else:
                        modified_key_actions = deepcopy(key_actions)
                        del modified_key_actions[key_action_idx]
                    modified_key_actions = list(set(modified_key_actions))
                    modified_key_actions.sort()
                    modified_sequence = [
                        completed_sequence[i] for i in modified_key_actions
                    ]
                    modified_parameters = [
                        completed_parameters[i] for i in modified_key_actions
                    ]
                    modified_completion_result = complete_sequence(
                        modified_sequence, modified_parameters, relevant_objects, self
                    )
                    if modified_completion_result is False:
                        continue
                    test_success, _, _ = self._test_completed_sequence(
                        modified_completion_result
                    )
                    if not test_success[2]:
                        continue
                    maximum_pushback[key_action_idx] = pushback
                    last_working_completion_result = modified_completion_result
                key_actions[key_action_idx] += maximum_pushback[key_action_idx]
            self.add_metric("maximum_pushback", maximum_pushback)
            print("Key actions refined")

        toc = time.time()
        sampling_timers["sequence_refinement"][seq_len] += toc - tic

        tic = toc
        completed_sequence = last_working_completion_result[0]
        completed_parameters = last_working_completion_result[1]
        precondition_sequence = last_working_completion_result[2]
        precondition_parameters = last_working_completion_result[3]
        key_actions = last_working_completion_result[4]

        effects_last_action = list()
        no_effect_key_actions = (
            list()
        )  # Collects key actions that have no corresponding precondition candidates
        if len(completed_sequence) > 1:
            # Precondition discovery
            precondition_ret = precondition_discovery(
                relevant_objects, last_working_completion_result, self
            )
            if precondition_ret is False:
                print("Precondition discovery failed")
                return found_plan
            precondition_candidates, precondition_actions = precondition_ret
            precondition_idx = 0
            for key_action_idx in range(len(key_actions) - 1):
                effects_this_action = list()

                # This loop finds all precondition candidates corresponding to a key action
                while True:
                    if (
                        precondition_idx >= len(precondition_actions)
                        or precondition_actions[precondition_idx]
                        > key_actions[key_action_idx]
                    ):
                        break
                    effects_this_action.append(
                        precondition_candidates[precondition_idx]
                    )
                    precondition_idx += 1

                if len(effects_this_action) == 0:
                    # This can happen if no precondition candidate that corresponds to this action
                    # was discovered.
                    no_effect_key_actions.append(key_actions[key_action_idx])
                    continue

                # Determine all key actions to include in new meta action
                if len(no_effect_key_actions) > 0:
                    first_key_action = min(no_effect_key_actions)
                    last_key_action = key_actions[key_action_idx]
                else:
                    first_key_action = key_actions[key_action_idx]
                    last_key_action = key_actions[key_action_idx]
                no_effect_key_actions.clear()

                self.pddl_extender.create_new_action(
                    goals=effects_this_action,
                    meta_preconditions=effects_last_action,
                    sequence=completed_sequence[first_key_action : last_key_action + 1],
                    parameters=completed_parameters[
                        first_key_action : last_key_action + 1
                    ],
                )
                effects_last_action = deepcopy(effects_this_action)
            print("Precondition discovery completed")

        # Make sure that the parameters fit all precondition sequence parameter types
        for idx_action, action in enumerate(precondition_sequence):
            self.pddl_extender.generalize_action(
                action_name=action, parameters=precondition_parameters[idx_action]
            )

        # Add action that reaches the goal
        if planning_failed:
            # Determine all key actions to include in new meta action
            if len(no_effect_key_actions) > 0:
                first_key_action = min(no_effect_key_actions)
                last_key_action = key_actions[-1]
            else:
                first_key_action = key_actions[-1]
                last_key_action = key_actions[-1]

            self.pddl_extender.create_new_action(
                goals=self.knowledge_base.goals,
                meta_preconditions=effects_last_action,
                sequence=completed_sequence[first_key_action : last_key_action + 1],
                parameters=completed_parameters[first_key_action : last_key_action + 1],
            )
            print("New action created")
        else:
            if len(no_effect_key_actions) > 0:
                # Thoughts: this can occur in two situations:
                # 1) A sequence needs to be prepended, a new extended sequence was found successfully, but
                #    the precondition discovery couldn't find out what difference the prepended part makes.
                #    In this case, we could introduce a dummy predicate that gets set by the precondition
                #    sequence, and that the final action depends on. However, we have no chance to detect
                #    what the predicate should be in a new situation, so it's useless.
                # 2) The execution of a sequence failed, because of some random error (e.g. box is super close
                #    shelf's edge and thus not detected to be "inside" the shelf). The two cases cannot really
                #    be distinguished.
                raise RuntimeError("Cannot deal with this situation")
            self.pddl_extender.generalize_action(
                action_name=completed_sequence[key_actions[-1]],
                parameters=completed_parameters[key_actions[-1]],
                additional_preconditions=effects_last_action,
            )
            print("Previous action generalized")
        sampling_timers["domain_extension"][seq_len] += time.time() - tic
        found_plan = True
        return found_plan

    def _sample_feasible_sequence(
        self,
        sequences_tried: set,
        sequence_length: int,
        sampling_timers: OrderedDict,
        given_seq: list = None,
        given_params: list = None,
        special_objects=None,
    ):
        """
        Sample sequences until an abstractly feasible one was found
        """

        if given_seq is None:
            given_seq = list()
            given_params = list()
        else:
            assert len(given_seq) <= sequence_length
            given_seq = deepcopy(given_seq)
            given_params = deepcopy(given_params)

        relevant_objects = special_objects["goal"] + special_objects["closeby"]

        failed_samples = 0
        success = True
        completion_result = None
        while True:
            failed_samples += 1
            if failed_samples > self.config_params["max_failed_samples"]:
                success = False
                break

            tic = time.time()
            if len(given_seq) < sequence_length:
                pre_seq = self._sample_sequence(sequence_length - len(given_seq))
                seq = pre_seq + given_seq
                pre_params = [{}] * (sequence_length - len(given_seq))
                fixed_params = pre_params + given_params
            else:
                seq = given_seq
                fixed_params = given_params

            try:
                params, params_tuple, contains_position_parameter = self._sample_parameters(
                    seq, fixed_params, relevant_objects
                )
            except NameError:
                continue
            if not contains_position_parameter:
                sequence_tuple = (tuple(seq), tuple(params_tuple))
                if sequence_tuple in sequences_tried:
                    sampling_timers["sampling"][sequence_length] += time.time() - tic
                    continue
                sequences_tried.add(sequence_tuple)
            sampling_timers["sampling"][sequence_length] += time.time() - tic

            # Fill in the gaps of the sequence to make it feasible
            tic = time.time()
            completion_result = complete_sequence(seq, params, relevant_objects, self)
            sampling_timers["sequence_completion"][sequence_length] += time.time() - tic
            if completion_result is False:
                continue
            break
        return success, completion_result

    def _sample_sequence(self, length, no_action_repetition=False):
        """
        Function to sample actions of a sequence.

        :param length [int]: length of the sequence to be sampled
        :param no_action_repetition [bool]: If true, there will not be twice the same action in a row.
        :return:
        """
        # Generate the sequence
        sequence = list()
        for _ in range(length):
            if no_action_repetition:
                while True:
                    temp = np.random.choice(self.action_list)
                    if len(sequence) == 0:
                        sequence.append(temp)
                        break
                    elif temp != sequence[-1]:
                        sequence.append(temp)
                        break
            else:
                sequence.append(np.random.choice(self.action_list))
        return sequence

    def _sample_parameters(self, sequence, given_params=None, relevant_objects=None):
        parameter_samples = list()
        parameter_samples_tuples = list()

        if relevant_objects is None:
            relevant_objects = list(self.scene_objects)

        # Create list of relevant items in the scene
        joined_object_dict = self.knowledge_base.joined_objects()
        objects_of_interest_dict = dict()
        for obj in relevant_objects:
            objects_of_interest_dict[obj] = joined_object_dict[obj]
        objects_of_interest_dict["robot1"] = joined_object_dict["robot1"]
        types_by_parent = logic_tools.invert_dict(self.knowledge_base.types)
        objects_of_interest_by_type = logic_tools.invert_dict(objects_of_interest_dict)

        contains_position_parameter = False

        for idx_action, action in enumerate(sequence):
            parameter_samples.append(dict())
            parameters_current_action = list()
            for parameter in self.knowledge_base.actions[action]["params"]:
                obj_type = parameter[1]
                obj_name = parameter[0]

                if given_params is not None and obj_name in given_params[idx_action]:
                    # Copy the given parameter
                    obj_sample = given_params[idx_action][obj_name]
                else:
                    # Sample a value for this parameter
                    if self.knowledge_base.type_x_child_of_y(obj_type, "position"):
                        position = self.sample_position(relevant_objects)
                        obj_sample = self.knowledge_base.add_temp_object(
                            object_type=obj_type, object_value=position
                        )
                        contains_position_parameter = True
                    elif self.knowledge_base.type_x_child_of_y(obj_type, "grasp_id"):
                        object_name = None
                        if "obj" in parameter_samples[idx_action]:
                            object_name = parameter_samples[idx_action]["obj"]
                        elif action in self.knowledge_base.meta_actions:
                            meta_param_translator = self.knowledge_base.meta_actions[
                                action
                            ]["param_translator"]
                            for idx_meta in range(len(meta_param_translator)):
                                meta_param_translator[idx_meta].values()
                                if obj_name in meta_param_translator[idx_meta].values():
                                    grasp_relevant_parameter_name = meta_param_translator[
                                        idx_meta
                                    ][
                                        "obj"
                                    ]
                                    if object_name is not None:
                                        assert (
                                            object_name
                                            == parameter_samples[idx_action][
                                                grasp_relevant_parameter_name
                                            ]
                                        )
                                    else:
                                        object_name = parameter_samples[idx_action][
                                            grasp_relevant_parameter_name
                                        ]
                        possible_grasps = list()
                        for grasp_id in ["grasp0", "grasp1"]:
                            if self.knowledge_base.predicate_funcs.call["has-grasp"](
                                object_name, grasp_id
                            ):
                                possible_grasps.append(grasp_id)
                        if len(possible_grasps) == 0:
                            raise NameError("No grasp for target object")
                        obj_sample = np.random.choice(possible_grasps)
                    elif self.knowledge_base.type_x_child_of_y(obj_type, "robot"):
                        obj_sample = "robot1"
                    else:
                        objects_to_sample_from = self.knowledge_base.get_objects_by_type(
                            obj_type,
                            types_by_parent,
                            objects_of_interest_by_type,
                            visible_only=True,
                            include_generalized_objects=True,
                        )
                        if len(objects_to_sample_from) == 0:
                            # If no suitable object is in the objects of interest, check among all objects
                            objects_all_by_type = logic_tools.invert_dict(
                                joined_object_dict
                            )
                            objects_to_sample_from = self.knowledge_base.get_objects_by_type(
                                obj_type,
                                types_by_parent,
                                objects_all_by_type,
                                visible_only=False,
                                include_generalized_objects=True,
                            )
                            if len(objects_to_sample_from) == 0:
                                # No object of the desired type exists, sample new sequence
                                raise NameError(
                                    "No object of desired type among objects of interest"
                                )
                        obj_sample = str(np.random.choice(list(objects_to_sample_from)))
                parameter_samples[idx_action][obj_name] = obj_sample
                parameters_current_action.append(obj_sample)
            parameter_samples_tuples.append(tuple(parameters_current_action))
        assert len(parameter_samples) == len(sequence)
        assert len(parameter_samples_tuples) == len(sequence)
        return parameter_samples, parameter_samples_tuples, contains_position_parameter

    def sample_position(self, relevant_objects):
        # Choose one goal object next to which to sample
        obj_sample = np.random.choice(relevant_objects)
        uid = self.scene_objects[obj_sample].model.uid

        # Get robot base position
        arm_base_pos, _ = self.robot.get_link_pose("panda_link0")

        # Get AABB
        bounding_box = get_combined_aabb(uid)

        # Inflate the bounding box
        min_coords = bounding_box[0]
        max_coords = bounding_box[1]
        max_coords += self.config_params["bounding_box_inflation_length"]
        min_coords -= self.config_params["bounding_box_inflation_length"]
        min_coords[2] = np.max([min_coords[2], arm_base_pos[2] - 0.15])
        max_coords[2] = np.max([max_coords[2], min_coords[2] + 0.01])

        assert min_coords[2] < max_coords[2]

        # Sample
        sample = np.random.uniform(low=min_coords, high=max_coords)
        return sample

    def sample_grasp(self, object_name):
        num_links = len(self.scene_objects[object_name].grasp_links)
        if num_links == 0:
            raise NameError
        link_idx = np.random.randint(num_links)
        link_id = self.scene_objects[object_name].grasp_links[link_idx]
        num_grasps = len(self.scene_objects[object_name].grasp_pos[link_id])
        grasp_idx = np.random.randint(num_grasps)
        return link_idx, grasp_idx

    # ----- Other tools ------------------------------------

    def _extract_goal_relevant_sequence(
        self, sequence, parameters, fix_all_params: bool = False
    ):
        """

        Args:
            sequence:
            parameters:
            fix_all_params: If set to true, all parameters of an action that contributes to the goal are fix.
                            If not, only the goal relevant parameters are fixed.

        Returns:

        """
        # Extract parameters from plan
        fixed_parameters_full = list()
        for action_idx, action_name in enumerate(sequence):
            action_description = self.knowledge_base.actions[action_name]
            parameter_assignments = parameters[action_idx]
            fixed_parameters_this_action = dict()
            for effect in action_description["effects"]:
                for goal in self.knowledge_base.goals:
                    if goal[0] == effect[0] and goal[1] == effect[1]:
                        goal_equals_effect = True
                        for goal_param_idx, goal_param in enumerate(goal[2]):
                            if (
                                goal_param
                                != parameter_assignments[effect[2][goal_param_idx]]
                            ):
                                goal_equals_effect = False
                                break
                        if goal_equals_effect:
                            for effect_param_idx, effect_param in enumerate(effect[2]):
                                if effect_param in fixed_parameters_this_action:
                                    assert (
                                        fixed_parameters_this_action[effect_param]
                                        == goal[2][effect_param_idx]
                                    )
                                else:
                                    fixed_parameters_this_action[effect_param] = goal[
                                        2
                                    ][effect_param_idx]
            if fix_all_params and len(fixed_parameters_this_action) > 0:
                for param in action_description["params"]:
                    if param[0] not in fixed_parameters_this_action:
                        fixed_parameters_this_action[param[0]] = parameter_assignments[
                            param[0]
                        ]
            fixed_parameters_full.append(fixed_parameters_this_action)
        assert len(fixed_parameters_full) == len(sequence)

        # Determine which actions are goal relevant and remove the rest
        relevant_parameters = list()
        relevant_sequence = list()
        for action_idx, action_name in enumerate(sequence):
            if len(fixed_parameters_full[action_idx]) > 0:
                relevant_parameters.append(fixed_parameters_full[action_idx])
                relevant_sequence.append(action_name)
        return relevant_sequence, relevant_parameters

    def _get_items_goal(self, objects_only=False):
        """
        Get objects that are involved in the goal description
        """
        item_list = list()
        for goal in self.knowledge_base.goals:
            for arg in goal[2]:
                if objects_only:
                    if arg in self.scene_objects:
                        item_list.append(arg)
                else:
                    item_list.append(arg)
        return item_list

    def _test_completed_sequence(self, completion_result):
        # Restore initial state
        self.world.restore_state(self.current_state_id)
        success = np.array([0, 0, 0])
        success_idx = None

        time_execution = 0.0
        time_goal_testing = 0.0

        (
            completed_sequence,
            completed_parameters,
            precondition_sequence,
            precondition_parameters,
            key_actions,
        ) = completion_result

        # Precondition sequence
        es = SequentialExecution(
            self.skill_set,
            precondition_sequence,
            precondition_parameters,
            self.knowledge_base,
        )
        es.setup()
        for i in range(len(precondition_sequence)):
            tic = time.time()
            step_success, _, step_msgs = es.step(index=i)
            toc = time.time()
            time_execution += toc - tic
            if not step_success:
                return success, success_idx, (time_execution, time_goal_testing)
            tic = toc
            goal_success = self.knowledge_base.test_goals()
            toc = time.time()
            time_goal_testing += toc - tic
            if goal_success:
                success[0] = 1
                success[2] = 1
                success_idx = ("pre", i)
                return success, success_idx, (time_execution, time_goal_testing)
        success[0] = 1

        # Main sequence
        es = SequentialExecution(
            self.skill_set,
            completed_sequence,
            completed_parameters,
            self.knowledge_base,
        )
        es.setup()
        for i in range(len(completed_sequence)):
            tic = time.time()
            step_success, _, step_msgs = es.step(index=i)
            toc = time.time()
            time_execution += toc - tic
            if not step_success:
                return success, success_idx, (time_execution, time_goal_testing)
            tic = toc
            goal_success = self.knowledge_base.test_goals()
            toc = time.time()
            time_goal_testing += toc - tic
            if goal_success:
                success[2] = 1
                success_idx = ("main", i)
                break
        success[1] = 1
        return success, success_idx, (time_execution, time_goal_testing)
