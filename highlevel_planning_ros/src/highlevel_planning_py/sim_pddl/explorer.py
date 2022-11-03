import time
import numpy as np
from collections import OrderedDict
import itertools

from highlevel_planning_py.exploration import logic_tools
from highlevel_planning_py.exploration.explorer import ExplorerBase
from highlevel_planning_py.sim_pddl.sequential_execution import SequentialExecutionPDDL


class PDDLExplorer(ExplorerBase):
    def __init__(
        self,
        entities,
        pddl_extender,
        knowledge_base,
        config,
        world,
        get_execution_system,
    ):
        super(PDDLExplorer, self).__init__(
            scene_objects=entities,
            pddl_extender=pddl_extender,
            knowledge_base=knowledge_base,
            config=config,
            world=world,
            get_execution_system=get_execution_system,
        )

        # Find predicates that cannot be set by any action
        self.uncontrollable_predicates_pos = set(
            knowledge_base.predicate_definitions.keys()
        )
        self.uncontrollable_predicates_neg = set(
            knowledge_base.predicate_definitions.keys()
        )
        for action_descr in knowledge_base.actions.values():
            for effect in action_descr["effects"]:
                if effect[1] and effect[0] in self.uncontrollable_predicates_pos:
                    self.uncontrollable_predicates_pos.remove(effect[0])
                if not effect[1] and effect[0] in self.uncontrollable_predicates_neg:
                    self.uncontrollable_predicates_neg.remove(effect[0])

    def new_sequential_execution(self, seq, params):
        return SequentialExecutionPDDL(seq, params, self.world, self.knowledge_base)

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

        # Save the state the world is currently in
        if state_id is None:
            self.current_state_id = self.world.save_state()
        else:
            self.current_state_id = state_id
            self.world.restore_state(state_id)

        # Identify objects that are involved in reaching the goal
        goal_objects = self._get_items_goal()

        # Default closeby objects for demo, prepending and generalization exploration
        closeby_objects = list(self.knowledge_base.objects.keys())

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
        if not res and not total_time_budget_exceeded:
            self.set_metrics_prefix(f"04_overall")
            tic = time.time()
            time_left = total_time_budget - (tic - exploration_start_time)
            time_budget = time_left
            self.add_metric("time_budget", time_budget)
            self.add_metric("special_objects", special_objects)
            res = self._explore_goal_objects(
                sequences_tried, special_objects, time_budget
            )
            total_time = time.time() - tic
            self.add_metric("total_time", total_time)
        return res, self.metrics

    def _sample_parameters(self, sequence, given_params=None, relevant_objects=None):

        # TODO maybe make sure that parameters satisfy the preconditions?

        parameter_samples = list()
        parameter_samples_tuples = list()

        if relevant_objects is None:
            raise NotImplementedError

        # Create list of relevant items in the scene
        joined_object_dict = self.knowledge_base.joined_objects()
        objects_of_interest_dict = dict()
        for obj in relevant_objects:
            objects_of_interest_dict[obj] = joined_object_dict[obj]
        types_by_parent = logic_tools.invert_dict(self.knowledge_base.types)
        objects_of_interest_by_type = logic_tools.invert_dict(objects_of_interest_dict)

        contains_position_parameter = False

        for idx_action, action in enumerate(sequence):
            parameter_samples.append(dict())

            # Fill in given parameters
            if given_params is not None:
                for param_name, param_value in given_params[idx_action].items():
                    parameter_samples[idx_action][param_name] = param_value

            # Fill parameters that are constrained by preconditions
            action_descr = self.knowledge_base.actions[action]
            param_type_dict = {i[0]: i[1] for i in action_descr["params"]}
            for precond in action_descr["preconds"]:
                if (
                    precond[1] and precond[0] in self.uncontrollable_predicates_pos
                ) or (
                    not precond[1] and precond[0] in self.uncontrollable_predicates_neg
                ):
                    # Parameters already need to fulfill this since it cannot be achieved by other action

                    # Get all possible parameter combinations that fulfill this
                    possible_param_assignments = list()
                    for parameter in precond[2]:
                        if parameter in parameter_samples[idx_action]:
                            possible_param_assignments.append(
                                [parameter_samples[idx_action][parameter]]
                            )
                        else:
                            param_type = param_type_dict[parameter]
                            possible_param_assignments.append(
                                objects_of_interest_by_type[param_type]
                            )
                    all_combinations = itertools.product(*possible_param_assignments)
                    possible_combinations = set()
                    for comb in all_combinations:
                        pred_to_check = (precond[0], *comb)
                        check_result = self.world.check_predicate(pred_to_check)
                        if (precond[1] and check_result) or (
                            not precond[1] and not check_result
                        ):
                            possible_combinations.add(comb)
                    possible_combinations = list(possible_combinations)
                    if len(possible_combinations) == 0:
                        raise NameError("Impossible to satisfy precondition")
                    selected_combination_idx = np.random.randint(
                        0, len(possible_combinations)
                    )
                    selected_combination = possible_combinations[
                        selected_combination_idx
                    ]
                    for i, parameter_name in enumerate(precond[2]):
                        parameter_samples[idx_action][
                            parameter_name
                        ] = selected_combination[i]

            # Fill all other parameters
            for parameter in self.knowledge_base.actions[action]["params"]:
                obj_type = parameter[1]
                obj_name = parameter[0]

                if obj_name in parameter_samples[idx_action]:
                    continue

                # Sample a value for this parameter
                objects_to_sample_from = self.knowledge_base.get_objects_by_type(
                    obj_type,
                    types_by_parent,
                    objects_of_interest_by_type,
                    visible_only=True,
                    include_generalized_objects=True,
                )
                if len(objects_to_sample_from) == 0:
                    # If no suitable object is in the objects of interest, check among all objects
                    objects_all_by_type = logic_tools.invert_dict(joined_object_dict)
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

            # Sort parameter_samples
            parameter_order = {
                param[0]: i for i, param in enumerate(action_descr["params"])
            }
            sorted_parameter_samples = {
                k: v
                for k, v in sorted(
                    parameter_samples[idx_action].items(),
                    key=lambda item: parameter_order[item[0]],
                )
            }
            assert len(sorted_parameter_samples) == len(parameter_samples[idx_action])
            parameter_samples[idx_action] = sorted_parameter_samples

            parameters_current_action = list(parameter_samples[idx_action].values())
            parameter_samples_tuples.append(tuple(parameters_current_action))
        assert len(parameter_samples) == len(sequence)
        assert len(parameter_samples_tuples) == len(sequence)
        return parameter_samples, parameter_samples_tuples, contains_position_parameter
