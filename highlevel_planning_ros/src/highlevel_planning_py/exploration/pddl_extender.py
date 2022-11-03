import time
from copy import deepcopy

from typing import List

from highlevel_planning_py.exploration.logic_tools import (
    determine_sequence_effects,
    determine_sequence_preconds,
    unparametrize_predicate,
    invert_dict_1to1,
)


class PDDLExtender(object):
    def __init__(self, knowledge_base, predicates):
        self.predicates = predicates
        self.knowledge_base = knowledge_base

    def create_new_action(self, goals, meta_preconditions, sequence, parameters):
        time_string = time.strftime("%y%m%d%H%M%S")
        action_name = sequence[0] + "-" + goals[0][0] + "-" + time_string

        if meta_preconditions is None:
            meta_preconditions = []

        # Compute parameters
        action_params = list()
        already_retyped = list()

        # Compute effects
        action_effects = list()
        for goal in goals:
            for arg in goal[2]:
                self._retype_argument(arg, action_params, already_retyped, time_string)
            action_effects.append(goal)

        sequence_effects = determine_sequence_effects(
            self.knowledge_base, sequence, parameters
        )
        for effect in sequence_effects:
            for arg in effect[2]:
                self._retype_argument(arg, action_params, already_retyped, time_string)
            action_effects.append(effect)

        # Compute preconditions
        action_preconditions = list()
        sequence_preconds = determine_sequence_preconds(
            self.knowledge_base, sequence, parameters
        )
        for precond in sequence_preconds:
            for arg in precond[2]:
                self._retype_argument(arg, action_params, already_retyped, time_string)
            action_preconditions.append(precond)
        for meta_precond in meta_preconditions:
            for arg in meta_precond[2]:
                self._retype_argument(arg, action_params, already_retyped, time_string)
            action_preconditions.append(meta_precond)

        # Determine translation between meta action argument names and sub action argument names
        param_translator = [dict.fromkeys(param_dict) for param_dict in parameters]
        for idx, params in enumerate(parameters):
            for param_name, param_value in params.items():
                if param_value in already_retyped:
                    param_translator[idx][param_name] = param_value
                else:
                    param_translator[idx][param_name] = param_name

        # Collect any effects that shall be ignored during execution
        action_exec_ignore_effects = list()
        for action_idx, action in enumerate(sequence):
            for ignore_effect in self.knowledge_base.actions[action][
                "exec_ignore_effects"
            ]:
                # Translate parameter names
                new_param_names = [
                    param_translator[action_idx][old_param_name]
                    for old_param_name in ignore_effect[2]
                ]
                new_param_names = tuple(new_param_names)
                new_ignore_effect = (
                    ignore_effect[0],
                    ignore_effect[1],
                    new_param_names,
                )

                if new_ignore_effect in action_effects:
                    action_exec_ignore_effects.append(new_ignore_effect)

        # Submit new action description
        new_action_description = {
            "params": action_params,
            "preconds": action_preconditions,
            "effects": action_effects,
            "exec_ignore_effects": action_exec_ignore_effects,
        }

        action_name = self.knowledge_base.add_action(
            action_name, new_action_description, overwrite=False, rename_if_exists=True
        )
        if action_name is False:
            raise ValueError("Failure when trying to add new action to knowledge base.")

        # Determine hidden parameters
        hidden_parameters = [{}] * len(parameters)
        for idx, params in enumerate(parameters):
            for param_name, param_value in params.items():
                if param_value not in already_retyped:
                    hidden_parameters[idx][param_name] = param_value
                    if param_value not in self.knowledge_base.objects:
                        self.knowledge_base.make_permanent(param_value)

        # Full action parameters: list of tuples (name, type, value)
        full_action_params = list()
        for param_spec in action_params:
            full_action_params.append((param_spec[0], param_spec[1], param_spec[0]))

        # Add non-generalizable parametrization
        self._add_parameterization(full_action_params, action_name)

        # Save action meta data
        self.knowledge_base.add_meta_action(
            action_name,
            sequence,
            parameters,
            param_translator,
            hidden_parameters,
            new_action_description,
        )

        return action_name

    def generalize_action(
        self, action_name: str, parameters: dict, additional_preconditions=None
    ):
        if action_name not in self.knowledge_base.meta_actions:
            return

        parameters = deepcopy(parameters)
        time_string = time.strftime("%y%m%d%H%M%S")

        new_description = deepcopy(self.knowledge_base.actions[action_name])
        if additional_preconditions is not None:
            inverted_params = invert_dict_1to1(parameters)
            for additional_pred in additional_preconditions:
                # Make sure that all parameters exist
                for pred_param_value in additional_pred[2]:
                    if pred_param_value not in inverted_params:
                        original_types = self.knowledge_base.objects[pred_param_value]
                        new_type = (
                            f"{original_types[0]}-{pred_param_value}-{time_string}"
                        )
                        new_description["params"].append(
                            [pred_param_value, new_type]
                        )
                        self.knowledge_base.add_type(new_type, original_types[0])
                        self.knowledge_base.add_object(pred_param_value, new_type)
                        parameters[pred_param_value] = pred_param_value
                        inverted_params[pred_param_value] = pred_param_value
                new_pred = unparametrize_predicate(additional_pred, parameters)
                new_description["preconds"].append(new_pred)
            self.knowledge_base.add_action(action_name, new_description, overwrite=True)

        param_list = list()
        for parameter_spec in new_description["params"]:
            parameter_name = parameter_spec[0]
            parameter_type = parameter_spec[1]
            parameter_value = parameters[parameter_name]
            if parameter_value not in self.knowledge_base.objects:
                self.knowledge_base.make_permanent(parameter_value)
            if not self.knowledge_base.is_type(
                object_to_check=parameter_value, type_query=parameter_type
            ):
                self.knowledge_base.add_object(parameter_value, parameter_type)
            param_list.append((parameter_name, parameter_type, parameter_value))
        self._extend_parameterizations(action_name, new_description, parameters)

        # Check if there already is a parameterization recorded for the objects passed.
        # This can happen when the existing parameterization is flaky and often doesn't work
        # in practice. If yes, remove old parameterization.
        self._remove_parameterization(param_list, action_name)

        # Add new parameterization
        self._add_parameterization(param_list, action_name)

    def _retype_argument(self, arg, action_params, already_retyped, time_string):
        if arg not in already_retyped:
            if arg not in self.knowledge_base.objects:
                # Make arguments we use permanent
                self.knowledge_base.make_permanent(arg)
            if self.knowledge_base.is_type(arg, "position"):
                action_params.append([arg, "position"])
            else:
                original_types = self.knowledge_base.objects[arg]
                new_type = f"{original_types[0]}-{arg}-{time_string}"
                action_params.append([arg, new_type])
                self.knowledge_base.add_type(new_type, original_types[0])
                self.knowledge_base.add_object(arg, new_type)
            already_retyped.append(arg)

    def _add_parameterization(self, param_list: List[tuple], action_name: str):
        """
        [summary]

        Args:
            param_list (list): List, where each element is a tuple (name, type, value)
            action_name (string): name of the action
        """
        object_params = list()
        for param in param_list:
            if param[1] != "position":
                object_params.append(param)
        object_params = tuple(object_params)
        for param in param_list:
            if param[1] == "position":
                if action_name not in self.knowledge_base.parameterizations:
                    self.knowledge_base.parameterizations[action_name] = dict()
                if (
                    object_params
                    not in self.knowledge_base.parameterizations[action_name]
                ):
                    self.knowledge_base.parameterizations[action_name][
                        object_params
                    ] = dict()
                if (
                    param[0]
                    not in self.knowledge_base.parameterizations[action_name][
                        object_params
                    ]
                ):
                    self.knowledge_base.parameterizations[action_name][object_params][
                        param[0]
                    ] = set()
                self.knowledge_base.parameterizations[action_name][object_params][
                    param[0]
                ].add(param[2])

    def _remove_parameterization(self, param_list, action_name):
        if action_name not in self.knowledge_base.parameterizations:
            # Nothing to remove
            return
        object_params = list()
        for param in param_list:
            if param[1] != "position":
                object_params.append(param)
        object_params = tuple(object_params)
        if object_params in self.knowledge_base.parameterizations[action_name]:
            for name in self.knowledge_base.parameterizations[action_name][
                object_params
            ]:
                self.knowledge_base.parameterizations[action_name][object_params][
                    name
                ] = set()

    def _extend_parameterizations(
        self, action_name: str, action_description, parameters
    ):
        """
        If a new parameter is added a posteriori, this needs to be reflected in existing parameterizations.

        Returns:

        """

        if action_name not in self.knowledge_base.parameterizations:
            # No need to extend anything
            return

        parameterization_replacements = dict()
        gt_parameter_name_list = [
            param[0] for param in action_description["params"] if param[1] != "position"
        ]
        gt_parameter_type_list = [
            param[1] for param in action_description["params"] if param[1] != "position"
        ]
        for parameterization in self.knowledge_base.parameterizations[action_name]:
            parameter_name_list = [param[0] for param in parameterization]
            if parameter_name_list != gt_parameter_name_list:
                new_parameterization = list(parameterization)
                for i, gt_param in enumerate(gt_parameter_name_list):
                    if (
                        len(new_parameterization) - 1 < i
                        or new_parameterization[i][0] != gt_param
                    ):
                        new_parameterization.insert(
                            i,
                            (gt_param, gt_parameter_type_list[i], parameters[gt_param]),
                        )

                parameterization_replacements[parameterization] = tuple(
                    new_parameterization
                )

        for parameterization in parameterization_replacements:
            new_parameterization = parameterization_replacements[parameterization]
            if (
                new_parameterization
                in self.knowledge_base.parameterizations[action_name]
            ):
                for pos_param_name in self.knowledge_base.parameterizations[
                    action_name
                ][parameterization]:
                    if (
                        pos_param_name
                        in self.knowledge_base.parameterizations[action_name][
                            new_parameterization
                        ]
                    ):
                        self.knowledge_base.parameterizations[action_name][
                            new_parameterization
                        ][pos_param_name].update(
                            self.knowledge_base.parameterizations[action_name][
                                parameterization
                            ][pos_param_name]
                        )
                    else:
                        self.knowledge_base.parameterizations[action_name][
                            new_parameterization
                        ][pos_param_name] = self.knowledge_base.parameterizations[
                            action_name
                        ][
                            parameterization
                        ][
                            pos_param_name
                        ]
            else:
                self.knowledge_base.parameterizations[action_name][
                    new_parameterization
                ] = self.knowledge_base.parameterizations[action_name][parameterization]

            del self.knowledge_base.parameterizations[action_name][parameterization]

    def create_new_predicates(self):
        pass
