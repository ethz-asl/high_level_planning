from copy import deepcopy
from itertools import product


def parametrize_predicate(predicate, action_parameters):
    assert len(predicate) == 3
    return (
        predicate[0],
        predicate[1],
        tuple([action_parameters[obj_name] for obj_name in predicate[2]]),
    )


def parametrize_predicate_list(predicates, action_parameters):
    parametrized_predicates = list()
    for predicate in predicates:
        parametrized_predicate = parametrize_predicate(predicate, action_parameters)
        parametrized_predicates.append(parametrized_predicate)
    return parametrized_predicates


def unparametrize_predicate(predicate, action_parameters):
    assert len(predicate) == 3
    inverted_params = invert_dict_1to1(action_parameters)
    return (
        predicate[0],
        predicate[1],
        tuple([inverted_params[param_value] for param_value in predicate[2]]),
    )


def determine_sequence_preconds(knowledge_base, sequence, parameters):
    seq_preconds = list()
    for action_idx, action_id in reversed(list(enumerate(sequence))):
        action_descr = knowledge_base.actions[action_id]

        # Remove all effects of this action from the precond list
        preconds_to_remove = list()
        for seq_precond in seq_preconds:
            for effect in action_descr["effects"]:
                parametrized_effect = parametrize_predicate(
                    effect, parameters[action_idx]
                )
                if seq_precond == parametrized_effect:
                    preconds_to_remove.append(seq_precond)
        for seq_precond in preconds_to_remove:
            seq_preconds.remove(seq_precond)

        # Add all preconditions of this action to the precond list
        for precond in action_descr["preconds"]:
            parametrized_precond = parametrize_predicate(
                precond, parameters[action_idx]
            )
            if parametrized_precond not in seq_preconds:
                seq_preconds.append(parametrized_precond)
    return seq_preconds


def determine_sequence_effects(knowledge_base, sequence, parameters):
    seq_effects = list()
    for action_idx, action_id in enumerate(sequence):
        action_descr = knowledge_base.actions[action_id]

        # Remove colliding effects from the effect list
        effects_to_remove = list()
        for effect in action_descr["effects"]:
            parametrized_effect = parametrize_predicate(effect, parameters[action_idx])
            for seq_effect in seq_effects:
                if (
                    seq_effect[0] == parametrized_effect[0]
                    and seq_effect[2] == parametrized_effect[2]
                ):
                    effects_to_remove.append(seq_effect)
        for seq_effect in effects_to_remove:
            try:
                seq_effects.remove(seq_effect)
            except ValueError:
                continue

        # Add all effects of this action to the effect list
        for effect in action_descr["effects"]:
            parametrized_effect = parametrize_predicate(effect, parameters[action_idx])
            if parametrized_effect not in seq_effects:
                seq_effects.append(parametrized_effect)
    return seq_effects


def test_abstract_feasibility(knowledge_base, sequence, parameters, preconds):
    """
        Takes an action sequence and suitable parameters as inputs and checks
        whether the sequence is logically feasible.
        
        Args:
            knowledge_base:
            sequence (list): The action sequence
            parameters (list): Parameters for each action
            preconds:
        
        Returns:
            bool: True if the sequence is feasible, False otherwise.
        """

    facts = deepcopy(preconds)
    sequence_invalid = False
    for action_idx, action_id in enumerate(sequence):
        action_descr = knowledge_base.actions[action_id]

        # Check if any fact contradicts the pre-conditions of this action
        for fact in facts:
            for precond in action_descr["preconds"]:
                parametrized_precond = parametrize_predicate(
                    precond, parameters[action_idx]
                )
                if (
                    fact[0] == parametrized_precond[0]
                    and fact[2] == parametrized_precond[2]
                    and not fact[1] == parametrized_precond[1]
                ):
                    sequence_invalid = True
                    break
            if sequence_invalid:
                break

        if sequence_invalid:
            break

        for effect in action_descr["effects"]:
            parametrized_effect = parametrize_predicate(effect, parameters[action_idx])
            facts_to_remove = list()
            for fact in facts:
                if (
                    fact[0] == parametrized_effect[0]
                    and fact[2] == parametrized_effect[2]
                ):
                    facts_to_remove.append(fact)
            for fact in facts_to_remove:
                facts.remove(fact)
            facts.append(parametrized_effect)
    return not sequence_invalid


def invert_dict(original_dict):
    inverted_dict = dict()
    for key in original_dict:
        if type(original_dict[key]) is list:
            for val in original_dict[key]:
                if val not in inverted_dict:
                    inverted_dict[val] = list()
                inverted_dict[val].append(key)
        else:
            if original_dict[key] not in inverted_dict:
                inverted_dict[original_dict[key]] = list()
            inverted_dict[original_dict[key]].append(key)
    for val in inverted_dict:
        inverted_dict[val] = list(dict.fromkeys(inverted_dict[val]))
    return inverted_dict


def invert_dict_1to1(original_dict):
    inverted_dict = dict()
    for key in original_dict:
        assert isinstance(key, str)
        assert isinstance(original_dict[key], str)
        inverted_dict[original_dict[key]] = key
    return inverted_dict


def parse_plan(plan, actions):
    sequence = list()
    parameters = list()
    for plan_item in plan:
        plan_item_list = plan_item.split(" ")
        action_name = plan_item_list[1]
        action_name = action_name.split("_")[0]
        if len(plan_item_list) > 2:
            action_parameters = plan_item_list[2:]
        else:
            action_parameters = []

        action_description = actions[action_name]
        param_dict = dict()
        assert len(action_parameters) == len(action_description["params"])
        for param_idx, param_spec in enumerate(action_description["params"]):
            param_dict[param_spec[0]] = action_parameters[param_idx]
        sequence.append(action_name)
        parameters.append(param_dict)
    return sequence, parameters


def apply_effects_to_state(states: set, effects):
    states_to_remove = set()
    for effect in effects:
        for state in states:
            if effect[0] == state[0] and list(effect[2]) == list(state[1:]):
                states_to_remove.add(state)
    for state in states_to_remove:
        states.remove(state)
    for effect in effects:
        if effect[1]:
            states.add((effect[0],) + tuple(effect[2]))


def find_all_parameter_assignments(parameters, relevant_objects, knowledge_base):
    # Find possible parameter assignments
    parameter_assignments = list()
    for param_idx, param in enumerate(parameters):
        assignments_this_param = list()
        if param[1] == "robot":
            assignments_this_param.append("robot1")
        else:
            for obj in relevant_objects:
                if knowledge_base.is_type(obj, param[1]):
                    assignments_this_param.append(obj)
        parameter_assignments.append(assignments_this_param)
    return parameter_assignments


def determine_relevant_predicates(
    relevant_objects, knowledge_base, ignore_predicates=None
):
    """
    Determine all predicates of objects involved in this action and objects that are close to them
    """
    predicate_descriptions = knowledge_base.predicate_funcs.descriptions
    relevant_predicates = list()
    for pred in predicate_descriptions:
        if ignore_predicates is not None and pred in ignore_predicates:
            continue

        parameters = predicate_descriptions[pred]

        parameter_assignments = find_all_parameter_assignments(
            parameters, relevant_objects, knowledge_base
        )

        for parametrization in product(*parameter_assignments):
            relevant_predicates.append((pred, parametrization))
    return relevant_predicates


def measure_predicates(predicates, knowledge_base):
    measurements = list()
    for pred in predicates:
        res = knowledge_base.predicate_funcs.call[pred[0]](*pred[1])
        measurements.append(res)
    return measurements
