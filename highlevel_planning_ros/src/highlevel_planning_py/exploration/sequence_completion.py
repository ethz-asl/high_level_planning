from highlevel_planning_py.exploration import logic_tools
from copy import deepcopy


def complete_sequence(sequence, parameters, relevant_objects, explorer):
    completed_sequence, completed_parameters = list(), list()
    precondition_sequence, precondition_params = list(), list()
    key_action_indices = [0] * len(sequence)

    # Determine initial state
    states = explorer.knowledge_base.initial_state_predicates.union(
        explorer.knowledge_base.object_predicates
    )

    for action_idx, action_name in enumerate(sequence):
        action_description = explorer.knowledge_base.actions[action_name]
        goals = action_description["preconds"]
        parameterized_goals = logic_tools.parametrize_predicate_list(
            goals, parameters[action_idx]
        )

        # Find sequence that makes this action possible
        plan = explorer.knowledge_base.solve_temp(
            parameterized_goals,
            initial_predicates=states,
            specific_generalized_objects=explorer.generalized_objects,
        )
        if plan is False:
            return False
        elif len(plan) > 0:
            fill_sequence, fill_parameters = plan

            # Resample positions because the planner just randomly picked some
            resample_positions(
                fill_sequence,
                fill_parameters,
                relevant_objects,
                states,
                parameterized_goals,
                explorer,
            )

            fill_sequence_effects = logic_tools.determine_sequence_effects(
                explorer.knowledge_base, fill_sequence, fill_parameters
            )

            # Apply fill sequence to current state
            logic_tools.apply_effects_to_state(states, fill_sequence_effects)

            # Save the sequence extension
            if action_idx == 0:
                precondition_sequence = deepcopy(fill_sequence)
                precondition_params = deepcopy(fill_parameters)
            else:
                completed_sequence.extend(fill_sequence)
                completed_parameters.extend(fill_parameters)

        # Apply actual action to current state
        parameterized_effects = logic_tools.parametrize_predicate_list(
            action_description["effects"], parameters[action_idx]
        )
        logic_tools.apply_effects_to_state(states, parameterized_effects)

        completed_sequence.append(action_name)
        completed_parameters.append(parameters[action_idx])
        key_action_indices[action_idx] = len(completed_sequence) - 1
    return (
        completed_sequence,
        completed_parameters,
        precondition_sequence,
        precondition_params,
        key_action_indices,
    )


def resample_positions(
    sequence, parameters, relevant_objects, initial_state, goal_state, explorer
):
    action_descriptions = list()
    parameters_fixed = [] * len(sequence)

    # Forward pass
    states = [(st[0], True, list(st[1:])) for st in initial_state]
    for action_idx, action in enumerate(sequence):
        action_descriptions.append(explorer.knowledge_base.actions[action])
        parameters_fixed.append(
            {
                param_spec[0]: False
                for param_spec in action_descriptions[action_idx]["params"]
            }
        )
        parameterized_preconds = logic_tools.parametrize_predicate_list(
            action_descriptions[action_idx]["preconds"], parameters[action_idx]
        )
        for precond_idx, precond in enumerate(parameterized_preconds):
            for state_idx, state in enumerate(states):
                if precond[:2] == state[:2] and tuple(precond[2]) == tuple(state[2]):
                    for precond_param_idx in range(len(precond[2])):
                        param_name = action_descriptions[action_idx]["preconds"][
                            precond_idx
                        ][2][precond_param_idx]
                        if not explorer.knowledge_base.is_type(
                            parameters[action_idx][param_name], "robot"
                        ):
                            parameters_fixed[action_idx][param_name] = True

        # Apply parts of the effect that contain fixed variables
        state_idx_to_remove = list()
        for effect in action_descriptions[action_idx]["effects"]:
            for effect_parameter in effect[2]:
                if parameters_fixed[action_idx][effect_parameter]:
                    parameterized_effect = logic_tools.parametrize_predicate(
                        effect, parameters[action_idx]
                    )
                    for state_idx, state in enumerate(states):
                        if parameterized_effect[0] == state[0] and tuple(
                            parameterized_effect[2]
                        ) == tuple(state[2]):
                            state_idx_to_remove.append(state_idx)
                    states.append(parameterized_effect)
                    break
        state_idx_to_remove.sort(reverse=True)
        for state_idx in state_idx_to_remove:
            del states[state_idx]

    # Backward pass
    states = deepcopy(goal_state)
    for action_idx, action in reversed(list(enumerate(sequence))):
        parameterized_effects = logic_tools.parametrize_predicate_list(
            action_descriptions[action_idx]["effects"], parameters[action_idx]
        )
        for effect_idx, effect in enumerate(parameterized_effects):
            for state_idx, state in enumerate(states):
                if effect[:2] == state[:2] and tuple(effect[2]) == tuple(state[2]):
                    for effect_param_idx in range(len(effect[2])):
                        param_name = action_descriptions[action_idx]["effects"][
                            effect_idx
                        ][2][effect_param_idx]
                        if not explorer.knowledge_base.is_type(
                            parameters[action_idx][param_name], "robot"
                        ):
                            parameters_fixed[action_idx][param_name] = True

        # Remove effects of this actions and add pre-conditions that contain fixed variables
        state_idx_to_remove = list()
        for effect in parameterized_effects:
            for state_idx, state in enumerate(states):
                if effect[:2] == state[:2] and tuple(effect[2]) == tuple(state[2]):
                    state_idx_to_remove.append(state_idx)
        state_idx_to_remove.sort(reverse=True)
        for state_idx in state_idx_to_remove:
            del states[state_idx]
        for precondition in action_descriptions[action_idx]["preconds"]:
            for precondition_parameter in precondition[2]:
                if parameters_fixed[action_idx][precondition_parameter]:
                    parameterized_precondition = logic_tools.parametrize_predicate(
                        precondition, parameters[action_idx]
                    )
                    states.append(parameterized_precondition)
                    break

    # Resample position parameters that are not fixed
    resampled_parameters = dict()
    for idx, action in enumerate(sequence):
        action_description = explorer.knowledge_base.actions[action]
        for param_name, param_type in action_description["params"]:
            param_value = parameters[idx][param_name]
            if not parameters_fixed[idx][param_name]:
                if param_value in resampled_parameters:
                    parameters[idx][param_name] = resampled_parameters[param_value]
                elif explorer.knowledge_base.is_type(param_value, "position"):
                    new_position = explorer.sample_position(relevant_objects)
                    new_param_value = explorer.knowledge_base.add_temp_object(
                        object_type="position", object_value=new_position
                    )
                    parameters[idx][param_name] = new_param_value
                    resampled_parameters[param_value] = new_param_value
