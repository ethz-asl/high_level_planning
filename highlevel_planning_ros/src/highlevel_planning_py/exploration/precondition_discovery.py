import numpy as np
from copy import deepcopy

from highlevel_planning_py.execution.es_sequential_execution import (
    execute_plan_sequentially,
)
from highlevel_planning_py.exploration.logic_tools import (
    determine_relevant_predicates,
    measure_predicates,
)


def precondition_discovery(relevant_objects, completion_results, explorer):
    precondition_candidates = list()
    precondition_actions = list()

    (
        completed_sequence,
        completed_parameters,
        precondition_sequence,
        precondition_params,
        _,
    ) = completion_results

    # Restore initial state
    explorer.world.restore_state(explorer.current_state_id)

    relevant_predicates = determine_relevant_predicates(
        relevant_objects, explorer.knowledge_base
    )

    # Check the predicates
    pre_predicates = measure_predicates(relevant_predicates, explorer.knowledge_base)

    # Execute the pre-condition sequence
    es = explorer.get_execution_system(precondition_sequence, precondition_params)
    res = execute_plan_sequentially(es)
    if not res:
        print("[precondition discovery] Failure during precondition sequence execution")
        return False

    current_predicates = measure_predicates(
        relevant_predicates, explorer.knowledge_base
    )
    new_side_effects = detect_predicate_changes(
        relevant_predicates,
        pre_predicates,
        current_predicates,
        precondition_sequence,
        precondition_params,
        explorer,
    )
    precondition_candidates.extend(new_side_effects)
    precondition_actions.extend([-1] * len(new_side_effects))

    # Execute actions one by one, check for non-effect predicate changes
    for idx, action in enumerate(completed_sequence):
        pre_predicates = deepcopy(current_predicates)
        es = explorer.get_execution_system([action], [completed_parameters[idx]])
        res = execute_plan_sequentially(es)
        if not res:
            print(f"[precondition discovery] Failure during action {action}")
            return False
        current_predicates = measure_predicates(
            relevant_predicates, explorer.knowledge_base
        )
        new_side_effects = detect_predicate_changes(
            relevant_predicates,
            pre_predicates,
            current_predicates,
            [action],
            [completed_parameters[idx]],
            explorer,
        )
        precondition_candidates.extend(new_side_effects)
        precondition_actions.extend([idx] * len(new_side_effects))

    # Filter out goals
    candidates_to_remove = list()
    for goal in explorer.knowledge_base.goals:
        for idx, candidate in enumerate(precondition_candidates):
            if tuple(goal[:2]) == tuple(candidate[:2]) and tuple(goal[2]) == tuple(
                candidate[2]
            ):
                candidates_to_remove.append(idx)

    # Filter side effects of last action
    for idx, action_idx in enumerate(precondition_actions):
        if action_idx == len(completed_sequence) - 1:
            candidates_to_remove.append(idx)

    # Filter side effects that get cancelled out again
    for idx, precondition in enumerate(precondition_candidates):
        opposite = (precondition[0], not precondition[1], precondition[2])
        try:
            opposite_index = precondition_candidates.index(opposite, 0, idx)
        except ValueError:
            continue
        candidates_to_remove.extend([idx, opposite_index])

    # Remove duplicates
    precondition_candidate_set = set()
    for i in range(len(precondition_candidates)):
        if precondition_candidates[i] in precondition_candidate_set:
            candidates_to_remove.append(i)
        else:
            precondition_candidate_set.add(precondition_candidates[i])

    candidates_to_remove = list(set(candidates_to_remove))
    candidates_to_remove.sort(reverse=True)
    for idx in candidates_to_remove:
        del precondition_candidates[idx]
        del precondition_actions[idx]

    return precondition_candidates, precondition_actions


def detect_predicate_changes(
    predicate_definitions,
    old_predicates,
    new_predicates,
    action_sequence,
    action_parameters,
    explorer,
):
    side_effects = list()

    changed_indices = np.nonzero(np.logical_xor(old_predicates, new_predicates))
    assert len(changed_indices) == 1
    changed_indices = changed_indices[0]
    for idx in changed_indices:
        predicate_def = predicate_definitions[idx]
        if (len(explorer.config_params["predicate_precondition_allowlist"]) > 0) and (
            predicate_def[0]
            not in explorer.config_params["predicate_precondition_allowlist"]
        ):
            continue
        predicate_state = new_predicates[idx]

        # Check if the last action(s) have this predicate change in their effect list. If yes, ignore.
        predicate_expected = False
        for action_idx, action in enumerate(action_sequence):
            action_descr = explorer.knowledge_base.actions[action]
            for effect in action_descr["effects"]:
                effect_params = tuple(
                    [
                        action_parameters[action_idx][param_name]
                        for param_name in effect[2]
                    ]
                )
                if (
                    effect[0] == predicate_def[0]
                    and effect[1] == predicate_state
                    and effect_params == predicate_def[1]
                ):
                    predicate_expected = True
                    break
            if predicate_expected:
                break
        if predicate_expected:
            continue

        # If we reach here, this is a candidate for the precondition we are trying to determine.
        side_effects.append((predicate_def[0], new_predicates[idx], predicate_def[1]))
    return side_effects
