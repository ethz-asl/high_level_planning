from typing import Tuple
from highlevel_planning_py.pddl_interface.custom_pddl_parser import CustomPDDLParser


def create_state_tuple(precond, action_parameters):
    precond_tuple = [precond[0]]
    if len(precond) > 1:
        for param_name in precond[1:]:
            precond_tuple.append(action_parameters[param_name.strip("?")])
    precond_tuple = tuple(precond_tuple)
    return precond_tuple


class PDDLSimWorld:
    def __init__(self, domain_file: str, parser_init: CustomPDDLParser):
        self.parser_gt = CustomPDDLParser()
        self.parser_gt.parse_domain(domain_file)
        self.action_indices = {
            action.name: i for i, action in enumerate(self.parser_gt.actions)
        }

        # Initialize state variable
        self.current_state = set()
        for pred_tuple in parser_init.state:
            self.current_state.add(pred_tuple)

        # Initialize structure for saved states
        self.saved_states = dict()
        self.num_saved_states = 0

    def save_state(self) -> int:
        this_state_id = self.num_saved_states
        self.saved_states[this_state_id] = self.current_state.copy()
        self.num_saved_states += 1
        return this_state_id

    def restore_state(self, state_id: int) -> None:
        self.current_state = self.saved_states[state_id].copy()

    def execute_action(
        self, action_name: str, action_parameters: dict
    ) -> Tuple[bool, str]:
        action_idx = self.action_indices[action_name]
        action_description = self.parser_gt.actions[action_idx]

        # Check if preconditions hold
        for precond in action_description.positive_preconditions:
            precond_tuple = create_state_tuple(precond, action_parameters)
            if precond_tuple not in self.current_state:
                return False, f"Precondition not met: {precond_tuple}"
        for precond in action_description.negative_preconditions:
            precond_tuple = create_state_tuple(precond, action_parameters)
            if precond_tuple in self.current_state:
                return False, f"Precondition not met: {precond_tuple}"

        # Apply effects
        for effect in action_description.add_effects:
            effect_tuple = create_state_tuple(effect, action_parameters)
            self.current_state.add(effect_tuple)
        for effect in action_description.del_effects:
            effect_tuple = create_state_tuple(effect, action_parameters)
            if effect_tuple in self.current_state:
                self.current_state.remove(effect_tuple)

        return True, ""

    def check_predicate(self, predicate: Tuple[str, ...]) -> bool:
        return predicate in self.current_state
