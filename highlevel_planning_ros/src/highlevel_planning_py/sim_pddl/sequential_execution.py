from typing import List
from highlevel_planning_py.execution.es_sequential_execution import (
    SequentialExecutionBase,
)
from highlevel_planning_py.sim_pddl.world import PDDLSimWorld


class SequentialExecutionPDDL(SequentialExecutionBase):
    def __init__(
        self, sequence: List, parameters: List, world: PDDLSimWorld, knowledge_base
    ):
        super().__init__(sequence, parameters)
        self.world = world
        self.kb = knowledge_base

    def execute_action(self, action_name, action_parameters):
        msgs = list()
        success = True
        if action_name in self.kb.meta_actions:
            expanded = self.kb.expand_step(action_name, action_parameters)
            for sub_step in expanded:
                ret_success, ret_msgs = self.execute_action(sub_step[0], sub_step[1])
                success &= ret_success
                msgs.extend(ret_msgs)
        else:
            res = self.world.execute_action(action_name, action_parameters)
            success &= res[0]
            msgs.append(res[1])
        return success, msgs
