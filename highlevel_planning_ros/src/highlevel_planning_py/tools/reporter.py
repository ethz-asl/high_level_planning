import os
import pickle
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy

from highlevel_planning_py.knowledge.knowledge_base import KnowledgeBase
from highlevel_planning_py.tools.config import ConfigYaml


def write_nested_dict(f, nested_dict, depth):
    spacing = "  " * depth
    for key, value in nested_dict.items():
        this_type = type(value)
        if this_type is dict or this_type is OrderedDict:
            f.write(spacing + key + ":\n")
            write_nested_dict(f, value, depth + 1)
        else:
            tmp = spacing + key
            f.write(f"{tmp:36}: {value}\n")


class Reporter:
    def __init__(
        self,
        paths,
        config: ConfigYaml,
        domain_name,
        time_string: str = None,
        domain_file="_domain.pkl",
    ):
        self.paths = paths
        self.domain_name = domain_name
        self.domain_file = domain_file
        self.data = dict()
        self.metrics = OrderedDict()
        self.metrics["general"] = OrderedDict()
        if time_string is None:
            time_stamp = datetime.now()
            self.metrics["general"]["time"] = time_stamp.strftime("%y%m%d-%H%M%S")
        else:
            self.metrics["general"]["time"] = time_string
        self.metrics["general"]["domain_file"] = domain_file
        print(f"Time string for reporting is {self.metrics['general']['time']}")
        self.data["configuration"] = deepcopy(config._cfg)

        self.planning_idx = 0
        self.explore_idx = 0
        self.run_idx = 0

    @staticmethod
    def _extract_kb_metrics(kb):
        ret = dict()
        ret["actions"] = len(kb.actions)
        ret["types"] = len(kb.types)
        ret["objects"] = len(kb.objects)
        ret["visible_objects"] = len(kb.visible_objects)
        ret["object_predicates"] = len(kb.object_predicates)
        ret["initial_state_predicates"] = len(kb.initial_state_predicates)
        ret["lookup_table"] = len(kb.lookup_table)
        ret["parameterizations"] = len(kb.parameterizations)
        ret["meta_actions"] = len(kb.meta_actions)
        return ret

    def report_after_planning(self, plan, kb):
        self.data[f"plan_{self.planning_idx}"] = dict()
        self.data[f"plan_{self.planning_idx}"]["plan"] = deepcopy(plan)
        success = True if plan is not False else False
        self.metrics[f"plan_{self.planning_idx}"] = OrderedDict(success=str(success))
        kb.save_domain(filename_appendix=f"_plan_step_{self.planning_idx}")
        self.planning_idx += 1

    def report_after_execution(self, res: bool):
        self.metrics[f"exec_{self.run_idx}"] = OrderedDict(success=str(res))
        self.run_idx += 1

    def _assert_explore_dict_exists(self):
        if f"explore_{self.explore_idx}" not in self.data:
            self.data[f"explore_{self.explore_idx}"] = dict()
        if f"explore_{self.explore_idx}" not in self.metrics:
            self.metrics[f"explore_{self.explore_idx}"] = OrderedDict()

    def report_before_exploration(self, knowledge_base: KnowledgeBase, plan):
        self._assert_explore_dict_exists()
        kb_clone = KnowledgeBase(
            self.paths, domain_name=self.domain_name, domain_file=self.domain_file
        )
        kb_clone.duplicate(knowledge_base)

        self.data[f"explore_{self.explore_idx}"]["kb_before"] = kb_clone
        self.metrics[f"explore_{self.explore_idx}"]["goal"] = str(kb_clone.goals)
        self.metrics[f"explore_{self.explore_idx}"][
            "kb_before"
        ] = self._extract_kb_metrics(kb_clone)
        self.data[f"explore_{self.explore_idx}"]["plan_before"] = deepcopy(plan)
        self.metrics[f"explore_{self.explore_idx}"]["plan_before_success"] = str(
            True if plan is not False else False
        )

    def report_after_exploration(
        self, knowledge_base: KnowledgeBase, exploration_metrics: OrderedDict
    ):
        self._assert_explore_dict_exists()
        kb_clone = KnowledgeBase(
            self.paths, domain_name=self.domain_name, domain_file=self.domain_file
        )
        kb_clone.duplicate(knowledge_base)

        self.data[f"explore_{self.explore_idx}"]["kb_after"] = kb_clone
        self.metrics[f"explore_{self.explore_idx}"][
            "kb_after"
        ] = self._extract_kb_metrics(kb_clone)
        self.metrics[f"explore_{self.explore_idx}"]["counters"] = exploration_metrics
        self.explore_idx += 1

    def write_result_file(self):
        savedir = os.path.join(self.paths["data_dir"], "reports")
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        # Write index file
        savefile = os.path.join(savedir, f"{self.metrics['general']['time']}_index.txt")
        with open(savefile, "w") as f:
            write_nested_dict(f, self.metrics, 0)

        # Write data
        self.data["metrics"] = self.metrics
        with open(
            os.path.join(savedir, f"{self.metrics['general']['time']}_data.pkl"), "wb"
        ) as f:
            pickle.dump(self.data, f)

        print(
            f"Reporter wrote result files. Time string: {self.metrics['general']['time']}."
        )
