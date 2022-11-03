from datetime import datetime
from os import path
from copy import deepcopy

from highlevel_planning_py.exploration.logic_tools import invert_dict


def add_type(type_dict, new_type, parent_type=None):
    assert isinstance(new_type, str)
    if new_type in type_dict:
        # There can only be one parent per type
        ValueError("Type already exists")
    else:
        type_dict[new_type] = parent_type


def add_object(object_dict, object_name, object_type, object_value=None):
    if object_name in object_dict:
        if object_type not in object_dict[object_name]:
            object_dict[object_name].append(object_type)
    else:
        object_dict[object_name] = [object_type]


def type_x_child_of_y(type_dict, x, y):
    parent_type = type_dict[x]
    if x == y:
        return True
    elif parent_type is None:
        return False
    return type_x_child_of_y(type_dict, parent_type, y)


def _preprocess_knowledge(
    actions,
    objects,
    types,
    parameterizations,
    joker_objects,
    specific_generalized_objects,
):
    actions_processed = dict()
    types_processed = deepcopy(types)
    objects_processed = deepcopy(objects)

    for action_name in actions:
        action_descr = actions[action_name]
        if action_name in parameterizations:
            action_suffix = 1
            type_suffix = 1
            for object_param_set in parameterizations[action_name]:
                new_param_types = dict()
                for object_param in object_param_set:
                    new_type = "".join((object_param[1], "___", str(type_suffix)))
                    type_suffix += 1
                    add_type(types_processed, new_type, object_param[1])
                    add_object(objects_processed, object_param[2], new_type)
                    new_param_types[object_param[0]] = new_type
                    if object_param[1] in specific_generalized_objects:
                        for gen_obj in specific_generalized_objects[object_param[1]]:
                            add_object(objects_processed, gen_obj, new_type)
                for hidden_param_name in parameterizations[action_name][
                    object_param_set
                ]:
                    new_type = "".join((hidden_param_name, "___", str(type_suffix)))
                    type_suffix += 1
                    add_type(
                        types_processed, new_type, "position"
                    )  # Works because only positions can be hidden parameters for now
                    new_param_types[hidden_param_name] = new_type
                    for hidden_param_value in parameterizations[action_name][
                        object_param_set
                    ][hidden_param_name]:
                        add_object(objects_processed, hidden_param_value, new_type)
                new_params = [
                    [old_param_spec[0], new_param_types[old_param_spec[0]]]
                    for old_param_spec in action_descr["params"]
                ]

                new_action_name = "".join((action_name, "___", str(action_suffix)))
                action_suffix += 1
                actions_processed[new_action_name] = {
                    "params": new_params,
                    "preconds": action_descr["preconds"],
                    "effects": action_descr["effects"],
                }
        else:
            actions_processed[action_name] = deepcopy(action_descr)

    if joker_objects is not None and len(joker_objects) > 0:
        for new_type in types_processed:
            if not type_x_child_of_y(types_processed, new_type, "item"):
                continue
            for obj in joker_objects:
                add_object(objects_processed, obj, new_type)

    return actions_processed, types_processed, objects_processed


class PDDLFileInterface:
    def __init__(
        self,
        domain_dir,
        problem_dir=None,
        domain_name="",
        time_string=None,
        domain_file=None,
        readonly: bool = False,
    ):
        self._domain_dir = domain_dir
        self._problem_dir = problem_dir if problem_dir is not None else domain_dir
        self._domain_name = domain_name
        self._readonly = readonly

        self.domain_file_pddl = domain_file
        self.problem_file_pddl = None
        self._requirements = ":strips :typing"

        if time_string is None:
            time_now = datetime.now()
            self._time_now_str = time_now.strftime("%y%m%d-%H%M%S")
        else:
            self._time_now_str = time_string

    # ----- Loading and saving PDDL files --------------------------------------

    def write_pddl(
        self,
        knowledge_base,
        objects,
        initial_predicates,
        goals,
        joker_objects=None,
        specific_generalized_objects=None,
    ):
        if self._readonly:
            raise RuntimeError("Trying to write from readonly class")

        if specific_generalized_objects is None:
            specific_generalized_objects = dict()
        (actions_processed, types_processed, object_processed) = _preprocess_knowledge(
            knowledge_base.actions,
            objects,
            knowledge_base.types,
            knowledge_base.parameterizations,
            joker_objects,
            specific_generalized_objects,
        )

        # Write files
        self.write_domain_pddl(
            actions_processed, knowledge_base.predicate_definitions, types_processed
        )
        self.write_problem_pddl(object_processed, initial_predicates, goals)

    def write_domain_pddl(self, actions, predicates, types):
        if self._readonly:
            raise RuntimeError("Trying to write from readonly class")

        all_types_present = self.check_types(actions, types, predicates)
        if not all_types_present:
            raise ValueError("Not all types were defined properly")

        types_by_parent = invert_dict(types)

        pddl_str = ""
        pddl_str += "(define (domain " + self._domain_name + ")\n"
        pddl_str += "\t(:requirements " + self._requirements + ")\n\n"

        pddl_str += "\t(:types\n"
        if None in types_by_parent:
            pddl_str += "\t\t"
            for type_item in types_by_parent[None]:
                pddl_str += type_item + " "
            pddl_str += "- object\n"
        for parent_type in types_by_parent:
            if parent_type is None:
                continue
            pddl_str += "\t\t"
            for type_item in types_by_parent[parent_type]:
                pddl_str += type_item + " "
            pddl_str += "- " + parent_type + "\n"
        pddl_str += "\t)\n\n"

        pddl_str += "\t(:predicates\n"
        for pred in predicates:
            pddl_str += "\t\t(" + pred
            for item in predicates[pred]:
                pddl_str += " ?" + item[0] + " - " + item[1]
            pddl_str += ")\n"
        pddl_str += "\t)\n\n"
        for act in actions:
            pddl_str += "\t(:action " + act + "\n"
            pddl_str += "\t\t:parameters\n"
            pddl_str += "\t\t\t("
            for item in actions[act]["params"]:
                pddl_str += "?" + item[0] + " - " + item[1] + " "
            pddl_str = pddl_str[:-1]
            pddl_str += ")\n\n"
            pddl_str += "\t\t:precondition\n\t\t\t(and\n"
            for item in actions[act]["preconds"]:
                pddl_str += "\t\t\t\t("
                if not item[1]:
                    pddl_str += "not ("
                pddl_str += item[0]
                for param in item[2]:
                    pddl_str += " ?" + param
                if not item[1]:
                    pddl_str += ")"
                pddl_str += ")\n"
            pddl_str += "\t\t\t)\n\n"
            pddl_str += "\t\t:effect\n\t\t\t(and\n"
            for item in actions[act]["effects"]:
                pddl_str += "\t\t\t\t("
                if not item[1]:
                    pddl_str += "not ("
                pddl_str += item[0]
                for param in item[2]:
                    pddl_str += " ?" + param
                if not item[1]:
                    pddl_str += ")"
                pddl_str += ")\n"
            pddl_str += "\t\t\t)\n\n"
            pddl_str += "\t)\n\n"
        pddl_str += ")"

        new_filename = path.join(self._domain_dir, self._time_now_str + "_domain.pddl")
        with open(new_filename, "w") as f:
            f.write(pddl_str)
        self.domain_file_pddl = new_filename

    def read_domain_pddl(self):
        with open(self.domain_file_pddl, "r") as f:
            dom = f.read()
        dom = dom.split("\n")

        predicates = dict()
        actions = dict()
        types = dict()

        # Parse predicates
        while len(dom) > 0:
            curr = dom.pop(0)
            curr = curr.strip()
            if curr.find("(define") > -1:
                splitted = curr.split(" ")
                self._domain_name = splitted[-1][:-1]
            elif curr.find("(:types") > -1:
                while True:
                    sub_curr = dom.pop(0)
                    sub_curr = sub_curr.strip()
                    if sub_curr == ")":
                        break
                    line_split = sub_curr.split("-")
                    child_types = line_split[0].strip().split(" ")
                    parent_type = line_split[1].strip()
                    for child_type in child_types:
                        if child_type in types:
                            if parent_type not in types[child_type]:
                                types[child_type].append(parent_type)
                        else:
                            types[child_type] = [parent_type]
            elif curr.find("(:predicates") > -1:
                while True:
                    sub_curr = dom.pop(0)
                    sub_curr = sub_curr.strip()
                    if sub_curr == ")":
                        break
                    splitted = sub_curr.split(" ")
                    predicate = splitted.pop(0)[1:]
                    params = []
                    params_typed_idx = 0
                    while len(splitted) > 0:
                        param = splitted.pop(0)
                        param = param.replace(")", "")
                        if param == "-":
                            this_type = splitted.pop(0)
                            this_type = this_type.replace(")", "")
                            for i in range(len(params[params_typed_idx:])):
                                params[params_typed_idx + i][1] = this_type
                            params_typed_idx = len(params)
                        else:
                            param = param.replace("?", "")
                            params.append([param, None])
                    predicates[predicate] = params
            elif curr.find("(:action") > -1:
                action = curr.split(" ")[1]
                while True:
                    sub_curr = dom.pop(0)
                    sub_curr = sub_curr.strip()
                    if sub_curr == ")":
                        break
                    elif sub_curr.find(":parameters") > -1:
                        sub_curr = dom.pop(0)
                        sub_curr = sub_curr.strip()
                        splitted = sub_curr.split(" ")
                        params = []
                        params_typed_idx = 0
                        while len(splitted) > 0:
                            param = splitted.pop(0)
                            param = param.replace("(", "")
                            param = param.replace(")", "")
                            if param == "-":
                                this_type = splitted.pop(0)
                                this_type = this_type.replace(")", "")
                                for i in range(len(params[params_typed_idx:])):
                                    params[params_typed_idx + i][1] = this_type
                                params_typed_idx = len(params)
                            else:
                                param = param.replace("?", "")
                                params.append([param, None])
                    elif sub_curr.find(":precondition") > -1:
                        sub_curr = dom.pop(0).strip()
                        preconds = []
                        if sub_curr == "(and":
                            sub_curr = dom.pop(0).strip()
                        while len(sub_curr) > 0:
                            if sub_curr == ")":
                                break
                            sub_curr = sub_curr.replace("(", "")
                            sub_curr = sub_curr.replace(")", "")
                            splitted = sub_curr.split(" ")
                            first_token = splitted.pop(0)
                            if first_token == "not":
                                negated = True
                                precond_name = splitted.pop(0)
                            else:
                                negated = False
                                precond_name = first_token
                            assert precond_name in list(predicates.keys())
                            precond_params = []
                            while len(splitted) > 0:
                                param = splitted.pop(0)
                                param = param.replace("?", "")
                                precond_params.append(param)
                            preconds.append((precond_name, not negated, precond_params))
                            sub_curr = dom.pop(0).strip()
                    elif sub_curr.find(":effect") > -1:
                        sub_curr = dom.pop(0).strip()
                        effects = []
                        if sub_curr == "(and":
                            sub_curr = dom.pop(0).strip()
                        while len(sub_curr) > 0:
                            if sub_curr == ")":
                                break
                            sub_curr = sub_curr.replace("(", "")
                            sub_curr = sub_curr.replace(")", "")
                            splitted = sub_curr.split(" ")
                            first_token = splitted.pop(0)
                            if first_token == "not":
                                negated = True
                                effect_name = splitted.pop(0)
                            else:
                                negated = False
                                effect_name = first_token
                            assert effect_name in list(predicates.keys())
                            effect_params = []
                            while len(splitted) > 0:
                                param = splitted.pop(0)
                                param = param.replace("?", "")
                                effect_params.append(param)
                            effects.append((effect_name, not negated, effect_params))
                            sub_curr = dom.pop(0).strip()
                actions[action] = {
                    "params": params,
                    "preconds": preconds,
                    "effects": effects,
                }
        print("Read PDDL domain file")
        return predicates, actions, types

    def write_problem_pddl(self, objects, initial_predicates, goals):
        if self._readonly:
            raise RuntimeError("Trying to write from readonly class")

        pddl_str = ""
        pddl_str += "(define (problem chimera-auto-problem)\n"

        pddl_str += "\t(:domain\n"
        pddl_str += "\t\t" + self._domain_name + "\n"
        pddl_str += "\t)\n\n"

        pddl_str += "\t(:objects\n"
        for obj in objects:
            for obj_type in objects[obj]:
                pddl_str += "\t\t" + obj + " - " + obj_type + "\n"
        pddl_str += "\t)\n\n"

        if len(initial_predicates) > 0:
            pddl_str += "\t(:init\n"
            for init in initial_predicates:
                pddl_str += "\t\t("
                for it in init:
                    pddl_str += it + " "
                pddl_str = pddl_str[:-1]
                pddl_str += ")\n"
            pddl_str += "\t)\n\n"

        pddl_str += "\t(:goal\n"
        pddl_str += "\t\t(and\n"
        for g in goals:
            pddl_str += "\t\t\t("
            if not g[1]:
                # Negated
                pddl_str += "not ("
            pddl_str += g[0] + " "
            for it in g[2]:
                pddl_str += it + " "
            pddl_str = pddl_str[:-1]
            if not g[1]:
                pddl_str += ")"
            pddl_str += ")\n"
        pddl_str += "\t\t)\n"
        pddl_str += "\t)\n"

        pddl_str += ")\n"

        new_filename = path.join(
            self._problem_dir, self._time_now_str + "_problem.pddl"
        )
        with open(new_filename, "w") as f:
            f.write(pddl_str)
        # print("Wrote new PDDL problem file: " + new_filename.split("/")[-1])
        self.problem_file_pddl = new_filename

    # ----- Helper functions ---------------------------------------------------

    @staticmethod
    def check_types(actions, types, predicates):
        # Makes sure that all type were defined
        for pred in predicates:
            for item in predicates[pred]:
                if item[1] not in types:
                    print("The following type was not pre-defined: {}".format(item[1]))
                    return False
        for act in actions:
            for item in actions[act]["params"]:
                if item[1] not in types:
                    print("The following type was not pre-defined: {}".format(item[1]))
                    return False
        return True


### Assumed conventions:
# All arguments of a predicate are on the same line as the predicate name. Each line defines one predicate.
# For actions, all parameters are in the same line, starting below the :parameters keyword.
# For preconditions and effects, one is defined per line, starting on the line after "(and".
# After preconditions and effects, a blank line is expected.
# Lines starting with ';' are comments.
