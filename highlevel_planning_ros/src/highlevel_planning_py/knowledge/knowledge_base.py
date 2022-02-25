import pickle
from copy import deepcopy
from os import path, makedirs
from highlevel_planning_py.pddl_interface.pddl_file_if import PDDLFileInterface
from highlevel_planning_py.pddl_interface import planner_interface
from highlevel_planning_py.exploration.logic_tools import (
    determine_relevant_predicates,
    measure_predicates,
)
from highlevel_planning_py.exploration.exploration_tools import get_items_closeby


def check_path_exists(path_to_check):
    if not path.isdir(path_to_check):
        makedirs(path_to_check)


class KnowledgeBase:
    def __init__(
        self, paths, domain_name="", time_string=None, domain_file="_domain.pkl"
    ):
        self.predicate_funcs = None

        # Folder book keeping
        self.bin_dir = paths["bin_dir"]
        self.knowledge_dir = path.join(paths["data_dir"], "knowledge", domain_name)
        self.domain_dir = path.join(self.knowledge_dir, "main")
        check_path_exists(self.domain_dir)
        problem_dir = self.domain_dir
        temp_domain_dir = path.join(self.knowledge_dir, "explore")
        check_path_exists(temp_domain_dir)
        temp_problem_dir = temp_domain_dir

        # Domain definition
        self.domain_name = domain_name
        self.predicate_definitions = dict()
        self.actions = dict()
        self.types = dict()

        # Problem definition
        self.objects = dict()
        self.visible_objects = set()
        self.object_predicates = set()
        self.initial_state_predicates = set()
        self.goals = list()

        # Value lookups (e.g. for positions)
        self.lookup_table = dict()
        self.parameterizations = dict()
        # Meta action info
        self.meta_actions = dict()

        # Load previous knowledge base
        self._domain_file_path = path.join(self.domain_dir, domain_file)
        self._domain_file_name = domain_file.split(".")[0]
        print(f"Using domain file {self._domain_file_path}")
        self.load_domain()

        # PDDL file interfaces
        self.pddl_if = PDDLFileInterface(
            self.domain_dir, problem_dir, domain_name, time_string
        )
        self.pddl_if_temp = PDDLFileInterface(
            temp_domain_dir, temp_problem_dir, domain_name, time_string
        )

        # Temporary variables (e.g. for exploration)
        self._temp_objects = dict()
        self._temp_object_predicates = set()
        self._temp_generalized_objects = set()

    def duplicate(self, original):
        self.domain_name = deepcopy(original.domain_name)
        self.predicate_definitions = deepcopy(original.predicate_definitions)
        self.actions = deepcopy(original.actions)
        self.types = deepcopy(original.types)
        self.objects = deepcopy(original.objects)
        self.visible_objects = deepcopy(original.visible_objects)
        self.object_predicates = deepcopy(original.object_predicates)
        self.initial_state_predicates = deepcopy(original.initial_state_predicates)
        self.goals = deepcopy(original.goals)
        self.lookup_table = deepcopy(original.lookup_table)
        self.parameterizations = deepcopy(original.parameterizations)
        self.meta_actions = deepcopy(original.meta_actions)
        self._temp_objects = deepcopy(original._temp_objects)
        self._temp_object_predicates = deepcopy(original._temp_object_predicates)
        self._temp_generalized_objects = deepcopy(original._temp_generalized_objects)

        # -------------

        self.pddl_if = None
        self.pddl_if_temp = None

    def set_predicate_funcs(self, preds):
        self.predicate_funcs = preds

    # ----- Loading and saving pickle with domain and problem ---------------------------

    def load_domain(self):
        print("Trying to load domain file...")
        if path.exists(self._domain_file_path):
            with open(self._domain_file_path, "rb") as f:
                load_obj = pickle.load(f)
            self.domain_name = load_obj[0]
            self.predicate_definitions = load_obj[1]
            self.actions = load_obj[2]
            self.types = load_obj[3]
            self.objects = load_obj[4]
            self.lookup_table = load_obj[5]
            self.parameterizations = load_obj[6]
            self.meta_actions = load_obj[7]
            print("Trying to load domain file... DONE")
        else:
            print("Trying to load domain file... NOT FOUND --> starting from scratch")

    def save_domain(self, filename_appendix=""):
        save_obj = (
            self.domain_name,
            self.predicate_definitions,
            self.actions,
            self.types,
            self.objects,
            self.lookup_table,
            self.parameterizations,
            self.meta_actions,
        )
        file_name = self._domain_file_name + filename_appendix + ".pkl"
        file_path = path.join(self.domain_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(save_obj, f)
        print("Saved domain file")

    # ----- Adding to the domain description ------------------------------------

    def add_action(
        self, action_name, action_definition, overwrite=False, rename_if_exists=False
    ):
        if not overwrite and action_name in self.actions:
            if rename_if_exists:
                i = 1
                while True:
                    new_name = action_name + "-" + str(i)
                    i += 1
                    if new_name not in self.actions:
                        break
                action_name = new_name
            else:
                print(
                    "Action "
                    + action_name
                    + " already exists and no overwrite was requested. Ignoring request."
                )
                return False
        assert isinstance(action_name, str)
        for field in ["params", "preconds", "effects", "exec_ignore_effects"]:
            assert field in action_definition
        self.actions[action_name] = action_definition
        return action_name

    def add_predicate(self, predicate_name, predicate_definition, overwrite=False):
        if not overwrite and predicate_name in self.predicate_definitions:
            print(
                "Predicate "
                + predicate_name
                + " already exists and no overwrite was requested. Ignoring request."
            )
        else:
            assert isinstance(predicate_name, str)
            self.predicate_definitions[predicate_name] = predicate_definition

    def add_artificial_predicate(self):
        counter = 1
        while True:
            predicate_name = "artificial_predicate_{}".format(counter)
            if predicate_name not in self.predicate_definitions:
                break
            counter += 1
        predicate_definition = []
        self.add_predicate(predicate_name, predicate_definition, overwrite=False)
        return predicate_name

    def add_type(self, new_type, parent_type=None):
        assert isinstance(new_type, str)
        if new_type in self.types:
            # There can only be one parent per type
            ValueError("Type already exists")
        else:
            if parent_type is not None:
                assert parent_type in self.types
            self.types[new_type] = parent_type

    # ----- Adding to the problem description ----------------------------------

    def add_object(self, object_name, object_type, object_value=None):
        assert object_type in self.types
        if object_name in self.objects:
            if object_type not in self.objects[object_name]:
                self.objects[object_name].append(object_type)
                red_types = self._get_redundant_type(self.objects[object_name])
                for red_type in red_types:
                    if self.objects[object_name].index(red_type) != 0:
                        self.objects[object_name].remove(red_type)
        else:
            self.objects[object_name] = [object_type]
        if object_value is not None:
            self.lookup_table[object_name] = object_value
        self.visible_objects.add(object_name)

    def _get_redundant_type(self, object_types):
        redundant_types = set()
        for i, obj_type in enumerate(object_types):
            for potential_child in (x for j, x in enumerate(object_types) if j != i):
                if self.type_x_child_of_y(potential_child, obj_type):
                    redundant_types.add(obj_type)
                    break
        return list(redundant_types)

    def add_objects(self, object_dict):
        for obj, obj_type in object_dict.items():
            self.add_object(obj, obj_type)

    def add_goal(self, goal_list):
        self.goals += goal_list

        # Remove duplicates
        self.goals = list(dict.fromkeys(self.goals))

    def set_goals(self, goal_list):
        self.goals = goal_list
        self.goals = list(set(self.goals))

    # ----- Solving ------------------------------------------------------------

    def solve(self):
        self.save_domain()
        self.pddl_if.write_pddl(
            self,
            self.objects,
            self.object_predicates.union(self.initial_state_predicates),
            self.goals,
        )
        return planner_interface.pddl_planner(
            self.pddl_if.domain_file_pddl,
            self.pddl_if.problem_file_pddl,
            self.actions,
            self.bin_dir,
        )

    # ----- Meta action handling -----------------------------------------------

    def expand_step(self, action_name, action_parameters):
        expanded_step = list()
        if action_name in self.meta_actions:
            meta_action = self.meta_actions[action_name]
            for idx, sub_action_name in enumerate(meta_action["seq"]):
                new_plan_item = [sub_action_name, {}]
                sub_action_parameters = self.actions[sub_action_name]["params"]
                for param_spec in sub_action_parameters:
                    old_param_name = param_spec[0]
                    if old_param_name in meta_action["hidden_params"][idx]:
                        new_plan_item[1][old_param_name] = meta_action["hidden_params"][
                            idx
                        ][old_param_name]
                    elif old_param_name in meta_action["param_translator"][idx]:
                        new_param_name = meta_action["param_translator"][idx][
                            old_param_name
                        ]
                        new_plan_item[1][old_param_name] = action_parameters[
                            new_param_name
                        ]
                    else:
                        raise RuntimeError(
                            f"Parameter '{old_param_name}' for sub action '{sub_action_name}'"
                            f" of meta action '{action_name}' undefined. Param translator of"
                            f" meta action: {meta_action['param_translator']}"
                        )
                expanded_step.append(new_plan_item)
        else:
            new_plan_item = [action_name, action_parameters]
            expanded_step.append(new_plan_item)
        return expanded_step

    def add_meta_action(
        self,
        name: str,
        sequence: list,
        parameters,
        param_translator,
        hidden_parameters,
        description,
    ):
        self.meta_actions[name] = {
            "seq": sequence,
            "params": parameters,
            "param_translator": param_translator,
            "hidden_params": hidden_parameters,
            "description": description,
        }

    # ----- Utilities ----------------------------------------------------------

    def test_goals(self):
        for goal in self.goals:
            if not self.predicate_funcs.call[goal[0]](*goal[2]) == goal[1]:
                return False
        return True

    def check_predicates(self, scene_objects, robot_uid, pb_client_id):
        """
        If predicates need to be initialized when the system is launched, this can be done here.
        """

        if self.predicate_funcs.empty_hand("robot1"):
            self.initial_state_predicates.add(("empty-hand", "robot1"))
        self.initial_state_predicates.add(("in-reach", "origin", "robot1"))
        self.initial_state_predicates.add(("at", "origin", "robot1"))

        # Check any predicates in relation with the goal
        relevant_objects = set()
        for goal in self.goals:
            if self.predicate_funcs.call[goal[0]](*goal[2]):
                pred_tuple = (goal[0],) + goal[2]
                self.initial_state_predicates.add(pred_tuple)

            relevant_objects.update(goal[2])

        if "robot1" in relevant_objects:
            relevant_objects.remove("robot1")

        closeby_objects = get_items_closeby(
            relevant_objects, scene_objects, pb_client_id, robot_uid, distance_limit=1.0
        )
        relevant_objects.update(closeby_objects)

        relevant_predicates = determine_relevant_predicates(
            relevant_objects, self, ignore_predicates=["at"]
        )
        measured_predicates = measure_predicates(relevant_predicates, self)
        for i in range(len(relevant_predicates)):
            if measured_predicates[i]:
                pred_tuple = (relevant_predicates[i][0],) + relevant_predicates[i][1]
                self.initial_state_predicates.add(pred_tuple)

    def populate_visible_objects(self, scene):
        for obj in scene.objects:
            self.add_object(obj, "item")
            for grasp in ["grasp0", "grasp1"]:
                if self.predicate_funcs.call["has-grasp"](obj, grasp):
                    self.object_predicates.add(("has-grasp", obj, grasp))
                    self.add_object(obj, "item-graspable")

        # Add "objects" that are always visible
        for object_name in self.objects:
            for object_type in self.objects[object_name]:
                if self.type_x_child_of_y(object_type, "position"):
                    self.add_object(object_name, "position")
                    self.object_predicates.add(("has-grasp", object_name, "grasp0"))
                    break

    def type_x_child_of_y(self, x, y):
        parent_type = self.types[x]
        if x == y:
            return True
        elif parent_type is None:
            return False
        return self.type_x_child_of_y(parent_type, y)

    def is_type(self, object_to_check, type_query):
        if object_to_check in self.objects:
            obj_types = self.objects[object_to_check]
        else:
            obj_types = self._temp_objects[object_to_check]
        for obj_type in obj_types:
            if self.type_x_child_of_y(obj_type, type_query):
                return True
        return False

    def get_objects_by_type(
        self,
        type_query,
        types_by_parent,
        objects_by_type,
        object_set=None,
        visible_only=False,
        include_generalized_objects=False,
    ):
        if object_set is None:
            object_set = set()
        if type_query in objects_by_type:
            for obj in objects_by_type[type_query]:
                if not visible_only or obj in self.visible_objects:
                    object_set.add(obj)
        if type_query in types_by_parent:
            for sub_type in types_by_parent[type_query]:
                object_set = self.get_objects_by_type(
                    sub_type,
                    types_by_parent,
                    objects_by_type,
                    object_set,
                    visible_only=visible_only,
                )
        if include_generalized_objects:
            for obj in self._temp_generalized_objects:
                if not visible_only or obj in self.visible_objects:
                    object_set.add(obj)
        return object_set

    # ----- Handling temporary goals, e.g. for exploration ---------------------

    def add_temp_object(self, object_type, object_name=None, object_value=None):
        assert object_type in self.types
        if object_name is not None:
            assert object_name not in self.lookup_table
        else:
            counter = 1
            while True:
                object_name = "{}_sample_{}".format(object_type, counter)
                if (
                    object_name not in self.objects
                    and object_name not in self._temp_objects
                ):
                    break
                counter += 1
        if object_name in self._temp_objects:
            if object_type not in self._temp_objects[object_name]:
                self._temp_objects[object_name].append(object_type)
                red_types = self._get_redundant_type(self._temp_objects[object_name])
                for red_type in red_types:
                    if self._temp_objects[object_name].index(red_type) != 0:
                        self._temp_objects[object_name].remove(red_type)
        else:
            self._temp_objects[object_name] = [object_type]
            if self.is_type(object_name, "position"):
                self._temp_object_predicates.add(("has-grasp", object_name, "grasp0"))
        if object_value is not None:
            self.lookup_table[object_name] = object_value
        return object_name

    def remove_temp_object(self, object_name):
        if object_name in self.lookup_table:
            del self.lookup_table[object_name]
        if object_name in self._temp_objects:
            del self._temp_objects[object_name]

    def generalize_temp_object(self, object_name):
        assert object_name in self.objects
        self._temp_generalized_objects.add(object_name)

    def make_permanent(self, obj_name):
        self.objects[obj_name] = self._temp_objects[obj_name]
        del self._temp_objects[obj_name]
        for pred in self._temp_object_predicates:
            if obj_name in pred:
                self.object_predicates.add(pred)

    def joined_objects(self):
        objects = deepcopy(self._temp_objects)
        for obj in self.objects:
            if obj in objects:
                objects[obj] = list(dict.fromkeys(self.objects[obj] + objects[obj]))
            else:
                objects[obj] = self.objects[obj]
        return objects

    def solve_temp(
        self, goals, initial_predicates=None, specific_generalized_objects=None
    ):
        objects = self.joined_objects()
        if initial_predicates is None:
            initial_predicates = self.initial_state_predicates
        self.pddl_if_temp.write_pddl(
            self,
            objects,
            self.object_predicates.union(self._temp_object_predicates).union(
                initial_predicates
            ),
            goals,
            self._temp_generalized_objects,
            specific_generalized_objects,
        )
        return planner_interface.pddl_planner(
            self.pddl_if_temp.domain_file_pddl,
            self.pddl_if_temp.problem_file_pddl,
            self.actions,
            self.bin_dir,
        )

    def clear_temp_samples(self):
        objects_to_remove = set()
        preds_to_remove = set()
        for obj in self._temp_objects:
            if obj in self.lookup_table:
                objects_to_remove.add(obj)
                del self.lookup_table[obj]
                for pred in self._temp_object_predicates:
                    if obj in pred:
                        preds_to_remove.add(pred)
        for obj in objects_to_remove:
            del self._temp_objects[obj]
        for pred in preds_to_remove:
            self._temp_object_predicates.remove(pred)

    def clear_temp(self):
        self._temp_object_predicates.clear()
        self.clear_temp_samples()
        self._temp_generalized_objects.clear()
        self._temp_objects.clear()
