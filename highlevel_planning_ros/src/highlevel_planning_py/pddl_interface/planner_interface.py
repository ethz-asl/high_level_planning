import subprocess
import os
from highlevel_planning_py.exploration.logic_tools import parse_plan


def pddl_planner(domain_file, problem_file, action_specs, bin_dir, debug_print=False):
    try:
        res = subprocess.check_output(
            [
                os.path.join(bin_dir, "ff"),
                "-s",
                "0",
                "-o",
                domain_file,
                "-f",
                problem_file,
            ]
        )
        if type(res) is not str:
            res = res.decode("utf-8")
    except subprocess.CalledProcessError as e:
        # Check if empty plan solves it
        output = e.output if type(e.output) is str else e.output.decode("utf-8")
        empty_idx = output.find("The empty plan solves it")
        if empty_idx > -1:
            # print("Empty plan solves the goal.")
            return []
        else:
            if debug_print:
                print("Planning failed: ")
                print(output)
            return False
    try:
        res = cut_string_before(res, "ff: found legal plan as follows", complain=True)
    except NameError:
        # print("Planning failed: ")
        # print(res)
        return False
    try:
        res = cut_string_before(res, "0:", complain=True)
    except NameError:
        # Empty plan solves this problem
        return []
    res = cut_string_at(res, "time spent")
    res = res.split("\n")
    for i in range(len(res)):
        res[i] = res[i].strip().lower()
    while True:
        try:
            res.remove("")
        except ValueError:
            break
    # print(res)
    sequence, parameters = parse_plan(res, action_specs)
    return sequence, parameters


def cut_string_before(string, query, complain=False):
    # Finds query in string and cuts everything before it.
    start_idx = string.find(query)
    if start_idx > -1:
        string = string[start_idx:]
    elif complain:
        raise NameError("Query not found")
    return string


def cut_string_at(string, query):
    # Finds query in string and cuts it away, together with everything that comes behind.
    start_idx = string.find(query)
    if start_idx > -1:
        string = string[:start_idx]
    return string
