# highlevel_planning_ros

## Setup Instructions

Tested on Ubuntu 20.04 with ROS Noetic. This code does NOT run on Ubuntu 18.04 with ROS Melodic, because python3 is required.

### ROS workspace

This package needs to be built in a ROS workspace. From now on, we assume that it is called `~/catkin_ws`. Move this directory (`highlevel_planning_ros`) into `~/catkin_ws/src`.

### Dependencies

Install ROS according to the following instructions: https://wiki.ros.org/noetic/Installation

Install system dependencies:

```bash
./install_requirements.sh
```

Install python dependencies (in a virtual environment if desired):

```bash
pip3 install -r requirements.txt
```

A symbolic planner is needed to run this code. Specifically, we use MetricFF 2.1. Download the code here: https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html.
Then compile it (simply by running `make`) and place the resulting binary called `ff` into the directory `highlevel_planning_ros/bin`.

### Install Development Code

Build the package:

```
catkin build -DCMAKE_BUILD_TYPE=Release highlevel_planning_ros
```

### Generate URDFs

To generate the required URDFs, run the command

```bash
cd highlevel_planning_ros/models
./parse_xacros.bash
```

## Run

To run any command, make sure that your ROS distribution, the ROS workspace, as well as (if applicable) the virtual environment are sourced.

User inputs such as goal and (optionally) demonstration can be configured in `config/main.xml` (for the _rearrangement task domain_) or `config/main_pddl_bm.yaml` (for the _PDDL benchmark domain_).

Running the code will create the directory `~/Data/highlevel_planning`. Make sure that there is nothing there yet, to avoid that files are overwritten.

The following table shows the commands required to run our algorithm or the MCTS baseline on the _PDDL benchmark domain_ or the _rearrangement task domain_ respectively:

| Domain                    | Algorithm | Command                                      |
|---------------------------|-----------|----------------------------------------------|
| PDDL benchmark domain     | ours      | `python3 scripts/run_pddl_benchmark_ours.py` |
| PDDL benchmark domain     | MCTS      | `python3 scripts/run_pddl_benchmark_mcts.py` |
| Rearrangement task domain | ours      | `python3 scripts/run_auto.py`                |
| Rearrangement task domain | MCTS      | `python3 scripts/run_mcts.py`                |

For command line options, pass `--help` to any of the commands above.

## Documentation

### Representation of planning problem in Python

The file `highlevel_planning_py/pddl_interface/pddl_file_if.py` contains the code to parse PDDL files and fill the definitions into python data structures, safe and load the python data structures (using pickle), and write modified data structures to new PDDL files. In the following, the layout of the relevant Python data structures is defined. 

**Predicates**

```python
{
    "<predicate1>": [["<param1>", "<type1>"], ["<param2>", "<type2>"], ...],
    "<predicate2>": [...],
    ...
}
```

**Actions**

```python
{
    "<action1>":
    {
        "params": [["<param1>", "<type1>"], ["<param2>", "<type2>"], ...],
        "preconds": [
            ("<predicate1>", <true/false>, ["<param1>", "<param2>", ...]),
        	("<predicate2>", <true/false>, ["<param8>", "<param6>", ...]),
            ...
        ],
        "effects": [
            ("<predicate1>", <true/false>, ["<param1>", "<param2>", ...]),
        	("<predicate7>", <true/false>, ["<param2>", "<param4>", ...]),
            ...
        ]
    },
    "<action2>": ...
}
```

The variable `<true/false>` is true if the predicate must hold before the action can be run (for preconditions) or does hold after the action was run (for effects).

**Objects**

```python
[
    ("<object_label1>", "<type1>"),
    ("<object_label2>", "<type2>"),
    ("<object_label3>", "<type3>")
]
```

**Initial Predicates and Goals**

```python
[
    ("<predicate1>", "<param1>", "<param2>", ...),
    ("<predicate2>", "<param1>", "<param2>", ...),
    ("<predicate3>", "<param1>", "<param2>", ...)
]
```

**Type Definitions**

```python
{
    "<type_name1>": "<parent_type1>",
    "<type_name2>": "<parent_type2>",
    ...
}
```

### Internal Representations

**Objects**

```python
knowledge_base.objects = {
    "<object_label1>": ["<base_type1>", "<type2>"],
    "<object_label2>": ["<base_type2>", "<type3>"],
}
```
