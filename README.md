# High Level Planning

This is the code implementing symbolic skill set extension for an agent relying on symbolic planning.

There are two options to run the code: either natively or in a docker container (no GUI available).

## Run Natively

For this, please follow the instructions in [`highlevel_planning_ros/README.md`](./highlevel_planning_ros/README.md).

## Run in Docker

For this, a copy of Docker Desktop needs to be installed on your machine (https://www.docker.com/get-started).

Download version 2.1 of the Metric-FF symbolic planner from https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html, extract it, and place it next to this file, such that the directory `high_level_planning/Metric-FF-v2.1` contains the Metric-FF source code.

Build the Docker container:

```bash
docker build -t hlp_docker .
```

Launch the container:

```bash
docker run -it --rm hlp_docker
```

You should end up in a bash shell within the container. From here, our exploration algorithm can be run (only in `direct` mode since no GUI is available from within the container):

```bash
python3 scripts/run_auto.py -m direct
```

More details on changing configurations, etc. can be found in [the other readme](./highlevel_planning_ros/README.md#run).
