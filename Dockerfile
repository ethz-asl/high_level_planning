FROM ros:noetic-ros-core

RUN apt-get update && apt-get upgrade -y

# Satisfy requirements
COPY highlevel_planning_ros/install_requirements.sh /install_requirements.sh
COPY highlevel_planning_ros/requirements.txt /requirements.txt
RUN /install_requirements.sh
RUN pip install -r requirements.txt

# Build symbolic planner
RUN mkdir -p /build
COPY Metric-FF-v2.1 /build/metric-ff
WORKDIR /build/metric-ff
RUN apt-get install -y flex bison
RUN make

# Copy code
RUN mkdir -p /catkin_ws/src
COPY highlevel_planning_ros /catkin_ws/src/highlevel_planning_ros
WORKDIR /catkin_ws/src/highlevel_planning_ros
RUN cp /build/metric-ff/ff /catkin_ws/src/highlevel_planning_ros/bin/

# Build hlp code
WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && catkin init
RUN . /opt/ros/noetic/setup.sh && catkin config -DCMAKE_BUILD_TYPE=Release
RUN . /opt/ros/noetic/setup.sh && catkin build highlevel_planning_ros
RUN . /opt/ros/noetic/setup.sh && cd src/highlevel_planning_ros/models && ./parse_xacros.bash

# Entrypoint
COPY envfile /root/.bashrc
ENTRYPOINT /usr/bin/bash
