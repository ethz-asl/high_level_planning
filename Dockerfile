FROM ros:noetic-ros-core

RUN apt-get update && apt-get upgrade -y

RUN mkdir -p /catkin_ws/src
COPY highlevel_planning_ros /catkin_ws/src/highlevel_planning_ros

WORKDIR /catkin_ws/src/highlevel_planning_ros

RUN ./install_requirements.sh
RUN pip install -r requirements.txt

RUN mkdir -p /build
COPY Metric-FF-v2.1 /build/metric-ff
WORKDIR /build/metric-ff
RUN apt-get install -y flex bison
RUN make
RUN cp ff /catkin_ws/src/highlevel_planning_ros/bin/

WORKDIR /catkin_ws
RUN . /opt/ros/noetic/setup.sh && catkin init
RUN . /opt/ros/noetic/setup.sh && catkin config -DCMAKE_BUILD_TYPE=Release
RUN . /opt/ros/noetic/setup.sh && catkin build highlevel_planning_ros
RUN . /opt/ros/noetic/setup.sh && cd src/highlevel_planning_ros/models && ./parse_xacros.bash

COPY envfile /root/.bashrc
ENTRYPOINT /usr/bin/bash
