cmake_minimum_required(VERSION 3.0.2)
project(highlevel_planning_ros)

add_compile_options(-std=c++11)

find_package(catkin QUIET COMPONENTS)

if (catkin_FOUND)
    catkin_python_setup()
    catkin_package()
endif()
