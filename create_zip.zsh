#!/usr/bin/env zsh

mkdir /tmp/highlevel_planning

cp -r ./highlevel_planning_ros /tmp/highlevel_planning/
cp Dockerfile /tmp/highlevel_planning/
cp envfile /tmp/highlevel_planning/
cp README.md /tmp/highlevel_planning/

DEST_PATH=$(pwd)

cd /tmp
zip -r highlevel_planning.zip highlevel_planning
mv highlevel_planning.zip $DEST_PATH/

rm -r /tmp/highlevel_planning
