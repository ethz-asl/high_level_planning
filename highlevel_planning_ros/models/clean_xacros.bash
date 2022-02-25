#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

trash "$DIR"/cupboard_drawers/cupboard_drawers.urdf
trash "$DIR"/cupboard2/cupboard2.urdf
trash "$DIR"/container/container_no_lid.urdf
trash "$DIR"/container/lid.urdf
trash "$DIR"/container/container_sliding_lid.urdf
trash "$DIR"/box_panda_hand_pb.urdf
trash "$DIR"/shelf/shelf.urdf
