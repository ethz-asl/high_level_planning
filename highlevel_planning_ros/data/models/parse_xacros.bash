#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OUTDIR=parsed_xacros
mkdir -p $DIR/$OUTDIR
xacro "$DIR"/cupboard_drawers/cupboard_drawers.urdf.xacro > "$DIR"/"$OUTDIR"/cupboard_drawers.urdf
xacro "$DIR"/cupboard2/cupboard2.urdf.xacro > "$DIR"/"$OUTDIR"/cupboard2.urdf
cp "$DIR"/cupboard2/corpus.stl "$DIR"/"$OUTDIR"/corpus.stl
cp "$DIR"/cupboard2/drawer.stl "$DIR"/"$OUTDIR"/drawer.stl
xacro "$DIR"/container/container_no_lid.urdf.xacro > "$DIR"/"$OUTDIR"/container_no_lid.urdf
xacro "$DIR"/container/lid.urdf.xacro > "$DIR"/"$OUTDIR"/lid.urdf
xacro "$DIR"/container/container_sliding_lid.urdf.xacro > "$DIR"/"$OUTDIR"/container_sliding_lid.urdf
xacro "$DIR"/box_panda_hand.urdf.xacro > "$DIR"/"$OUTDIR"/box_panda_hand_pb.urdf
xacro "$DIR"/shelf/shelf.urdf.xacro > "$DIR"/"$OUTDIR"/shelf.urdf
FRANKA_DESCR_PATH="$(rospack find franka_description | sed 's./.\\\/.g')"
sed -i "s/package:\/\/franka_description/$FRANKA_DESCR_PATH/g" "$DIR"/"$OUTDIR"/box_panda_hand_pb.urdf
