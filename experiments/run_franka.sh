#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DEFORMLABSIM_ROOT=${DEFORMLABSIM_ROOT:-"$here/../runtime"}
export DEFORMLABSIM_ROOT=$(realpath "$DEFORMLABSIM_ROOT")
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Isaac Sim python.sh path}"
: "${ISAAC_BUILD_RECEIPT:?Set ISAAC_BUILD_RECEIPT to the local runtime build receipt}"
: "${CONTACT_BUILD_DIR:?Set CONTACT_BUILD_DIR to the locally built contact variants}"
case_id=${1:?Provide the case ID, for example F01}
method=${2:?Provide the method ID, B, D, or Dense}
[[ $case_id =~ ^F(0[1-9]|1[0-2])$ ]]
variant=sparse
surface=persistent
case "$method" in
    B) surface=off ;;
    D) ;;
    Dense) variant=dense ;;
    *) exit 2 ;;
esac
spec="$here/franka/cases/$case_id.json"
offset=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["offset_x_m"])' "$spec")
export FRANKA_MAPPING_LIB=$(realpath "$CONTACT_BUILD_DIR/$variant/libphysx_robot_surface.so")
export FRANKA_ROWS_PTX=$(realpath "$CONTACT_BUILD_DIR/$variant/rows.ptx")
[[ -f $FRANKA_MAPPING_LIB && -f $FRANKA_ROWS_PTX ]]
out=$(realpath -m "${OUTPUT_DIR:-outputs/franka/$case_id-$method}")
mkdir -p -- "$(dirname -- "$out")"
if [[ -e $out ]]; then
    printf 'Output already exists: %s\n' "$out" >&2
    exit 2
fi
export LD_PRELOAD="$FRANKA_MAPPING_LIB"
exec "$ISAAC_PYTHON" "$here/franka/isaac_volume_grasp.py" \
    --output-dir "$out" --build-receipt "$ISAAC_BUILD_RECEIPT" \
    --physics-hz 960 --position-iterations 32 --robot-collision on \
    --cp-conservative-trial split-velocity --no-captures --zero-settling \
    --surface-runtime "$surface" --block-offset-x "$offset" --visual-controller rgbd \
    --actuation-offset-m 0 --goal-dir "$here/franka" --case "$spec"
