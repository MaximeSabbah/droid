#!/usr/bin/env bash
set -euo pipefail

CONDA_HOOK="${DROID_CONDA_HOOK:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${DROID_POLYMETIS_CONDA_ENV:-polymetis-local}"

if [[ -n "${DROID_ENV_ACTIVATE:-}" ]]; then
    set +u
    eval "$DROID_ENV_ACTIVATE"
    set -u
elif [[ -f "$CONDA_HOOK" ]]; then
    source "$CONDA_HOOK"
    set +u
    conda activate "$CONDA_ENV"
    set -u
elif command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    set +u
    micromamba activate "$CONDA_ENV"
    set -u
else
    echo "Could not activate $CONDA_ENV. Set DROID_ENV_ACTIVATE or DROID_CONDA_HOOK." >&2
    exit 1
fi

pkill -9 run_server || true
pkill -9 franka_panda_cl || true
pkill -9 franka_panda_client || true

if [[ -z "${DROID_ROBOT_LAUNCH_CMD:-}" ]]; then
    if ! command -v launch_robot.py >/dev/null 2>&1; then
        echo "launch_robot.py was not found in the active environment: $CONDA_ENV" >&2
        exit 1
    fi

    if [[ -n "${DROID_POLYMETIS_CONFIG_DIR:-}" ]]; then
        if [[ ! -d "$DROID_POLYMETIS_CONFIG_DIR" ]]; then
            echo "DROID_POLYMETIS_CONFIG_DIR does not exist: $DROID_POLYMETIS_CONFIG_DIR" >&2
            exit 1
        fi
        DROID_ROBOT_CLIENT_CONFIG="${DROID_ROBOT_CLIENT_CONFIG:-droid_franka_hardware}"
        DROID_ROBOT_MODEL_CONFIG="${DROID_ROBOT_MODEL_CONFIG:-droid_franka_panda}"
        DROID_ROBOT_LAUNCH_CMD="launch_robot.py --config-dir ${DROID_POLYMETIS_CONFIG_DIR} robot_client=${DROID_ROBOT_CLIENT_CONFIG} robot_model=${DROID_ROBOT_MODEL_CONFIG}"
    else
        DROID_ROBOT_LAUNCH_CMD="launch_robot.py robot_client=franka_hardware"
    fi
fi

echo "Launching robot with: $DROID_ROBOT_LAUNCH_CMD"

if [[ "${DROID_ROBOT_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec bash -lc "$DROID_ROBOT_LAUNCH_CMD"
