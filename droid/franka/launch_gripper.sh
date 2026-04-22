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

pkill -9 gripper || true
pkill -9 franka_hand_client || true

if [[ -z "${DROID_GRIPPER_LAUNCH_CMD:-}" ]]; then
    if ! command -v launch_gripper.py >/dev/null 2>&1; then
        echo "launch_gripper.py was not found in the active environment: $CONDA_ENV" >&2
        exit 1
    fi

    # Expected Polymetis Franka Hand launcher. Override this whole command if
    # the installed Polymetis stack uses a different gripper config name.
    DROID_GRIPPER_LAUNCH_CMD="launch_gripper.py gripper=franka_hand"
fi

echo "Launching gripper with: $DROID_GRIPPER_LAUNCH_CMD"

if [[ "${DROID_GRIPPER_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec bash -lc "$DROID_GRIPPER_LAUNCH_CMD"
