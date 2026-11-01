#!/usr/bin/env bash
# Smoke-test one bundled example: set it up, train it briefly, then evaluate the checkpoint.
# Requires the `tensorial` command (e.g. run with `uv run`).
#
# Usage: run_example.sh <example name, e.g. mlip.si> [work dir]
#
# The work dir is reused between runs, so datasets downloaded into it (its `data/` and
# `.cache/` folders) don't need downloading again.  Extra Hydra overrides can be passed
# in the EXAMPLE_OVERRIDES environment variable.
set -euo pipefail

name="${1:?usage: run_example.sh <example name> [work dir]}"
work_dir="${2:-examples-smoke}"
overrides="${EXAMPLE_OVERRIDES:-}"

# `mlip.si` is initialised into `<work dir>/mlip-si`
example_dir="${work_dir}/${name//./-}"

echo "::group::Initialise ${name}"
tensorial examples init "${name}" "${work_dir}"
echo "::endgroup::"

cd "${example_dir}"
# Start from a clean slate, so we evaluate the checkpoint from this run
rm -rf logs

echo "::group::Train ${name}"
# shellcheck disable=SC2086  # overrides are deliberately split into words
tensorial train -i train.yaml debug=smoke ${overrides}
echo "::endgroup::"

# Training ends by testing the best checkpoint (`test: true` in debug=smoke).  Also evaluate the
# run the way a user would, from its directory, to check that the saved config can be reloaded.
run_dir="$(find logs/train -mindepth 1 -maxdepth 1 -type d | head -n 1)"
ckpt="${run_dir}/checkpoints/last.ckpt"
if [[ ! -f "${ckpt}" ]]; then
    echo "::error::Training ${name} did not write a checkpoint"
    exit 1
fi

echo "::group::Evaluate ${name}"
tensorial predict -i "${run_dir}" ckpt_path="${PWD}/${ckpt}" +config_path="${PWD}/${run_dir}/config.yaml"
echo "::endgroup::"

echo "${name}: OK"
