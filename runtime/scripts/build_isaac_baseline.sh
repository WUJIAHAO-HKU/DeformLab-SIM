#!/usr/bin/env bash
# Rebuild the pinned official application without activating the research venv.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_root="$project_root/upstream/IsaacSim"
expected_commit=7c206f75bdadd9e05fc457f19863ca4c3f0cb693
mode="${1:-}"
if [[ "$#" -ne 1 || ( "$mode" != fetch && "$mode" != build ) ]]; then
    echo 'Usage: scripts/build_isaac_baseline.sh fetch|build' >&2
    exit 2
fi

if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$expected_commit" ]]; then
    echo 'Refusing: upstream commit differs from the recorded v6.1.0 baseline.' >&2
    exit 2
fi
if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
    echo 'Refusing: tracked upstream source is modified; preserve a clean baseline.' >&2
    exit 2
fi
if [[ ! -f "$source_root/.eula_accepted" ]]; then
    echo 'Official NVIDIA license acknowledgment is required before this build.' >&2
    echo "After the operator accepts the terms, use: cd '$source_root' && ./tools/eula_check.sh" >&2
    exit 3
fi
# Pixi 0.77.1 reads project config even with PIXI_NO_CONFIG=1, which disables
# system/user config. Preserve an existing different project profile for review.
network_profile="$project_root/configs/isaac-pixi-network-v1.toml"
pixi_profile="$source_root/.pixi/config.toml"
if [[ ! -f "$network_profile" || -L "$source_root/.pixi" || -L "$pixi_profile" ]]; then
    echo 'Recorded project network profile is missing or has an unexpected symlink.' >&2
    exit 2
fi
mkdir -p "$source_root/.pixi"
if [[ ! -e "$pixi_profile" ]]; then
    (umask 077; cp --no-clobber "$network_profile" "$pixi_profile")
fi
if ! cmp -s "$network_profile" "$pixi_profile"; then
    echo 'Project Pixi config differs from the recorded network profile; inspect it before building.' >&2
    exit 2
fi

git lfs version
command -v gcc >/dev/null
command -v g++ >/dev/null
mkdir -p "$project_root/logs/isaac-build" "$project_root/.cache/isaac/packman"
export XDG_CACHE_HOME="$project_root/.cache/isaac"
mkdir -p "$XDG_CACHE_HOME/tmp"
export TMPDIR="$XDG_CACHE_HOME/tmp"
export PM_PACKAGES_ROOT="$XDG_CACHE_HOME/packman"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
pip_profile="$project_root/configs/isaac-pip-network-v1.conf"
if [[ ! -f "$pip_profile" || -L "$pip_profile" ]]; then
    echo 'Recorded project pip network profile is missing or is a symlink.' >&2
    exit 2
fi
# Unlike PIP_INDEX_URL, this file is read by the pinned pip 25.3 even when
# repo_build invokes --isolated. Explicit upstream --index-url takes precedence.
export PIP_CONFIG_FILE="$pip_profile"
# wheel-stub has no socket timeout; use its supported vendor index override.
export WHEEL_STUB_PIP_INDEX_URL=https://pypi.nvidia.cn/
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export OM_PIP_CACHE=1 OM_UV_CACHE=1
# Fixed manifests use Azure's extension registry, PyTorch's CUDA wheel index,
# conda-forge for the libraries' Pixi environment, and the public Isaac asset
# bucket. Direct TLS reachability was probed on this server; retain direct routes.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}ovextensionsprod.blob.core.windows.net,download.pytorch.org,download-r2.pytorch.org,conda.anaconda.org,omniverse-content-production.s3-us-west-2.amazonaws.com,pypi.nvidia.com,pypi.nvidia.cn,mirrors.aliyun.com"
export no_proxy="$NO_PROXY"
unset PYTHONPATH PYTHONHOME PYTHONEXE VIRTUAL_ENV
# The official marker above records the user's accepted terms. Kit's Python
# bootstrap uses a separate noninteractive switch for that same license.
export OMNI_KIT_ACCEPT_EULA=YES
if [[ "$(id -u)" == 0 ]]; then
    # Matches the fixed upstream build/test CI environment; no privilege change.
    export OMNI_KIT_ALLOW_ROOT=1
fi

run_tag="$(date -u +%Y%m%dT%H%M%SZ)-$mode-$$"
log_file="$project_root/logs/isaac-build/$run_tag.log"
echo "Pinned source: $expected_commit"
echo "Mode: $mode; log: $log_file"
df -h "$source_root"
cd "$source_root"
if [[ "$mode" == fetch ]]; then
    (
        git lfs install --local &&
        git lfs pull &&
        # Fixed v6.1.0 repo.toml disables Docker already. Its actual parser
        # rejects --no-docker, despite the older .cursor build instructions.
        ./build.sh --release --fetch-only
    ) 2>&1 | tee "$log_file"
else
    start_record="$project_root/results/isaac-build/$run_tag-start.json"
    event_record="$project_root/results/isaac-build/$run_tag-event.json"
    python3 "$project_root/scripts/isaac_build_receipt.py" begin \
        --source-root "$source_root" --log "$log_file" --output "$start_record" \
        -- ./build.sh --release --jobs 8
    ./build.sh --release --jobs 8 2>&1 | tee "$log_file"
fi
# This repo_man version can return 0 after argparse rejects unknown flags.
# Such a log is a failed invocation regardless of the shell exit code.
if grep -qE '^error: (unrecognized arguments|argument )' "$log_file"; then
    echo 'Official CLI rejected the invocation; treating its zero exit as failure.' >&2
    exit 2
fi
if [[ "$mode" == build ]]; then
    # set -e/pipefail and the CLI-error check above must both succeed before
    # recording exit 0. Final provenance also requires a real Kit inventory.
    python3 "$project_root/scripts/isaac_build_receipt.py" record-build \
        --start "$start_record" --exit-code 0 --output "$event_record"
fi
