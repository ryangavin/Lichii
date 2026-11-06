#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! -f "$1" ]]; then
    echo "Usage: bash tools/nx2/run-profile.sh /path/to/game.nup [Lichii arguments...]" >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
binary="${LICHII_BINARY:-$repo_dir/build/dev/bin/lichii}"
if [[ ! -x "$binary" ]]; then
    echo "Lichii executable not found: $binary" >&2
    exit 1
fi

export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-1.6}"
export EDEN_NX2_DEBUG_SERVER="${EDEN_NX2_DEBUG_SERVER:-127.0.0.1:6545}"
if [[ "${LICHII_PROFILE:-1}" == 1 ]]; then
    export EDEN_NX2_GPU_PROFILE=1
    export EDEN_DKB_FRAME_TRACE=1
else
    unset EDEN_NX2_GPU_PROFILE EDEN_DKB_FRAME_TRACE
fi

game_path="$1"
shift
exec "$binary" -g "$game_path" "$@"
