#!/usr/bin/env bash
# Launch the Lichii desktop app in a window for a watched test run: the performance overlay
# (fps, avg/min/max, frame time) is switched on, and the debug server, GPU profile and frame
# trace are enabled as in run-profile.sh. Ctrl+V toggles the overlay in the window.
#
#   bash tools/nx2/run-window.sh [/path/to/game.nup] [Lichii arguments...]
#
# Without a game path, LICHII_GAME is used. LICHII_PROFILE=0 leaves the profiling traces off.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
config="${XDG_CONFIG_HOME:-$HOME/.config}/eden/qt-config.ini"

game_path="${1:-${LICHII_GAME:-}}"
if [[ -z "$game_path" || ! -f "$game_path" ]]; then
    echo "Usage: bash tools/nx2/run-window.sh /path/to/game.nup [Lichii arguments...]" >&2
    echo "       (or set LICHII_GAME to the game path)" >&2
    exit 2
fi
[[ $# -gt 0 ]] && shift

if pgrep -x lichii >/dev/null || pgrep -x eden-cli >/dev/null; then
    echo "An emulator is already running; the config is rewritten on exit, so start the test after it quits." >&2
    pgrep -a -x lichii; pgrep -a -x eden-cli
    exit 1
fi

# The Qt frontend saves this key on exit, so set it while nothing is running. The reader
# skips the stored value while the "\default" marker is true, so clear that marker too.
if [[ -f "$config" ]] && grep -q '^UIGameList\\show_perf_overlay=' "$config"; then
    sed -i -e 's/^UIGameList\\show_perf_overlay=.*/UIGameList\\show_perf_overlay=true/' \
           -e 's/^UIGameList\\show_perf_overlay\\default=.*/UIGameList\\show_perf_overlay\\default=false/' "$config"
    echo "Performance overlay enabled in $config"
else
    echo "No show_perf_overlay key in $config yet; press Ctrl+V in the window to show the overlay." >&2
fi

echo "Log: ${XDG_DATA_HOME:-$HOME/.local/share}/eden/log/eden_log.txt"
echo "Live profile: python3 -I tools/nx2/debug_mcp.py --server 127.0.0.1:6545 --call '{\"method\":\"profile_snapshot\",\"params\":{\"limit\":5}}'"
exec bash "$repo_dir/tools/nx2/run-profile.sh" "$game_path" "$@"
