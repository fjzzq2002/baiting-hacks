#!/bin/bash
# Serve the dashboard on this machine from data that tools/study_dash_data.py already wrote: no tunnel and no
# rebuilding. The page and the five data files are copied into <site dir>, which is served on 127.0.0.1 only.
#   bash tools/study_dash/serve_local.sh <data dir> <site dir> [port]
set -eu
DATA=$1; SITE=$2; PORT=${3:-8766}
mkdir -p "$SITE"
cp "$(dirname "$0")/index.html" "$SITE/index.html"
for name in runs curves lanes paths probes; do cp "$DATA/$name.json" "$SITE/$name.json"; done
echo "http://127.0.0.1:$PORT/"
exec python3 -m http.server --bind 127.0.0.1 --directory "$SITE" "$PORT"
