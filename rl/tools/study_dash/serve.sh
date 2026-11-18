#!/bin/bash
# Serve the study's dashboard read-only: static files behind a random path, on a Cloudflare quick tunnel, with the data
# rebuilt every 2 minutes. Run from rl/ after `source $ROOT/env.sh`. The address is in <work>/tunnel.log.
#   bash tools/study_dash/serve.sh <study log> <work dir>
set -eu
STUDY_LOG=$1; WORK=$2; PORT=8765
SITE=$WORK/site
mkdir -p "$SITE"
[ -f "$WORK/token" ] || python3 -c "import secrets; print(secrets.token_urlsafe(12))" > "$WORK/token"
TOKEN=$(cat "$WORK/token")
mkdir -p "$SITE/$TOKEN"
: > "$SITE/index.html"  # so the root shows nothing, not a listing with the token in it
cp tools/study_dash/index.html "$SITE/$TOKEN/index.html"

publish() {
  python tools/study_dash_data.py --study_log "$STUDY_LOG" --out "$WORK/data" | tail -1
  for name in runs curves lanes paths probes; do
    cp "$WORK/data/$name.json" "$SITE/$TOKEN/.$name.tmp" && mv "$SITE/$TOKEN/.$name.tmp" "$SITE/$TOKEN/$name.json"
  done
}
publish
(while true; do sleep 120; nice -n 10 bash -c "$(declare -f publish); STUDY_LOG=$STUDY_LOG WORK=$WORK SITE=$SITE TOKEN=$TOKEN; publish" || true; done) &

python3 -m http.server --bind 127.0.0.1 --directory "$SITE" $PORT > "$WORK/http.log" 2>&1 &
[ -x "$WORK/cloudflared" ] || { curl -sL -o "$WORK/cloudflared" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64; chmod +x "$WORK/cloudflared"; }
"$WORK/cloudflared" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$WORK/tunnel.log" 2>&1 &
wait
