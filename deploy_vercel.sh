#!/usr/bin/env bash
# Optional: deploy the static frontend shell to Vercel, so you can open the UI
# from any phone without hosting it yourself. The shell talks back to your local
# server over a tunnel; no agent code runs on Vercel.
#
# Auth: create a token at vercel.com/account/settings/tokens and save it to
# .vercel_token next to this script (git-ignored). Set VERCEL_SCOPE/VERCEL_PROJECT
# to your own team slug and project name.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .vercel_token ]]; then
  echo "Missing $(pwd)/.vercel_token — create a Vercel token and save it there." >&2
  exit 1
fi
TOKEN="$(cat .vercel_token)"
SCOPE="${VERCEL_SCOPE:?set VERCEL_SCOPE to your Vercel team/account slug}"
PROJECT="${VERCEL_PROJECT:-hermes-voice}"

cp static/index.html static/manifest.webmanifest static/icon-180.png static/icon-512.png static/sw.js vercel/
# Activity system loaded as /static/agent-activity.js from index.html
mkdir -p vercel/static
cp static/agent-activity.js vercel/static/agent-activity.js
cp static/agent-activity.js vercel/agent-activity.js
cd vercel
if [[ ! -f .vercel/project.json ]]; then
  vercel link --yes --project "$PROJECT" --scope "$SCOPE" --token "$TOKEN"
fi
exec vercel deploy --prod --yes --scope "$SCOPE" --token "$TOKEN"
