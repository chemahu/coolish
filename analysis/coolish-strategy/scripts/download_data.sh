#!/usr/bin/env bash
# download_data.sh — fetch a tagged release from bwjoke/BTC-Trading-Since-2020,
# verify SHA-256 checksums from manifest.json, and print basic statistics.
#
# Usage:
#   bash scripts/download_data.sh                       # latest tag
#   bash scripts/download_data.sh --tag data-2026-04-17

set -euo pipefail

REPO="bwjoke/BTC-Trading-Since-2020"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
TAG="latest"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# ── resolve "latest" to a real tag name ───────────────────────────────────────
if [[ "$TAG" == "latest" ]]; then
  echo "[download] Resolving latest tag from GitHub API…"
  TAG=$(curl -fsSL \
    "https://api.github.com/repos/${REPO}/tags?per_page=100" \
    | python3 -c "
import sys, json
tags = json.load(sys.stdin)
data_tags = [t['name'] for t in tags if t['name'].startswith('data-')]
data_tags.sort()
print(data_tags[-1])
")
  echo "[download] Latest tag resolved to: $TAG"
fi

ARCHIVE_URL="https://github.com/${REPO}/archive/refs/tags/${TAG}.tar.gz"
ARCHIVE_FILE="$DATA_DIR/${TAG}.tar.gz"
EXTRACT_DIR="$DATA_DIR/${TAG}"

mkdir -p "$DATA_DIR"

# ── download ──────────────────────────────────────────────────────────────────
if [[ -f "$ARCHIVE_FILE" ]]; then
  echo "[download] Archive already exists: $ARCHIVE_FILE  (skipping download)"
else
  echo "[download] Fetching $ARCHIVE_URL …"
  curl -L --retry 3 --retry-delay 5 -o "$ARCHIVE_FILE" "$ARCHIVE_URL"
  echo "[download] Saved to $ARCHIVE_FILE"
fi

# ── extract ───────────────────────────────────────────────────────────────────
if [[ -d "$EXTRACT_DIR" ]]; then
  echo "[download] Extract dir already exists: $EXTRACT_DIR  (skipping extract)"
else
  echo "[download] Extracting…"
  mkdir -p "$EXTRACT_DIR"
  # GitHub archives wrap content inside a repo-name-tag/ top-level dir; strip it
  tar -xzf "$ARCHIVE_FILE" -C "$EXTRACT_DIR" --strip-components=1
  echo "[download] Extracted to $EXTRACT_DIR"
fi

# ── SHA-256 verification via manifest.json ────────────────────────────────────
MANIFEST="$EXTRACT_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "[warn] manifest.json not found – skipping checksum verification" >&2
else
  echo "[download] Verifying checksums from manifest.json…"
  MANIFEST_PATH="$MANIFEST" EXTRACT_DIR="$EXTRACT_DIR" python3 - <<'PYEOF'
import json, hashlib, sys, os

manifest_path = os.environ.get("MANIFEST_PATH")
extract_dir   = os.environ.get("EXTRACT_DIR")

with open(manifest_path) as f:
    manifest = json.load(f)

files = manifest.get("files", manifest) if isinstance(manifest, dict) else manifest

errors = []
for entry in (files if isinstance(files, list) else [{"name": k, "sha256": v} for k, v in files.items()]):
    name   = entry.get("name") or entry.get("filename")
    sha256 = entry.get("sha256") or entry.get("checksum")
    if not name or not sha256:
        continue
    path = os.path.join(extract_dir, name)
    if not os.path.exists(path):
        print(f"  [warn]  MISSING   {name}")
        continue
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest == sha256:
        print(f"  [ok]    {name}")
    else:
        print(f"  [FAIL]  {name}  expected={sha256[:16]}…  got={digest[:16]}…")
        errors.append(name)

if errors:
    print(f"\n[error] Checksum mismatch for {len(errors)} file(s)!", file=sys.stderr)
    sys.exit(1)
else:
    print("[download] All checksums OK")
PYEOF
fi

# ── print statistics for each CSV ─────────────────────────────────────────────
echo ""
echo "[download] File statistics:"
printf "%-55s %10s %12s   %-24s  %-24s\n" "File" "Size" "Lines" "First timestamp" "Last timestamp"
printf "%-55s %10s %12s   %-24s  %-24s\n" "----" "----" "-----" "---------------" "--------------"

for csv_file in "$EXTRACT_DIR"/*.csv; do
  [[ -f "$csv_file" ]] || continue
  fname=$(basename "$csv_file")
  size=$(du -sh "$csv_file" | cut -f1)
  lines=$(( $(wc -l < "$csv_file") - 1 ))  # subtract header

  # Try to find timestamp column and print first/last value
  ts_info=$(python3 - <<PYEOF2
import csv, sys
path = "$csv_file"
try:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        ts_col = None
        for candidate in ("timestamp", "transactTime", "date"):
            if candidate in header:
                ts_col = header.index(candidate)
                break
        if ts_col is None:
            print("N/A  N/A")
            sys.exit(0)
        first_val = ""
        last_val  = ""
        for row in reader:
            if ts_col < len(row) and row[ts_col]:
                if not first_val:
                    first_val = row[ts_col]
                last_val = row[ts_col]
    print(f"{first_val[:24]}  {last_val[:24]}")
except Exception as e:
    print(f"N/A  N/A  ({e})")
PYEOF2
)
  first_ts=$(echo "$ts_info" | awk '{print $1}')
  last_ts=$(echo "$ts_info"  | awk '{print $2}')
  printf "%-55s %10s %12s   %-24s  %-24s\n" "$fname" "$size" "$lines" "$first_ts" "$last_ts"
done

echo ""
echo "[download] Done. Data directory: $EXTRACT_DIR"
