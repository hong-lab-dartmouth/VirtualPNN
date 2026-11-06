#!/usr/bin/env bash
# Download the trained weights (one zip archive), verify the archive and every file in it against
# WEIGHTS_MANIFEST.csv, and lay the files out under weights/ where test.py and infer.py look for them
# (see INFERENCE.md).
#
#   ./fetch_weights.sh                                # download from WEIGHTS_ZIP_URL below
#   ./fetch_weights.sh path/to/VirtualPNN_weights_v1.0.zip   # use a copy you already downloaded
#
# VPNN_WEIGHTS_ZIP_URL, if set, overrides WEIGHTS_ZIP_URL.
set -euo pipefail
cd "$(dirname "$0")"

WEIGHTS_ZIP_URL="https://github.com/hong-lab-dartmouth/VirtualPNN/releases/download/v1.0/VirtualPNN_weights_v1.0.zip"
ZIP_NAME=VirtualPNN_weights_v1.0.zip
ZIP_SHA256=9b72e8037a537c9a94e93025d04b11d8861fab99d6d6e0a3bb6747130b428d85
MANIFEST=WEIGHTS_MANIFEST.csv
DEST=weights
PYTHON="${PYTHON:-python3}"

sha () { sha256sum "$1" | cut -d' ' -f1; }

# archive file name -> path under weights/ that test.py expects
target_path () {
  local a="$1" fold
  fold=$(echo "$a" | sed -nE 's/.*_Fold([0-9]+)_.*/\1/p')
  case "$a" in
    VirtualPNN-P+AV_*)              echo "$DEST/VirtualPNN_AV6_Fold${fold}/latest_net_G.pth" ;;
    VirtualPNN-P_*)                 echo "$DEST/VirtualPNN_noAV_Fold${fold}/latest_net_G.pth" ;;
    CUT_*)                          echo "$DEST/CUT_Fold${fold}/latest_net_G.pth" ;;
    AI-FFPE_*)                      echo "$DEST/AIFFPE_Fold${fold}/latest_net_G.pth" ;;
    CycleGAN_*_net_G_B.pth)         echo "$DEST/CycleGAN_Fold${fold}/latest_net_G_B.pth" ;;
    CycleGAN_*)                     echo "$DEST/CycleGAN_Fold${fold}/latest_net_G_A.pth" ;;
    aiffpe_pretrained_base_net_*)   echo "$DEST/aiffpe/5_net_${a#aiffpe_pretrained_base_net_}" ;;
    *) echo "unknown archive name: $a" >&2; return 1 ;;
  esac
}

# 1. Already in place?
missing=0
while IFS=, read -r model fold role archive bytes want; do
  [ "$model" = "model" ] && continue
  out=$(target_path "$archive")
  if [ ! -f "$out" ] || [ "$(sha "$out")" != "$want" ]; then missing=$((missing+1)); fi
done < "$MANIFEST"

if [ "$missing" -gt 0 ]; then
  # 2. Get the archive
  mkdir -p "$DEST"
  if [ $# -ge 1 ]; then
    zip="$1"
  else
    url="${VPNN_WEIGHTS_ZIP_URL:-$WEIGHTS_ZIP_URL}"
    [ -n "$url" ] || { echo "No download link: set VPNN_WEIGHTS_ZIP_URL, or pass the downloaded $ZIP_NAME" >&2; exit 1; }
    zip="$DEST/$ZIP_NAME"
    echo "download     $ZIP_NAME"
    curl -fL --retry 3 -o "$zip" "$url"
  fi
  got=$(sha "$zip")
  if [ "$got" != "$ZIP_SHA256" ]; then
    echo "SHA256 MISMATCH for the archive $zip" >&2; echo "  expected $ZIP_SHA256" >&2; echo "  got      $got" >&2; exit 1
  fi
  echo "archive ok   $zip"

  # 3. Unpack, verify each file, move it into place
  tmp=$(mktemp -d "$DEST/.unpack.XXXXXX")
  trap 'rm -rf "$tmp"' EXIT
  "$PYTHON" -m zipfile -e "$zip" "$tmp"
  src="$tmp/VirtualPNN_weights"
  while IFS=, read -r model fold role archive bytes want; do
    [ "$model" = "model" ] && continue
    f="$src/$archive"
    [ -f "$f" ] || { echo "missing from archive: $archive" >&2; exit 1; }
    if [ "$(stat -c %s "$f")" != "$bytes" ] || [ "$(sha "$f")" != "$want" ]; then
      echo "SHA256 or size mismatch for $archive" >&2; exit 1
    fi
    out=$(target_path "$archive"); mkdir -p "$(dirname "$out")"; mv -f "$f" "$out"
  done < "$MANIFEST"
  [ $# -ge 1 ] || rm -f "$zip"
fi

# 4. The baselines read the same base under the folder name aiffpe_weights, CycleGAN under its own names
mkdir -p "$DEST/aiffpe_weights"
for pair in 5_net_G:5_net_G 5_net_D:5_net_D 5_net_F:5_net_F 5_net_G:5_net_G_A 5_net_G:5_net_G_B 5_net_D:5_net_D_A 5_net_D:5_net_D_B; do
  s="$DEST/aiffpe/${pair%%:*}.pth"; d="$DEST/aiffpe_weights/${pair#*:}.pth"
  [ -f "$d" ] || ln "$s" "$d" 2>/dev/null || cp "$s" "$d"
done

n=$(($(wc -l < "$MANIFEST") - 1))
echo "verified $n files under $DEST/ (generators per model and fold; AI-FFPE base in aiffpe/ and aiffpe_weights/)"
