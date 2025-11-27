#!/usr/bin/env bash
#
# Download the movie dataset into the Movie folder.
#
# The dataset is the MovieLens data joined with TMDB metadata, published on Kaggle as
# "Movie Dataset Collection". This script pulls it with the Kaggle CLI when that is
# available, and falls back to a plain download when DATASET_URL is set.

set -euo pipefail

DATASET_SLUG="${DATASET_SLUG:-utkarshsaxena/movie-dataset-collection}"
DATASET_URL="${DATASET_URL:-}"
TARGET_DIR="${TARGET_DIR:-Movie}"
ARCHIVE="${ARCHIVE:-movie-dataset.zip}"

cd "$(dirname "$0")/.."

mkdir -p "$TARGET_DIR"

if [ -n "$DATASET_URL" ]; then
    echo "Downloading $DATASET_URL"
    curl -L --fail --progress-bar "$DATASET_URL" -o "$ARCHIVE"
elif command -v kaggle >/dev/null 2>&1; then
    echo "Downloading $DATASET_SLUG with the Kaggle CLI"
    kaggle datasets download -d "$DATASET_SLUG" -p . --unzip
else
    cat <<'MSG'
Could not find the Kaggle CLI and DATASET_URL is not set.

Either install and authenticate it:
    pip install kaggle
    kaggle config

or point the script straight at an archive:
    DATASET_URL=https://example.com/movie-dataset.zip ./scripts/fetch_data.sh
MSG
    exit 1
fi

if [ -f "$ARCHIVE" ]; then
    echo "Extracting $ARCHIVE into $TARGET_DIR"
    unzip -o "$ARCHIVE" -d "$TARGET_DIR"
fi

echo "Done. The dataset is in $TARGET_DIR"
