#!/usr/bin/env bash
#
# Download the movie dataset into the Movie folder.
#
# The dataset is published on Kaggle as "Movie Dataset Collection".

set -euo pipefail

TARGET_DIR="${TARGET_DIR:-Movie}"

cd "$(dirname "$0")/.."

mkdir -p "$TARGET_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
    echo "The Kaggle CLI is not installed. Install it with: pip install kaggle"
    exit 1
fi

kaggle datasets download -d utkarshsaxena/movie-dataset-collection -p . --unzip

echo "Done. The dataset is in $TARGET_DIR"
