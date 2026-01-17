#!/bin/bash
# Download Amazon Video Games and Beauty datasets
# Source: https://jmcauley.ucsd.edu/data/amazon_v2/

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=========================================="
echo "Downloading Amazon Review Datasets"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Create directories
mkdir -p "${PROJECT_ROOT}/data/raw/amazon-videogames"
mkdir -p "${PROJECT_ROOT}/data/raw/amazon-beauty"

BASE_URL_REVIEWS="https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall"
BASE_URL_META="https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2"

# Video Games
echo "1. Downloading Video Games dataset..."
echo "   - Reviews (5-core)..."
wget -c "${BASE_URL_REVIEWS}/Video_Games_5.json.gz" \
    -O "${PROJECT_ROOT}/data/raw/amazon-videogames/Video_Games_5.json.gz"

echo "   - Metadata..."
wget -c "${BASE_URL_META}/meta_Video_Games.json.gz" \
    -O "${PROJECT_ROOT}/data/raw/amazon-videogames/meta_Video_Games.json.gz"

echo "✓ Video Games downloaded"
echo ""

# Beauty
echo "2. Downloading Beauty dataset..."
echo "   - Reviews (5-core)..."
wget -c "${BASE_URL_REVIEWS}/Beauty_5.json.gz" \
    -O "${PROJECT_ROOT}/data/raw/amazon-beauty/Beauty_5.json.gz"

echo "   - Metadata..."
wget -c "${BASE_URL_META}/meta_Beauty.json.gz" \
    -O "${PROJECT_ROOT}/data/raw/amazon-beauty/meta_Beauty.json.gz"

echo "✓ Beauty downloaded"
echo ""

echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Files downloaded:"
echo "  data/raw/amazon-videogames/Video_Games_5.json.gz"
echo "  data/raw/amazon-videogames/meta_Video_Games.json.gz"
echo "  data/raw/amazon-beauty/Beauty_5.json.gz"
echo "  data/raw/amazon-beauty/meta_Beauty.json.gz"
echo ""
echo "Next steps:"
echo "  1. Unzip: gunzip data/raw/amazon-*/*.gz"
echo "  2. Explore data: python scripts/explore_amazon_data.py"
echo ""
