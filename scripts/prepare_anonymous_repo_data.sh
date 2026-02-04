#!/bin/bash
# Copy dataset files into anonymous_repo/data/
# Run from project root: bash scripts/prepare_anonymous_repo_data.sh

SRC=/data/xuao/llm-knowledge-extraction-rec/data/recbole
DST=/data/xuao/llm-knowledge-extraction-rec/anonymous_repo/data

mkdir -p "$DST/ml-1m" "$DST/amazon-beauty" "$DST/amazon-videogames"

echo "Copying dataset files for anonymous repo..."

# ML-1M
echo "  ML-1M..."
cp "$SRC/ml-1m/ml-1m.inter"    "$DST/ml-1m/"
cp "$SRC/ml-1m/ml-1m.item.kg"  "$DST/ml-1m/"
cp "$SRC/ml-1m/ml-1m.user.kg"  "$DST/ml-1m/"

# Amazon Beauty
echo "  Amazon Beauty..."
cp "$SRC/amazon-beauty/amazon-beauty.inter"    "$DST/amazon-beauty/"
cp "$SRC/amazon-beauty/amazon-beauty.item.kg"  "$DST/amazon-beauty/"
cp "$SRC/amazon-beauty/amazon-beauty.user.kg"  "$DST/amazon-beauty/"

# Amazon Video Games
echo "  Amazon Video Games..."
cp "$SRC/amazon-videogames/amazon-videogames.inter"    "$DST/amazon-videogames/"
cp "$SRC/amazon-videogames/amazon-videogames.item.kg"  "$DST/amazon-videogames/"
cp "$SRC/amazon-videogames/amazon-videogames.user.kg"  "$DST/amazon-videogames/"

echo ""
echo "Done. Verifying:"
for f in "$DST"/*/*.{inter,kg}; do
    lines=$(wc -l < "$f")
    size=$(du -h "$f" | cut -f1)
    echo "  $f: $lines lines ($size)"
done
