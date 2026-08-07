#!/usr/bin/env bash
# Usage: ./setup_ua_detrac.sh <KAGGLE_DATASET_SLUG> <SEQ1> <SEQ2> <SEQ3> <SEQ4>
# Example: ./setup_ua_detrac.sh solesensei/solesensei_uadtrac MVI_20011 MVI_20012 MVI_20032 MVI_20052

set -euo pipefail

if ! command -v kaggle &> /dev/null; then
    echo "❌ Error: 'kaggle' CLI not found. Please pip install kaggle and setup ~/.kaggle/kaggle.json"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: 'ffmpeg' not found. Please install it (e.g., sudo apt install ffmpeg)"
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <OWNER/DATASET-SLUG> [SEQ1] [SEQ2] [SEQ3] [SEQ4]"
    exit 1
fi

KAGGLE_DATASET="$1"
SEQ1="${2:-MVI_20011}"
SEQ2="${3:-MVI_20012}"
SEQ3="${4:-MVI_20032}"
SEQ4="${5:-MVI_20052}"
SEQS=("$SEQ1" "$SEQ2" "$SEQ3" "$SEQ4")
CAMS=("cam_01" "cam_02" "cam_03" "cam_04")

ARCHIVE_DIR="data/raw/ua_detrac/_archives"
IMG_DIR="data/raw/ua_detrac/images"
ANN_DIR="data/raw/ua_detrac/annotations"
VID_DIR="data/processed/videos"
MAN_DIR="data/manifests"

mkdir -p "$ARCHIVE_DIR" "$IMG_DIR" "$ANN_DIR" "$VID_DIR" "$MAN_DIR"

# 1. Download Dataset
echo "📥 Downloading dataset $KAGGLE_DATASET..."
kaggle datasets download -d "$KAGGLE_DATASET" -p "$ARCHIVE_DIR"
ARCHIVE_PATH=$(find "$ARCHIVE_DIR" -name "*.zip" | head -n 1)

if [ -z "$ARCHIVE_PATH" ]; then
    echo "❌ Error: Failed to find downloaded zip archive in $ARCHIVE_DIR"
    exit 1
fi

echo "📦 Extracting selected sequences from $ARCHIVE_PATH..."

# Find the common root for images and xmls inside the zip using zipinfo
# We use python's zipfile module to avoid dependency on 'unzip'
for SEQ in "${SEQS[@]}"; do
    echo "  -> Extracting $SEQ..."
    python3 -c "
import zipfile
import sys
import os
from pathlib import Path

archive_path = sys.argv[1]
seq = sys.argv[2]
img_dir = Path(sys.argv[3]) / seq
ann_dir = Path(sys.argv[4])

img_dir.mkdir(parents=True, exist_ok=True)
ann_dir.mkdir(parents=True, exist_ok=True)

try:
    with zipfile.ZipFile(archive_path, 'r') as z:
        for info in z.infolist():
            if seq in info.filename:
                # Extract image
                if info.filename.endswith('.jpg'):
                    filename = Path(info.filename).name
                    target = img_dir / filename
                    target.write_bytes(z.read(info.filename))
                # Extract XML
                elif info.filename.endswith('.xml'):
                    filename = Path(info.filename).name
                    target = ann_dir / filename
                    target.write_bytes(z.read(info.filename))
except Exception as e:
    print(f'⚠️ Warning: Issues extracting {seq}: {e}')
" "$ARCHIVE_PATH" "$SEQ" "$IMG_DIR" "$ANN_DIR"
done

# 2. Convert to MP4
echo "🎥 Converting sequences to MP4 (25 FPS)..."
for i in "${!SEQS[@]}"; do
    SEQ="${SEQS[$i]}"
    CAM="${CAMS[$i]}"
    
    IMG_SEQ_DIR="$IMG_DIR/$SEQ"
    NUM_FRAMES=$(find "$IMG_SEQ_DIR" -name "*.jpg" 2>/dev/null | wc -l)
    
    if [ "$NUM_FRAMES" -eq 0 ]; then
        echo "❌ Error: No JPEG frames found for sequence $SEQ in $IMG_SEQ_DIR"
        continue
    fi
    
    echo "  -> Processing $SEQ ($NUM_FRAMES frames) into $CAM.mp4"
    
    # We find the exact filename pattern (e.g. img00001.jpg)
    FIRST_FILE=$(ls "$IMG_SEQ_DIR"/*.jpg | sort | head -n 1)
    BASENAME=$(basename "$FIRST_FILE")
    PREFIX="${BASENAME%%[0-9]*.jpg}" # Extract prefix before numbers
    DIGITS=$(echo "$BASENAME" | grep -o "[0-9]*" | head -n 1 | awk '{print length}')
    
    ffmpeg -y \
        -framerate 25 \
        -i "$IMG_SEQ_DIR/${PREFIX}%0${DIGITS}d.jpg" \
        -c:v libx264 \
        -preset medium \
        -crf 18 \
        -pix_fmt yuv420p \
        -movflags +faststart \
        "$VID_DIR/${CAM}.mp4" -loglevel error
done

# 3. Create Manifest
MANIFEST_PATH="$MAN_DIR/ua_detrac_selected.yaml"
echo "📄 Generating manifest $MANIFEST_PATH..."

cat <<EOF > "$MANIFEST_PATH"
dataset:
  name: UA-DETRAC
  source_fps: 25
  stream_relationship: independent_sequences
  notes: >
    These sequences are used as independent traffic-camera inputs.
    They are not synchronized views of the same road network.

cameras:
EOF

for i in "${!SEQS[@]}"; do
    SEQ="${SEQS[$i]}"
    CAM="${CAMS[$i]}"
    cat <<EOF >> "$MANIFEST_PATH"
  - camera_id: $CAM
    sequence_id: $SEQ
    video_path: data/processed/videos/${CAM}.mp4
    annotation_path: data/raw/ua_detrac/annotations/${SEQ}.xml
    scenario: unclassified

EOF
done

echo "✅ Setup complete!"
