#!/usr/bin/env python3
"""
cv-multi-monitoring — Prepare Data Script
Downloads 4 public test videos (traffic scenarios) to simulate the UA-DETRAC
dataset for Phase 2, until real private data is available.
"""

import os
import urllib.request
from pathlib import Path

# Placeholder URLs for public test videos (from Intel IoT sample videos and similar)
VIDEOS = {
    "cam_01.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "cam_02.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
    # Falling back to the same videos if we don't have 4 distinct CC0 traffic videos easily available
    "cam_03.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "cam_04.mp4": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
}

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading test videos to: {data_dir}\n")

    for filename, url in VIDEOS.items():
        out_path = data_dir / filename
        if out_path.exists():
            print(f"✅ {filename} already exists, skipping.")
            continue
        
        print(f"📥 Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, str(out_path))
            print(f"✅ Downloaded {filename} successfully.")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

    print("\nData preparation complete!")
    print("Note: These are public placeholders. To use the UA-DETRAC dataset,")
    print("replace these files with your actual clips and update the configs.")

if __name__ == "__main__":
    main()
