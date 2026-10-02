#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=data_dl
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --output=dl_logs/data_dl.%j.out
#SBATCH --error=dl_logs/data_dl_error.%j.out

# Define target directory
# Define base target directory relative to script location
BASE_DIR="$(pwd)/ref_dataset"
TARGET_DIR="$BASE_DIR/data"


# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Download ReferIt datasets directly into TARGET_DIR
wget -c --retry-connrefused --waitretry=5 --timeout=120 --tries=10 --show-progress \
    -O "$TARGET_DIR/refclef.zip" "https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip"

wget -c --retry-connrefused --waitretry=5 --timeout=120 --tries=10 --show-progress \
    -O "$TARGET_DIR/refcoco.zip" "https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"

wget -c --retry-connrefused --waitretry=5 --timeout=120 --tries=10 --show-progress \
    -O "$TARGET_DIR/refcoco+.zip" "https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"

wget -c --retry-connrefused --waitretry=5 --timeout=120 --tries=10 --show-progress \
    -O "$TARGET_DIR/refcocog.zip" "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"

# Extract ReferIt datasets directly into TARGET_DIR
unzip -o "$TARGET_DIR/refclef.zip" -d "$TARGET_DIR"
unzip -o "$TARGET_DIR/refcoco.zip" -d "$TARGET_DIR"
unzip -o "$TARGET_DIR/refcoco+.zip" -d "$TARGET_DIR"
unzip -o "$TARGET_DIR/refcocog.zip" -d "$TARGET_DIR"

# Remove ZIP files after extraction
# rm -f refclef.zip refcoco.zip refcoco+.zip refcocog.zip

echo "All ReferIt datasets downloaded and extracted successfully in $TARGET_DIR."

# ============================ COCO Dataset Download ============================

# Define COCO dataset paths
COCO_URL="http://images.cocodataset.org/zips/train2014.zip"
COCO_TARGET_DIR="ref_dataset/dl_demo"
COCO_ZIP_FILE="$COCO_TARGET_DIR/train2014.zip"
COCO_EXTRACT_DIR="$COCO_TARGET_DIR/"

# Create COCO target directories if they do not exist
mkdir -p "$COCO_TARGET_DIR"
mkdir -p "$COCO_EXTRACT_DIR"

# Download COCO dataset
wget -c --retry-connrefused --waitretry=5 --timeout=30 --tries=5 --show-progress -O "$COCO_ZIP_FILE" "$COCO_URL"

# Verify COCO download
if [ -f "$COCO_ZIP_FILE" ]; then
    echo "Download complete: $COCO_ZIP_FILE"
else
    echo "Download failed."
    exit 1
fi

# Extract COCO dataset
echo "Extracting COCO dataset to $COCO_EXTRACT_DIR..."
unzip -o "$COCO_ZIP_FILE" -d "$COCO_EXTRACT_DIR"

# Verify COCO extraction
if [ $? -eq 0 ]; then
    echo "Extraction complete."
else
    echo "Extraction failed."
    exit 1
fi

echo "COCO dataset is ready in $COCO_EXTRACT_DIR."