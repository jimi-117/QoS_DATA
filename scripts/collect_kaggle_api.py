import kagglehub
import yaml
import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import sys
import logging

"""Configuration settings for the dataset processing"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "from_api"

################################ Dataset settings ###############################
DATASET_ID = "henningheyen/lvis-fruits-and-vegetables-dataset"
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

# 3. List of class names to be considered vegetables (based on the provided data.yaml)
#    Carefully review and adjust this list according to your definition of "vegetable".
VEGETABLE_NAMES_TO_KEEP = [
    'artichoke',
    'asparagus',
    # 'bean curd/tofu', # Processed food, decide if you want it
    'bell pepper/capsicum',
    'broccoli',
    'brussels sprouts',
    'carrot',
    'cauliflower',
    'cayenne/cayenne spice/cayenne pepper/cayenne pepper spice/red pepper/red pepper',
    'celery',
    'chickpea/garbanzo', # Legume
    'chili/chili vegetable/chili pepper/chili pepper vegetable/chilli/chilli vegetable/chilly/chilly',
    'edible corn/corn/maize', # Grain, often treated as vegetable
    'cucumber/cuke',
    'eggplant/aubergine',
    'garlic/ail',
    'ginger/gingerroot', # Rhizome
    'gourd', # Includes pumpkin, squash etc.
    'green bean',
    'green onion/spring onion/scallion',
    'lettuce',
    'mushroom', # Fungus, often treated as vegetable
    'onion',
    'pea/pea food', # Legume
    # 'pickle', # Processed cucumber
    'potato',
    'pumpkin',
    'radish/daikon',
    'sweet potato',
    'turnip',
    'zucchini/courgette',
    # 'Tomato', 'tomato' (IDs 35, 59) are botanically fruits. Excluded unless specifically needed.
]

# --- Script Start ---

print("--- 1. Downloading Dataset ---")
downloaded_path = None
try:
    # Download the dataset; kagglehub returns the path to the extracted files
    downloaded_path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Dataset downloaded and extracted to: {downloaded_path}")
except Exception as e:
    print(f"Error: Failed to download dataset '{DATASET_ID}'. Check dataset ID and Kaggle API setup.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1) # Exit script if download fails

# Define paths based on the downloaded location and expected structure
# IMPORTANT: Assuming a standard YOLO structure where labels mirror image paths
# (e.g., if images are in 'images/train', labels are in 'labels/train')
original_data_root = downloaded_path / "LVIS_Fruits_And_Vegetables" # The path returned by kagglehub
original_data_yaml_path = original_data_root / "data.yaml"
original_train_images_dir = original_data_root / "images" / "train" / "train"# As per data.yaml
original_train_labels_dir = original_data_root / "labels" / "train" / "train"# Assumed path

# Verify necessary original directories and files exist
if not original_data_yaml_path.exists():
    print(f"Error: Original data.yaml not found at expected path: {original_data_yaml_path}", file=sys.stderr)
    sys.exit(1)
if not original_train_images_dir.is_dir():
    print(f"Error: Original train images directory not found: {original_train_images_dir}", file=sys.stderr)
    sys.exit(1)
if not original_train_labels_dir.is_dir():
    print(f"Error: Assumed original train labels directory not found: {original_train_labels_dir}", file=sys.stderr)
    print("Please verify the dataset structure within:", downloaded_path, file=sys.stderr)
    sys.exit(1)


# Create output directories
new_train_images_dir = OUTPUT_DIR / "images" / "train"
new_train_labels_dir = OUTPUT_DIR / "labels" / "train"
new_data_yaml_path = OUTPUT_DIR / "data.yaml"

try:
    new_train_images_dir.mkdir(parents=True, exist_ok=True)
    new_train_labels_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"Error: Could not create output directories in {OUTPUT_DIR}: {e}", file=sys.stderr)
    sys.exit(1)


print("\n--- 2. Parsing Original data.yaml and Identifying Vegetable Class IDs ---")
try:
    with open(original_data_yaml_path, 'r', encoding='utf-8') as f: # Added encoding
        original_data_config = yaml.safe_load(f)
except Exception as e:
    print(f"Error: Failed to read or parse original data.yaml: {e}", file=sys.stderr)
    sys.exit(1)

original_names_dict = original_data_config.get('names')
if not isinstance(original_names_dict, dict):
    print(f"Error: 'names' in original data.yaml is not a dictionary as expected.", file=sys.stderr)
    sys.exit(1)

print(f"Original number of classes: {len(original_names_dict)}")

# Create mapping from original vegetable IDs to new sequential IDs (0, 1, 2...)
vegetable_original_ids = []
vegetable_new_names = []
original_id_to_new_id = {}
new_id_counter = 0

# Sort items by original ID (key) to ensure consistent new ID assignment
sorted_original_items = sorted(original_names_dict.items())

for original_id, name in sorted_original_items:
    try:
        # Ensure original_id is an integer if it wasn't loaded as such
        current_original_id = int(original_id)
    except ValueError:
        print(f"Warning: Skipping non-integer class ID key '{original_id}' in original names.", file=sys.stderr)
        continue

    if name in VEGETABLE_NAMES_TO_KEEP:
        vegetable_original_ids.append(current_original_id)
        vegetable_new_names.append(name)
        original_id_to_new_id[current_original_id] = new_id_counter
        new_id_counter += 1

if not vegetable_original_ids:
    print("\nWarning: No vegetable classes found based on VEGETABLE_NAMES_TO_KEEP list.", file=sys.stderr)
    print("Please check the class names in the list against the data.yaml.", file=sys.stderr)
    # Decide whether to exit or continue creating an empty dataset
    # sys.exit(1) # Optional: exit if no target classes found

print(f"Number of vegetable classes to keep: {len(vegetable_original_ids)}")
if vegetable_new_names:
    print(f"Vegetable classes being extracted: {vegetable_new_names}")
# print(f"Original ID to New ID mapping: {original_id_to_new_id}")

print("\n--- 3. Filtering Labels and Images, Copying, and Remapping IDs ---")

label_files = list(original_train_labels_dir.glob("*.txt"))
copied_image_count = 0
processed_label_files = 0

if not label_files:
     print(f"Warning: No label files (.txt) found in {original_train_labels_dir}.", file=sys.stderr)

for label_file_path in tqdm(label_files, desc="Processing label files"):
    image_name_stem = label_file_path.stem
    # Find the corresponding image file (handle common extensions)
    original_image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        potential_image_path = original_train_images_dir / (image_name_stem + ext)
        if potential_image_path.exists():
            original_image_path = potential_image_path
            break

    if not original_image_path:
        print(f"\nWarning: No corresponding image found for label file: {label_file_path.name} in {original_train_images_dir}", file=sys.stderr)
        continue

    new_label_lines = []
    valid_label_file = True
    try:
        with open(label_file_path, 'r') as f_label:
            lines = f_label.readlines()

        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if not parts:
                continue # Skip empty lines

            try:
                original_class_id = int(parts[0])
                # Check if bounding box coordinates are valid floats
                coords = [float(p) for p in parts[1:]]
                if len(coords) != 4:
                    raise ValueError("Incorrect number of coordinates")

            except (ValueError, IndexError) as e:
                print(f"\nWarning: Invalid line format in {label_file_path.name}, line {line_num+1}: '{line.strip()}'. Error: {e}. Skipping line.", file=sys.stderr)
                continue # Skip malformed lines

            # Check if this class ID is one of the vegetables we want to keep
            if original_class_id in original_id_to_new_id:
                # Get the new sequential ID
                new_class_id = original_id_to_new_id[original_class_id]
                # Reconstruct the line with the new class ID
                new_line = f"{new_class_id} {' '.join(parts[1:])}"
                new_label_lines.append(new_line)

    except Exception as e:
        print(f"\nError: Failed to read or process label file {label_file_path.name}: {e}", file=sys.stderr)
        valid_label_file = False # Mark as invalid if reading failed

    # If the file was processed successfully and contains at least one vegetable annotation:
    if valid_label_file and new_label_lines:
        processed_label_files += 1
        # Write the filtered annotations to the new label file
        new_label_file_path = new_train_labels_dir / label_file_path.name
        try:
            with open(new_label_file_path, 'w', encoding='utf-8') as f_new_label:
                f_new_label.write("\n".join(new_label_lines))

            # Copy the corresponding image file
            new_image_path = new_train_images_dir / original_image_path.name
            copyfile(original_image_path, new_image_path)
            copied_image_count += 1
        except Exception as e:
             print(f"\nError: Failed to write label or copy image for {label_file_path.name}: {e}", file=sys.stderr)


print(f"\nProcessing complete.")
print(f" - Processed {len(label_files)} original label files.")
print(f" - Created {processed_label_files} new label files with vegetable annotations.")
print(f" - Copied {copied_image_count} corresponding image files.")

if copied_image_count == 0 and len(label_files) > 0 and len(vegetable_original_ids) > 0:
     print("\nWarning: No images were copied. This might happen if:", file=sys.stderr)
     print("  - The specified vegetable classes were not present in the 'train' annotations.", file=sys.stderr)
     print("  - There was an issue matching label files to image files.", file=sys.stderr)
     print("  - The VEGETABLE_NAMES_TO_KEEP list needs adjustment.", file=sys.stderr)


print("\n--- 4. Creating New data.yaml ---")

# Define the structure for the new YAML file
new_data_config = {
    # Path relative to where the training script might be run, or absolute.
    # Using a path relative to the YAML file itself is often robust.
    # 'path': str(OUTPUT_DIR.resolve()), # Option 1: Absolute path
    'path': '..',  # Option 2: Relative path assuming YAML is in root, data in images/labels
    'train': 'images/train', # Path relative to 'path'
    'val': 'images/train',   # Using train data also for validation as we only filtered train
    'test': '',              # No separate test set created

    # Class information
    'nc': len(vegetable_new_names),
    'names': vegetable_new_names, # List of names ordered by the new IDs (0, 1, ...)
}

try:
    with open(new_data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_data_config, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
    print(f"New data.yaml created successfully at: {new_data_yaml_path}")
except Exception as e:
    print(f"Error: Failed to write new data.yaml: {e}", file=sys.stderr)

print("\n--- All tasks finished ---")
print(f"The new vegetable-only YOLO dataset is located in: {OUTPUT_DIR}")
print(f"The YAML configuration file for training is: {new_data_yaml_path}")