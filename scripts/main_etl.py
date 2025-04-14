# qos_data/scripts/main_etl.py

import os
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg
import pandas as pd # Optional, used here for deduplication example
import yaml
from pathlib import Path
import shutil
import json
from sklearn.model_selection import train_test_split # For splitting data
from clearml import Dataset as ClearMLDataset # Use alias to avoid name clash
import subprocess # For running DVC commands
import random # For shuffling
from PIL import Image # For getting image dimensions
from datetime import datetime # For potential commit messages

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Log to console
)
logger = logging.getLogger(__name__)

# --- Database Connection Settings ---
# Fetched from environment variables defined in .env.dev
MONGO_HOST = os.getenv("SOURCE_MONGO_HOST", "source_mongodb_dev")
MONGO_PORT = int(os.getenv("SOURCE_MONGO_PORT", 27017))
MONGO_DB_NAME = os.getenv("SOURCE_MONGO_DB", "source_images_temp")
# Add MONGO_USER/PASS if using authentication
# MONGO_USER = os.getenv("SOURCE_MONGO_USER")
# MONGO_PASS = os.getenv("SOURCE_MONGO_PASSWORD")

POSTGRES_HOST = os.getenv("SOURCE_POSTGRES_HOST", "source_postgres_dev")
POSTGRES_PORT = int(os.getenv("SOURCE_POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("SOURCE_POSTGRES_USER", "etl_user")
POSTGRES_PASS = os.getenv("SOURCE_POSTGRES_PASSWORD", "etl_password")
POSTGRES_DB_NAME = os.getenv("SOURCE_POSTGRES_DB", "source_annotations_temp")

# --- Input Data Paths (inside container) ---
PRE_EXTRACTED_LVIS_DIR = Path("/app/data/processed/LVIS_vegetables")
PRE_EXTRACTED_SCRAPING_DIR = Path("/app/data/processed/scraping")
# Base path to the original images from the source YOLO dataset (needed for copying files)
# Adjust 'Fruits-detection' if your dataset name differs
SOURCE_YOLO_BASE = Path("/app/setup_data/source_yolo_dataset/Fruits-detection")
SOURCE_YOLO_DATA_YAML = SOURCE_YOLO_BASE.parent / "data.yaml" # Path to the data.yaml of the *source* yolo dataset

# --- Output Data Path (inside container) ---
FINAL_OUTPUT_DIR = Path("/app/data/final_yolo_dataset")

# --- ClearML Settings ---
CLEARML_PROJECT_NAME = os.getenv("CLEARML_PROJECT_NAME", "QoS_Project")
CLEARML_DATASET_NAME = os.getenv("CLEARML_DATASET_NAME", "Ingredients_Final")

# --- Constants ---
RANDOM_STATE = 42 # For reproducible splits


# --- Helper Functions ---

async def extract_data_from_temp_dbs() -> list:
    """
    -------- C2 ---------
    Extracts data (originally from the source YOLO dataset) that was loaded
    into the temporary MongoDB and PostgreSQL databases.

    Returns:
        list: A list of dictionaries, each representing an image sample.
              Keys: 'filename', 'original_path', 'annotations', 'metadata', 'source_origin'.
    """
    logger.info("Extracting data from temporary DBs (source: initial YOLO dataset)...")
    mongo_client = None
    pg_conn = None
    combined_data = []

    try:
        # Connect to MongoDB
        mongo_uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"
        mongo_client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=5000) # Add timeout
        # Verify connection
        await mongo_client.admin.command('ping')
        mongo_db = mongo_client[MONGO_DB_NAME]
        image_collection = mongo_db["images"]
        logger.info("MongoDB connected for extraction.")

        # Connect to PostgreSQL
        pg_conn = await asyncpg.connect(
            user=POSTGRES_USER, password=POSTGRES_PASS,
            database=POSTGRES_DB_NAME, host=POSTGRES_HOST, port=POSTGRES_PORT,
            timeout=10 # connection timeout
        )
        logger.info("PostgreSQL connected for extraction.")

        # Fetch annotations into a map for quick lookup
        annotations_map = {}
        try:
            rows = await pg_conn.fetch("SELECT image_filename, annotations, split FROM image_annotations")
            for row in rows:
                try:
                    # Use actual filename from PG as the key
                    annotations_map[row['image_filename']] = {
                        'annotations': json.loads(row['annotations']),
                        'original_split': row['split'] # Keep original split info if needed
                    }
                except json.JSONDecodeError:
                     logger.warning(f"Could not decode annotations for {row['image_filename']}")
                     annotations_map[row['image_filename']] = {'annotations': [], 'original_split': 'unknown'}
        except Exception as e:
             logger.exception(f"Error fetching annotations from PostgreSQL: {e}")


        # Fetch image data and combine with annotations
        processed_filenames = set() # To avoid duplicates if Mongo has extras
        async for doc in image_collection.find({}, {"_id": 0}): # Exclude mongo _id
            filename = doc.get('filename')
            if filename and filename not in processed_filenames:
                annotation_info = annotations_map.get(filename, {'annotations': [], 'original_split': 'unknown'})
                original_split = annotation_info.get('original_split', doc.get('split', 'unknown'))

                # Construct the original path based on the initial YOLO dataset structure
                original_path = SOURCE_YOLO_BASE / original_split / "images" / filename

                # Check if the original image file actually exists before adding
                if original_path.exists():
                    combined_data.append({
                        'filename': filename,
                        'original_path': original_path, # Store as Path object
                        'annotations': annotation_info['annotations'],
                        'metadata': doc.get('metadata', {}),
                        'source_origin': 'self_yolo_db' # Mark the origin
                    })
                    processed_filenames.add(filename)
                else:
                    logger.warning(f"Original image file not found for {filename} at {original_path}. Skipping this item from DB source.")

        logger.info(f"Successfully extracted {len(combined_data)} items from temporary DBs.")

    except Exception as e:
        logger.exception(f"Error extracting data from temp DBs: {e}")
    finally:
        if pg_conn:
            await pg_conn.close()
            logger.info("PostgreSQL connection closed.")
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed.")

    return combined_data


def process_yolo_directory(dataset_base_path: Path, source_origin_tag: str) -> list:
    """
    Reads a YOLO formatted dataset directory, relying *only* on data.yaml for structure.
    Returns data in the standard internal format.
    """
    processed_items = []
    logger.info(f"Processing YOLO dataset at: {dataset_base_path} (Source: {source_origin_tag}) using data.yaml")

    if not dataset_base_path.is_dir():
        logger.error(f"Dataset base directory not found: {dataset_base_path}")
        return processed_items

    # --- Read data.yaml ---
    data_yaml_path = dataset_base_path / "data.yaml"
    if not data_yaml_path.exists():
        logger.error(f"Mandatory data.yaml not found in {dataset_base_path}. Cannot process this dataset.")
        return processed_items

    try:
        with open(data_yaml_path, 'r') as f:
            data_yaml_content = yaml.safe_load(f)

        # Basic validation of data.yaml content
        if not isinstance(data_yaml_content, dict):
            logger.error(f"Invalid YAML format in {data_yaml_path}")
            return processed_items
        # Check for mandatory keys needed to find images (train/val at least)
        if not ('path' in data_yaml_content and ('train' in data_yaml_content or 'val' in data_yaml_content)):
             logger.error(f"data.yaml at {data_yaml_path} is missing mandatory keys ('path', and 'train' or 'val').")
             return processed_items

        class_names = data_yaml_content.get('names', [])
        logger.info(f"  Loaded {len(class_names)} class names from {data_yaml_path}")

        # Resolve the dataset's effective root path specified within data.yaml
        yaml_specified_path_str = data_yaml_content.get('path', '.')
        # Path is relative to the data.yaml file's directory
        yaml_base_path = (data_yaml_path.parent / yaml_specified_path_str).resolve()
        logger.info(f"  Resolved dataset base path from YAML: {yaml_base_path}")

    except Exception as e:
        logger.exception(f"Error reading or parsing data.yaml at {data_yaml_path}: {e}")
        return processed_items

    # --- Process splits defined in data.yaml ---
    processed_filenames_in_source = set() # Track filenames within this source
    for split in ["train", "val", "test"]:
        relative_image_dir_str = data_yaml_content.get(split) # e.g., "train/images" or "../images/train"
        if not relative_image_dir_str:
            # logger.info(f"  Split '{split}' not defined in {data_yaml_path}, skipping.")
            continue

        # Resolve absolute path to image directory for this split
        absolute_image_dir = (yaml_base_path / relative_image_dir_str).resolve()

        if not absolute_image_dir.is_dir():
            logger.warning(f"  Image directory for split '{split}' not found at resolved path: {absolute_image_dir}")
            continue

        logger.info(f"  Scanning {split} images in {absolute_image_dir}...")

        # Assume labels are in a parallel directory relative to the image dir base path
        # e.g., if images are in 'foo/train/images', labels are expected in 'foo/train/labels'
        # This relies on a consistent convention.
        try:
            label_dir_relative_to_yaml_base = Path(*Path(relative_image_dir_str).parent.parts) / "labels"
            absolute_label_dir = (yaml_base_path / label_dir_relative_to_yaml_base).resolve()
            if not absolute_label_dir.is_dir():
                 logger.warning(f"  Label directory not found at expected location: {absolute_label_dir} for split '{split}'. Labels will be empty.")
                 absolute_label_dir = None # Mark as not found
        except Exception: # Handle potential errors if path structure is unexpected
             logger.warning(f"  Could not determine label directory for split '{split}'. Labels will be empty.")
             absolute_label_dir = None

        for image_path in absolute_image_dir.iterdir():
            if not image_path.is_file() or image_path.name.startswith('.'):
                 continue # Skip directories and hidden files

            filename = image_path.name
            # Avoid processing the same file twice if listed in multiple splits in yaml?
            if filename in processed_filenames_in_source:
                logger.warning(f"Duplicate filename '{filename}' encountered in source '{source_origin_tag}'. Skipping.")
                continue
            processed_filenames_in_source.add(filename)

            # Determine corresponding label file path
            label_path = None
            if absolute_label_dir:
                 label_filename = image_path.stem + ".txt"
                 label_path = absolute_label_dir / label_filename

            annotations = []
            # Read label file if the directory and file exist
            if label_path and label_path.exists() and label_path.is_file():
                try:
                    with open(label_path, 'r') as f:
                        for line_num, line in enumerate(f):
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    # Basic validation for coordinates and class_id
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                                         logger.warning(f"Invalid bbox coordinate values in {label_path}, line {line_num+1}: {line.strip()}")
                                         continue # Skip invalid annotation line
                                    annotations.append({
                                        "class_id": class_id,
                                        "x_center": x_center, "y_center": y_center,
                                        "width": width, "height": height
                                    })
                                except ValueError:
                                     logger.warning(f"Invalid number format in label {label_path}, line {line_num+1}: '{line.strip()}'")
                            elif line.strip(): # Log non-empty lines that don't conform
                                logger.warning(f"Unexpected format in label {label_path}, line {line_num+1}: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"Failed read/parse label {label_path}: {e}")

            # Get image metadata
            metadata = {}
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                metadata = {"width": width, "height": height}
            except Exception as e:
                logger.error(f"Failed read metadata for {image_path}: {e}")
                # If metadata fails, maybe skip or add with empty metadata? Decide strategy.
                # For now, add with empty metadata:
                metadata = {}


            processed_items.append({
                'filename': filename,
                'original_path': image_path, # Keep original path for copying later
                'annotations': annotations,
                'metadata': metadata,
                'source_origin': source_origin_tag,
                'original_split_from_file': split # Record the split found via data.yaml
            })

    logger.info(f"Finished processing {dataset_base_path}. Found {len(processed_items)} valid items based on data.yaml.")
    return processed_items

async def load_pre_extracted_data() -> list:
    """Loads data from pre-extracted YOLO formatted directories using data.yaml."""
    logger.info("Loading data from pre-extracted YOLO dataset directories (using data.yaml)...")
    all_pre_extracted_data = []

    # Process LVIS directory
    lvis_data = process_yolo_directory(PRE_EXTRACTED_LVIS_DIR, 'lvis_vegetables')
    all_pre_extracted_data.extend(lvis_data)

    # Process Scraping directory
    scraping_data = process_yolo_directory(PRE_EXTRACTED_SCRAPING_DIR, 'scraping')
    all_pre_extracted_data.extend(scraping_data)

    logger.info(f"Loaded a total of {len(all_pre_extracted_data)} items from pre-extracted sources.")
    return all_pre_extracted_data

def generate_yolo_output(output_dir: Path, data_items: list, split_name: str):
    """Generates YOLO structure (images and labels) for a given data split."""
    image_dir = output_dir / split_name / "images"
    label_dir = output_dir / split_name / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating output for '{split_name}' set ({len(data_items)} items)...")
    copied_count = 0
    label_count = 0
    skipped_copy = 0
    skipped_label = 0

    for item in data_items:
        original_image_path = item.get('original_path') # Should be a Path object
        filename = item.get('filename')
        annotations = item.get('annotations', [])

        if not original_image_path or not filename:
            logger.warning(f"Skipping item due to missing path or filename: {item}")
            skipped_copy += 1
            skipped_label += 1
            continue

        # Ensure original_image_path is Path object if loaded differently
        if isinstance(original_image_path, str):
             original_image_path = Path(original_image_path)

        if not original_image_path.exists():
             logger.warning(f"Source image file not found: {original_image_path}. Skipping item {filename}.")
             skipped_copy += 1
             skipped_label += 1
             continue

        # Define destination paths
        image_dest_path = image_dir / filename
        label_filename = Path(filename).stem + ".txt" # Use stem for label filename
        label_dest_path = label_dir / label_filename

        # Copy image file
        try:
            # Avoid re-copying if file already exists (e.g., if running script multiple times without clearing)
            if not image_dest_path.exists():
                 shutil.copy2(str(original_image_path), str(image_dest_path))
            copied_count +=1
        except Exception as e:
            logger.error(f"Failed to copy image {original_image_path} to {image_dest_path}: {e}")
            skipped_copy += 1
            skipped_label += 1
            continue # Skip label generation if image copy failed

        # Generate label file only if annotations exist
        if annotations:
            try:
                with open(label_dest_path, "w") as f:
                    for ann in annotations:
                        # Ensure annotation format is correct before writing
                        if all(k in ann for k in ['class_id', 'x_center', 'y_center', 'width', 'height']):
                            line = (f"{ann['class_id']} "
                                    f"{ann['x_center']:.6f} {ann['y_center']:.6f} "
                                    f"{ann['width']:.6f} {ann['height']:.6f}\n")
                            f.write(line)
                        else:
                            logger.warning(f"Invalid annotation format in item {filename}: {ann}. Skipping annotation.")
                label_count += 1
            except Exception as e:
                logger.error(f"Failed to write label file {label_dest_path}: {e}")
                skipped_label += 1
        else:
            # If no annotations, ensure no leftover label file exists
            if label_dest_path.exists():
                 label_dest_path.unlink()
            # label_count remains unchanged

    logger.info(f"Finished '{split_name}' set: Processed {len(data_items)} items. Copied {copied_count} images (skipped {skipped_copy}), generated {label_count} label files (skipped {skipped_label}).")


def generate_data_yaml(output_dir: Path, class_names: list, train_path_rel: str, val_path_rel: str, test_path_rel: str | None):
    """Generates the data.yaml file in the output directory."""
    logger.info(f"Generating data.yaml in {output_dir}...")
    # Get absolute path for 'path' entry, but use relative paths for train/val/test
    data_yaml_content = {
        'path': str(output_dir.resolve()), # Absolute path to the dataset root
        'train': train_path_rel,           # e.g., 'train/images'
        'val': val_path_rel,               # e.g., 'valid/images'
        'nc': len(class_names),
        'names': class_names               # List of class names
    }
    if test_path_rel:
        data_yaml_content['test'] = test_path_rel # e.g., 'test/images'

    data_yaml_path = output_dir / "data.yaml"
    try:
        with open(data_yaml_path, 'w') as f:
            # Use default_flow_style=None for block style (more readable)
            yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=None)
        logger.info(f"Generated data.yaml at {data_yaml_path}")
    except Exception as e:
        logger.error(f"Failed to generate data.yaml: {e}")


async def main_etl_process():
    """Main ETL orchestration function."""
    start_time = datetime.now()
    logger.info(f"--- Main ETL Process Started at {start_time} ---")

    # --- Step 1: Aggregate all data sources ---
    logger.info("Step 1: Aggregating data sources...")
    # Extract data loaded from the initial YOLO dataset (now in temp DBs)
    own_data_task = extract_data_from_temp_dbs()
    # Load data from the directories containing pre-extracted LVIS and Scraping datasets
    other_data_task = load_pre_extracted_data()
    # Run concurrently
    own_data, other_data = await asyncio.gather(own_data_task, other_data_task)

    all_data = own_data + other_data
    logger.info(f"Total integrated data items before deduplication: {len(all_data)}")

    if not all_data:
        logger.error("No data found from any source. Exiting ETL.")
        return

    # --- Step 2: Prepare data for splitting (Deduplication) ---
    logger.info("Step 2: Preparing data for splitting (deduplication by filename)...")
    # Use filename as a key for deduplication, keeping the first encountered
    unique_data_map = {item['filename']: item for item in reversed(all_data)} # Keep first by reversing then taking first
    final_data_pool = list(unique_data_map.values())
    logger.info(f"Total unique items after deduplication: {len(final_data_pool)}")
    random.seed(RANDOM_STATE) # Set seed before shuffling
    random.shuffle(final_data_pool)

    # --- Step 3: Split the aggregated dataset (70:20:10) ---
    logger.info("Step 3: Splitting the aggregated dataset...")
    train_data, valid_data, test_data = [], [], []
    try:
        if len(final_data_pool) < 3:
             logger.warning("Dataset too small for train/valid/test split. Assigning all to train.")
             train_data = final_data_pool
        else:
            # Split into Train (70%) and Temp (30%)
            train_data, temp_data = train_test_split(
                final_data_pool,
                test_size=0.3,
                random_state=RANDOM_STATE
            )
            # Split Temp into Valid (20% of original) and Test (10% of original)
            # Validation size relative to temp_data = 0.2 / (0.2 + 0.1) = 0.2 / 0.3 = 2/3
            # Test size relative to temp_data = 1/3
            if len(temp_data) < 2: # Need at least 2 items in temp to split further
                logger.warning("Not enough items for valid/test split. Assigning remaining to valid.")
                valid_data = temp_data
            else:
                relative_test_size = 0.1 / 0.3 # Test proportion within the 30% temp data
                valid_data, test_data = train_test_split(
                    temp_data,
                    test_size=relative_test_size,
                    random_state=RANDOM_STATE
                )
        logger.info(f"Split sizes - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    except Exception as e:
        logger.exception(f"Error during data splitting: {e}")
        return

    # --- Step 4: Generate final YOLO output structure ---
    logger.info(f"Step 4: Generating final YOLO dataset at {FINAL_OUTPUT_DIR}...")

    # --- Determine Class Names ---
    class_names = ['unknown'] # Default
    try:
        if SOURCE_YOLO_DATA_YAML.exists():
             with open(SOURCE_YOLO_DATA_YAML, 'r') as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, dict) and 'names' in yaml_content and isinstance(yaml_content['names'], list):
                    class_names = yaml_content['names']
                    logger.info(f"Using class names from source data.yaml: {class_names}")
                else:
                    logger.warning(f"Could not find valid 'names' list in {SOURCE_YOLO_DATA_YAML}. Using default.")
        else:
            logger.warning(f"Source data.yaml not found at {SOURCE_YOLO_DATA_YAML}. Using default class names. Define class names manually if needed.")
        # Optional: Add logic here to verify class IDs found in annotations are valid against class_names list
    except Exception as e:
        logger.exception(f"Error loading class names from {SOURCE_YOLO_DATA_YAML}. Using default. Error: {e}")

    # --- Clear Output Directory ---
    if FINAL_OUTPUT_DIR.exists():
        logger.warning(f"Clearing existing output directory: {FINAL_OUTPUT_DIR}")
        try:
            shutil.rmtree(FINAL_OUTPUT_DIR)
        except OSError as e:
             logger.error(f"Error removing directory {FINAL_OUTPUT_DIR}: {e}")
             return # Stop if we can't clear the output directory
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Generate Split Files ---
    generate_yolo_output(FINAL_OUTPUT_DIR, train_data, "train")
    generate_yolo_output(FINAL_OUTPUT_DIR, valid_data, "valid")
    generate_yolo_output(FINAL_OUTPUT_DIR, test_data, "test") # Function handles empty list

    # --- Generate data.yaml ---
    generate_data_yaml(
        FINAL_OUTPUT_DIR,
        class_names,
        "train/images", # Relative path for YAML standard
        "valid/images", # Relative path
        "test/images" if test_data else None # Relative path or None
    )

    # --- Step 5: DVC Tracking ---
    logger.info("Step 5: Tracking final dataset with DVC...")
    # Path should be relative to the git repo root (which is /app in container)
    dvc_target = str(FINAL_OUTPUT_DIR.relative_to(Path('/app')))
    try:
        # Run dvc add. Ensure DVC is initialized and remote (like dagshub or s3) is configured.
        # Using capture_output=True to get stdout/stderr if needed
        # check=True raises CalledProcessError if command fails
        result = subprocess.run(['dvc', 'add', dvc_target], check=True, cwd='/app', capture_output=True, text=True)
        logger.info(f"DVC add {dvc_target} successful.")
        logger.debug(f"DVC add stdout:\n{result.stdout}")
        logger.debug(f"DVC add stderr:\n{result.stderr}") # Should be empty on success

        # Optional: Commit DVC files to Git (requires git setup in container/mount)
        # try:
        #     git_add_files = [f'{dvc_target}.dvc']
        #     gitignore_path = FINAL_OUTPUT_DIR / '.gitignore'
        #     if gitignore_path.exists(): # DVC might create/update a .gitignore
        #          git_add_files.append(str(gitignore_path.relative_to(Path('/app'))))
        #
        #     subprocess.run(['git', 'add'] + git_add_files, check=True, cwd='/app')
        #     commit_message = f"chore: Update processed dataset via ETL ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        #     # Check if there are staged changes before committing
        #     status_result = subprocess.run(['git', 'diff', '--staged', '--quiet'], cwd='/app')
        #     if status_result.returncode != 0: # If diff returns non-zero, there are changes
        #          subprocess.run(['git', 'commit', '-m', commit_message], check=True, cwd='/app')
        #          logger.info("Committed DVC file changes to Git.")
        #     else:
        #          logger.info("No DVC file changes to commit to Git.")
        # except Exception as git_e:
        #     logger.warning(f"Git commit failed (is git configured and are there changes?): {git_e}")

        # Optional: Push DVC data to remote storage
        # try:
        #     push_result = subprocess.run(['dvc', 'push'], check=True, cwd='/app', capture_output=True, text=True)
        #     logger.info("DVC push successful.")
        #     logger.debug(f"DVC push stdout:\n{push_result.stdout}")
        #     logger.debug(f"DVC push stderr:\n{push_result.stderr}")
        # except subprocess.CalledProcessError as push_e:
        #      logger.error(f"DVC 'push' command failed: {push_e}")
        #      logger.error(f"DVC push stdout:\n{push_e.stdout}")
        #      logger.error(f"DVC push stderr:\n{push_e.stderr}")
        # except Exception as e:
        #     logger.error(f"DVC push failed: {e}")

    except subprocess.CalledProcessError as dvc_e:
         logger.error(f"DVC 'add' command failed: {dvc_e}")
         logger.error(f"DVC add stdout:\n{dvc_e.stdout}")
         logger.error(f"DVC add stderr:\n{dvc_e.stderr}")
    except FileNotFoundError:
         logger.error("DVC command not found. Is DVC installed in the container?")
    except Exception as e:
        logger.error(f"DVC tracking failed: {e}")

    # --- Step 6: ClearML Upload ---
    logger.info("Step 6: Uploading final dataset to ClearML...")
    try:
        # Create a new version of the dataset (or the first version)
        clearml_dataset = ClearMLDataset.create(
            dataset_project=CLEARML_PROJECT_NAME,
            dataset_name=CLEARML_DATASET_NAME,
            # parent_datasets=[...] # Optional: Specify parent dataset ID for lineage
        )
        # Add the entire final dataset directory
        # Use verbose=True for more detailed logging during add_files
        clearml_dataset.add_files(path=FINAL_OUTPUT_DIR, verbose=True)
        # Upload the added files (this might take time)
        # Consider setting timeout for upload operation if needed
        clearml_dataset.upload(show_progress=True) # , timeout=3600) # Example: 1 hour timeout
        # Finalize (close) the dataset version - makes it immutable
        clearml_dataset.finalize()
        logger.info(f"Dataset finalized in ClearML. Dataset ID: {clearml_dataset.id}")

    except Exception as e:
        # Log the full exception traceback for debugging
        logger.exception(f"ClearML upload failed: {e}")


    end_time = datetime.now()
    logger.info(f"--- Main ETL Process Finished at {end_time} (Duration: {end_time - start_time}) ---")

if __name__ == "__main__":
    # Ensure necessary packages are installed via pyproject.toml & uv sync in Dockerfile
    # Key deps: motor, asyncpg, pyyaml, pillow, clearml, scikit-learn, pandas (optional)
    asyncio.run(main_etl_process())