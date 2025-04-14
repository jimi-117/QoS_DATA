# qos_data/scripts/load_sources_to_db.py
import os
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg
import yaml
from pathlib import Path
from PIL import Image
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection Settings ---
MONGO_HOST = os.getenv("SOURCE_MONGO_HOST", "source_mongodb_dev") # Use service name
MONGO_PORT = int(os.getenv("SOURCE_MONGO_PORT", 27017))
MONGO_DB_NAME = os.getenv("SOURCE_MONGO_DB", "source_images")
# Add MONGO_USER/PASS if using authentication

POSTGRES_HOST = os.getenv("SOURCE_POSTGRES_HOST", "source_postgres_dev") # Use service name
POSTGRES_PORT = int(os.getenv("SOURCE_POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("SOURCE_POSTGRES_USER", "etl_user")
POSTGRES_PASS = os.getenv("SOURCE_POSTGRES_PASSWORD", "etl_password")
POSTGRES_DB_NAME = os.getenv("SOURCE_POSTGRES_DB", "source_annotations")

# --- Source Data Path ---
# Assumes the source YOLO dataset is named 'Fruits-detection' as per image
# Adjust this path if your dataset directory name is different
SOURCE_YOLO_DIR = Path("/app/setup_data/source_dataset/Fruits-detection")

async def clear_dbs(mongo_db, pg_conn):
    """Clear existing data from temporary databases."""
    logging.info("Clearing temporary databases...")
    image_collection = mongo_db["images"]
    await image_collection.delete_many({})
    logging.info("Cleared MongoDB 'images' collection.")
    if pg_conn:
        await pg_conn.execute("DELETE FROM image_annotations;")
        # Optional: Clear metadata table if it exists
        # await pg_conn.execute("DELETE FROM dataset_metadata;")
        logging.info("Cleared PostgreSQL 'image_annotations' table.")

async def load_yolo_to_dbs():
    """Loads the source YOLO dataset into temporary MongoDB and PostgreSQL."""
    mongo_client = None
    pg_conn = None
    try:
        # --- Connect to Databases ---
        logging.info(f"Connecting to MongoDB at {MONGO_HOST}:{MONGO_PORT}...")
        mongo_uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}" # Add auth if needed
        mongo_client = AsyncIOMotorClient(mongo_uri)
        mongo_db = mongo_client[MONGO_DB_NAME]
        logging.info("MongoDB connected.")

        logging.info(f"Connecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}...")
        pg_conn = await asyncpg.connect(
            user=POSTGRES_USER, password=POSTGRES_PASS,
            database=POSTGRES_DB_NAME, host=POSTGRES_HOST, port=POSTGRES_PORT
        )
        logging.info("PostgreSQL connected.")

        # --- Clear existing data ---
        await clear_dbs(mongo_db, pg_conn)

        # --- Load class names from data.yaml ---
        data_yaml_path = SOURCE_YOLO_DIR / "data.yaml"
        class_names = []
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data_yaml_content = yaml.safe_load(f)
                class_names = data_yaml_content.get('names', [])
                logging.info(f"Loaded {len(class_names)} class names from data.yaml: {class_names}")
                # Optional: Store class names in PG metadata table
                # await pg_conn.execute("INSERT INTO dataset_metadata (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                #                       'class_names', json.dumps(class_names))
        else:
            logging.warning(f"data.yaml not found at {data_yaml_path}")

        # --- Scan dataset and prepare data for insertion ---
        images_to_insert_mongo = []
        annotations_to_insert_pg = []
        processed_count = 0

        for split in ["train", "valid"]: # Adjust splits as needed
            image_dir = SOURCE_YOLO_DIR / split / "images"
            label_dir = SOURCE_YOLO_DIR / split / "labels"

            if not image_dir.is_dir():
                logging.warning(f"Directory not found: {image_dir}")
                continue

            logging.info(f"Scanning {split} set in {image_dir}...")
            for image_path in image_dir.iterdir():
                if not image_path.is_file(): continue

                image_id_stem = image_path.stem # Use filename without extension as ID
                label_path = label_dir / (image_id_stem + ".txt")

                # Prepare MongoDB document
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                    mongo_doc = {
                        "_id": image_id_stem, # Use stem as MongoDB ID
                        "image_id": image_id_stem, # Also store as a field if preferred
                        "filename": image_path.name,
                        "split": split,
                        "metadata": {"width": width, "height": height},
                        "source": "yolo_import_mongo"
                    }
                    images_to_insert_mongo.append(mongo_doc)
                except Exception as e:
                    logging.error(f"Failed to process image {image_path}: {e}")
                    continue # Skip this image if metadata extraction fails

                # Prepare PostgreSQL record
                annotations = []
                if label_path.exists() and label_path.is_file():
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    annotations.append({
                                        "class_id": int(parts[0]),
                                        "x_center": float(parts[1]),
                                        "y_center": float(parts[2]),
                                        "width": float(parts[3]),
                                        "height": float(parts[4])
                                    })
                        # Use actual filename as the key for PG table
                        pg_record = (image_path.name, json.dumps(annotations), split)
                        annotations_to_insert_pg.append(pg_record)
                    except Exception as e:
                        logging.error(f"Failed to process label {label_path}: {e}")
                # else: Allow images without labels, don't add to PG list

                processed_count += 1
                if processed_count % 100 == 0:
                    logging.info(f"Scanned {processed_count} images...")

        # --- Bulk insert into databases ---
        if images_to_insert_mongo:
            logging.info(f"Inserting {len(images_to_insert_mongo)} documents into MongoDB...")
            try:
                await mongo_db["images"].insert_many(images_to_insert_mongo, ordered=False) # Allow continuing on duplicates if _id is reused
                logging.info("MongoDB insertion complete.")
            except Exception as e: # Catch potential bulk write errors
                 logging.error(f"MongoDB bulk insert failed: {e}")
        else:
            logging.info("No image metadata to insert into MongoDB.")

        if annotations_to_insert_pg and pg_conn:
            logging.info(f"Inserting {len(annotations_to_insert_pg)} records into PostgreSQL...")
            try:
                # Use ON CONFLICT clause to handle potential reruns or duplicate filenames
                await pg_conn.executemany(
                    """
                    INSERT INTO image_annotations (image_filename, annotations, split, source)
                    VALUES ($1, $2, $3, 'yolo_import_postgres')
                    ON CONFLICT (image_filename) DO UPDATE SET
                      annotations = EXCLUDED.annotations,
                      split = EXCLUDED.split,
                      source = EXCLUDED.source;
                    """,
                    annotations_to_insert_pg
                )
                logging.info("PostgreSQL insertion complete.")
            except Exception as e:
                logging.error(f"PostgreSQL bulk insert failed: {e}")
        else:
            logging.info("No annotations to insert into PostgreSQL.")

    except Exception as e:
        logging.exception(f"An error occurred during the loading process: {e}") # Log full traceback
    finally:
        # --- Close connections ---
        if pg_conn:
            await pg_conn.close()
            logging.info("PostgreSQL connection closed.")
        if mongo_client:
            mongo_client.close()
            logging.info("MongoDB connection closed.")

async def main():
    logging.info("Starting: Load source YOLO dataset into temporary DBs")
    await load_yolo_to_dbs()
    logging.info("Finished: Load source YOLO dataset into temporary DBs")

if __name__ == "__main__":
    # Ensure necessary packages are available
    # Consider adding checks or explicitly listing in Dockerfile build stage
    asyncio.run(main())