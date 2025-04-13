import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import asyncpg
import yaml  # for reading data.yaml
from pathlib import Path
from PIL import Image
import json  # for PostgreSQL JSONB type

# Database connection settings
MONGO_HOST = os.getenv("SOURCE_MONGO_HOST", "localhost")
# ... (other DB connection settings) ...
POSTGRES_DB = os.getenv("SOURCE_POSTGRES_DB", "source_annotations")

SOURCE_YOLO_DIR = Path("/app/setup_data/source_yolo_dataset")  # path inside container

async def load_yolo_to_dbs():
    # --- MongoDB connection and cleanup ---
    mongo_uri = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"
    mongo_client = AsyncIOMotorClient(mongo_uri)
    mongo_db = mongo_client[MONGO_DB]
    image_collection = mongo_db["images"]
    await image_collection.delete_many({})
    print("Cleared MongoDB 'images' collection.")

    # --- PostgreSQL connection and cleanup ---
    pg_conn = None
    try:
        pg_conn = await asyncpg.connect(
            user=POSTGRES_USER, password=POSTGRES_PASS,
            database=POSTGRES_DB, host=POSTGRES_HOST, port=POSTGRES_PORT
        )
        await pg_conn.execute("DELETE FROM image_annotations;")
        # Clear table containing data.yaml contents (if exists)
        # await pg_conn.execute("DELETE FROM dataset_metadata;")
        print("Cleared PostgreSQL 'image_annotations' table.")

        # --- Load data.yaml ---
        data_yaml_path = SOURCE_YOLO_DIR / "data.yaml"
        class_names = []
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data_yaml_content = yaml.safe_load(f)
                class_names = data_yaml_content.get('names', [])
                print(f"Loaded class names from data.yaml: {class_names}")
                # Optionally save other metadata to PostgreSQL
                # nc = data_yaml_content.get('nc')
                # await pg_conn.execute("INSERT INTO dataset_metadata (key, value) VALUES ($1, $2)", 'class_names', json.dumps(class_names))
        else:
            print("Warning: data.yaml not found.")

        # --- Scan images and labels for DB insertion ---
        images_to_insert_mongo = []
        annotations_to_insert_pg = []

        for split in ["train", "valid"]:  # add "test" if available
            image_dir = SOURCE_YOLO_DIR / split / "images"
            label_dir = SOURCE_YOLO_DIR / split / "labels"

            if not image_dir.is_dir():
                print(f"Warning: Image directory not found: {image_dir}")
                continue

            print(f"Processing {split} set...")
            for image_path in image_dir.glob('*.*'):  # scan image files
                if not image_path.is_file(): continue

                label_path = label_dir / (image_path.stem + ".txt")  # corresponding label file

                # Prepare image metadata for MongoDB
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                    mongo_doc = {
                        "_id": f"{split}_{image_path.stem}",  # unique ID including split name
                        "filename": image_path.name,
                        "split": split,
                        "metadata": {"width": width, "height": height},
                        "source": "yolo_import_mongo"
                    }
                    images_to_insert_mongo.append(mongo_doc)
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
                    continue  # skip if image cannot be read

                # Prepare label data for PostgreSQL
                if label_path.exists() and label_path.is_file():
                    annotations = []
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    # YOLO format: class_id cx cy w h
                                    annotations.append({
                                        "class_id": int(parts[0]),
                                        "x_center": float(parts[1]),
                                        "y_center": float(parts[2]),
                                        "width": float(parts[3]),
                                        "height": float(parts[4])
                                    })
                        # Store in JSONB format
                        pg_record = (image_path.name, json.dumps(annotations))
                        annotations_to_insert_pg.append(pg_record)
                    except Exception as e:
                        print(f"Error reading/parsing label {label_path}: {e}")
                # Images without labels will still be added to MongoDB (treated as unannotated images)

        # --- Bulk insertion into DBs ---
        if images_to_insert_mongo:
            print(f"Inserting {len(images_to_insert_mongo)} documents into MongoDB...")
            await image_collection.insert_many(images_to_insert_mongo)
            print("MongoDB insertion complete.")
        else:
            print("No images found to insert into MongoDB.")

        if annotations_to_insert_pg:
            print(f"Inserting {len(annotations_to_insert_pg)} records into PostgreSQL...")
            await pg_conn.executemany(
                "INSERT INTO image_annotations (image_filename, annotations, source) VALUES ($1, $2, 'yolo_import_postgres') ON CONFLICT (image_filename) DO UPDATE SET annotations = EXCLUDED.annotations, source = EXCLUDED.source",
                annotations_to_insert_pg
            )
            print("PostgreSQL insertion complete.")
        else:
            print("No annotations found to insert into PostgreSQL.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if pg_conn: await pg_conn.close()
        mongo_client.close()

async def main():
    print("Starting YOLO dataset loading into temporary DBs...")
    await load_yolo_to_dbs()
    print("Finished loading YOLO dataset into temporary DBs.")

if __name__ == "__main__":
    # PyYAML required
    asyncio.run(main())