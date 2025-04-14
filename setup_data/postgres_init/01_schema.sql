-- qos_data/setup_data/postgres_init/01_schema.sql
-- Create table to store annotations loaded from source YOLO dataset
CREATE TABLE IF NOT EXISTS public.image_annotations (
    image_filename VARCHAR(255) PRIMARY KEY, -- Use the actual filename as PK
    annotations JSONB NOT NULL, -- Store list of annotations [{class_id: int, bbox: [cx,cy,w,h]}, ...]
    split VARCHAR(10), -- Store original split (train/valid/test)
    source VARCHAR(100) DEFAULT 'yolo_import_postgres'
);

-- Optional: Table for data.yaml metadata
CREATE TABLE IF NOT EXISTS public.dataset_metadata (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB
);