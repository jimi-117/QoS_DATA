// qos_data/setup_data/mongo_init/init-mongo.js
// Use the database defined by MONGO_INITDB_DATABASE env var
const dbName = process.env.MONGO_INITDB_DATABASE;
db = db.getSiblingDB(dbName);

// Create the images collection for storing metadata loaded from source YOLO dataset
db.createCollection('images');

// Optional: Create indexes for faster querying if needed later by main_etl.py
// Example: Index on filename (which matches PG primary key)
db.images.createIndex({ filename: 1 });
db.images.createIndex({ split: 1 }); // Index on split if useful