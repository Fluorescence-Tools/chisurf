import chinet as cn
import os

# Check if MongoDB support is enabled
WITH_MONGODB = hasattr(cn, 'MongoObject')

DB_DICT = {
    'uri_string': "mongodb://localhost:27017",
    'db_string': "chinet",
    'app_string': "chisurf",
    'collection_string': "test_collection"
}

def connects_to_db():
    if WITH_MONGODB:
        # Try to connect to MongoDB
        mo = cn.MongoObject()
        mo.connect_to_db(**DB_DICT)
        return mo.is_connected_to_db
    else:
        # Using in-memory implementation
        mo = cn.DatabaseObject()
        mo.connect_to_db("memory", "memory", "memory", "memory")
        return mo.is_connected_to_db

# Check if we can connect to the database (MongoDB or in-memory)
CONNECTS = connects_to_db()

# For tests that specifically need MongoDB
MONGODB_AVAILABLE = WITH_MONGODB and CONNECTS
