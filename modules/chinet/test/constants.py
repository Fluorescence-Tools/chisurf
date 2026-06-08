import chinet as cn


DB_DICT = {
    'uri_string': "mongodb://localhost:27017",
    'db_string': "chinet",
    'app_string': "chisurf",
    'collection_string': "test_collection"
}

def connects_to_db():
    mo = cn.MongoObject()
    mo.connect_to_db(**DB_DICT)
    return mo.is_connected_to_db 

CONNECTS = connects_to_db()
