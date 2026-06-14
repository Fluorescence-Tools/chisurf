import chinet as cn

DB_DICT = {
    'uri_string': "memory://chinet",
    'db_string': "chinet",
    'app_string': "chisurf",
    'collection_string': "test_collection"
}


def connects_to_db():
    obj = cn.BaseObject()
    obj.connect_to_db(**DB_DICT)
    return obj.is_connected_to_db


CONNECTS = connects_to_db()
