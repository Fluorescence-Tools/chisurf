import chinet as cn
import pymongo as pm

uri_string = "mongodb://localhost:27017"
db_string = "chinet"
app_string = "chisurf"
collection_string = "test_collection"

mo = cn.MongoObject()
mo.connect(
    uri_string=uri_string,
    db_string=db_string,
    app_string=app_string,
    collection_string=collection_string
)

v = cn.Value([717, 11], False)
mo.document = v.to_bson()
mo.to_json()

mo.write()

# for testing
mongo_client = pm.MongoClient(uri_string)
db = mongo_client.chinet
collection = db.test_collection
collection.find_one({'_id': mo.get_oid()})
