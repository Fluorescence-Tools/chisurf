#include "MongoObject.h"


std::list<std::shared_ptr<MongoObject>> MongoObject::registered_objects =
        std::list<std::shared_ptr<MongoObject>>();

MongoObject::MongoObject(std::string name) :
    uri_string(""),
    db_string(""),
    app_string(""),
    collection_string(""),
    is_connected_to_db_(false),
    object_name(""),
    time_of_death(0)
{
#if CHINET_VERBOSE
    std::clog << "NEW MONGOOBJECT" << std::endl;
#endif
    bson_oid_init(&oid_document, nullptr);
    bson_oid_copy(&oid_document, &oid_precursor);
    uri = nullptr;
    client = nullptr;
    collection = nullptr;

    bson_t *doc = BCON_NEW(
            "_id", BCON_OID(&oid_document),
            "precursor", BCON_OID(&oid_document),
            "death", BCON_INT64(time_of_death)
    );
    set_document(doc);

    if(name.empty()){
        name = get_own_oid();
    }
#if CHINET_VERBOSE
    std::clog << "-- Name: " << name << std::endl;
    std::clog << "-- OID: " << get_own_oid() << std::endl;
#endif
    set_name(name);
}

MongoObject::~MongoObject()
{
#if CHINET_VERBOSE
    std::clog << "DESTROYING MONGOOBJECT" << std::endl;
    std::clog << "-- OID: " << get_own_oid() << std::endl;
    std::clog << "-- Connected to DB: " << is_connected_to_db()
    << std::endl;
#endif
    time_of_death = Functions::get_time();
    if (is_connected_to_db()) {
#if CHINET_VERBOSE
        std::clog << "-- Time of death: " << time_of_death << std::endl;
#endif
        write_to_db();
        disconnect_from_db();
    }
#if CHINET_VERBOSE
    std::clog << "-- Total number of MongoObject instances: " <<
    registered_objects.size() << std::endl;
#endif
}

void MongoObject::register_instance(std::shared_ptr<MongoObject> x){
#if CHINET_VERBOSE
    std::clog << "REGISTER MONGOOBJECT" << std::endl;
#endif
    auto& v = registered_objects;
#if CHINET_VERBOSE
    int use_count_offset = 0;
#endif
    if(x == nullptr){
        x = get_ptr();
#if CHINET_VERBOSE
        use_count_offset++;
#endif
    }
#if CHINET_VERBOSE
    if(x != nullptr){
        std::clog << "-- OID: " << x->get_own_oid() << std::endl;
        std::clog << "-- Name: " << x->get_name() << std::endl;
        std::clog << "-- Use count before registration: " << (x.use_count() - use_count_offset) << std::endl;
    }
#endif
    if(std::find(v.begin(),v.end(),x) == v.end())
    {
        v.emplace_back(x);
    }
#if CHINET_VERBOSE
    std::clog << "-- Use count after registration: " << (x.use_count() - use_count_offset)<< std::endl;
#endif
}

void MongoObject::unregister_instance(std::shared_ptr<MongoObject> x){
#if CHINET_VERBOSE
    std::clog << "UNREGISTER MONGOOBJECT" << std::endl;
    int use_count_offset = 0;
#endif
    if(x != nullptr){
        x = get_ptr();
        registered_objects.remove(x);
    }
#if CHINET_VERBOSE
   use_count_offset++;
    if(x != nullptr){
        std::clog << "-- Use count before unregister: " << (x.use_count() - use_count_offset) << std::endl;
    }
    std::clog << "-- Use count after unregister: " << (x.use_count() - use_count_offset)<< std::endl;
#endif

}


std::list<std::shared_ptr<MongoObject>> MongoObject::get_instances(){
#if CHINET_VERBOSE
    std::clog << "MONGOOBJECT GET INSTANCES" << std::endl;
#endif
#if CHINET_VERBOSE
    for(auto &v: registered_objects){
        if(v.use_count() > 1){
            std::clog << "-- OID: " << v->get_own_oid() << std::endl;
            std::clog << "-- Use count: " << v.use_count() << std::endl;
        }
    }
#endif
    return registered_objects;
}

std::shared_ptr<MongoObject> MongoObject::get_ptr(){
    return shared_from_this();
}

bool MongoObject::connect_to_db(
        const std::string &uri_string,
        const std::string &db_string,
        const std::string &app_string,
        const std::string &collection_string
)
{
#if CHINET_VERBOSE
    std::clog << "CONNECT MONGOOBJECT TO DB" << std::endl;
#endif
    this->uri_string = uri_string;
    this->db_string = db_string;
    this->app_string = app_string;
    this->collection_string = collection_string;

    mongoc_init();

    // Database
    //----------------------------------------------------------------
#if CHINET_VERBOSE
    std::clog << "-- Connecting to DB at URI: " << uri_string.c_str() << std::endl;
#endif

    uri = mongoc_uri_new_with_error(uri_string.c_str(), &error);
    if (!uri) {
#if CHINET_VERBOSE
        std::cerr << "-- Failed to parse URI:" << uri_string.c_str() << std::endl;
        std::cerr << "-- Error message: " << error.message << std::endl;
#endif
        is_connected_to_db_ = false;
        return false;
    } else {
        /*
        * Create a new client instance
        */
        client = mongoc_client_new_from_uri(uri);
        if (!client) {
            is_connected_to_db_ = false;
            return EXIT_FAILURE;
        }
        /*
        * Register the application name so we can track it in the profile logs
        * on the server. This can also be done from the URI (see other examples).
        */
        mongoc_client_set_appname(client, app_string.c_str());
        /*
        * Get a handle on the collection
        */
        collection = mongoc_client_get_collection(
                client,
                db_string.c_str(),
                collection_string.c_str()
                );
        is_connected_to_db_ = true;
        return true;
    }
}

void MongoObject::disconnect_from_db()
{
#if CHINET_VERBOSE
    std::clog << "DISCONNECT MONGOOBJECT FROM DB" << std::endl;
    std::clog << "-- is_connected_to_db: " << is_connected_to_db() << std::endl;
    std::clog << "-- collection: " << collection << std::endl;
    std::clog << "-- uri: " << uri << std::endl;
#endif
    if (is_connected_to_db()) {
        // destroy cursor, collection, session before the client they came from
        if (collection) {
            mongoc_collection_destroy(collection);
        }
        if (uri) {
            mongoc_uri_destroy(uri);
        }
        if (client) {
            mongoc_client_destroy(client);
        }
        mongoc_cleanup();
    }
    is_connected_to_db_ = false;
}

bool MongoObject::write_to_db(
        const bson_t &doc,
        int write_option
    )
{
#if CHINET_VERBOSE
    std::clog << "WRITING MONGOOBJECT TO DB" << std::endl;
#endif
    bool return_value = false;
#if CHINET_VERBOSE
    std::clog << "-- is_connected_to_db: " << is_connected_to_db() << std::endl;
    std::clog << "-- Write option: " << write_option << std::endl;
#endif
    if (is_connected_to_db()) {
        return_value = true;

        bson_t *query = nullptr;
        bson_t *update = nullptr;
        bson_t reply;

        query = BCON_NEW ("_id", BCON_OID(&oid_document));

        bson_t *opts = BCON_NEW(
                "upsert", BCON_BOOL(true)
        );

        switch (write_option) {
            case 1:
#if CHINET_VERBOSE
                std::clog << "-- Replacing object in the DB." << std::endl;
#endif
                // option 1 - write as a replacement
                if (
                    !mongoc_collection_replace_one(
                            collection,
                            query, &doc,
                            nullptr, &reply, &error
                    )
                ) {
#if CHINET_VERBOSE
                    std::cerr << error.message;
#endif
                    return_value &= false;
                }
                break;
            case 2:
#if CHINET_VERBOSE
                std::clog << "-- Inserting as a new object in DB." << std::endl;
#endif
                // option 2 - insert as a new document
                if (
                    !mongoc_collection_insert_one(
                        collection, &doc,
                        nullptr,
                        &reply, &error
                    )
                ) {
#if CHINET_VERBOSE
                    std::cerr << error.message;
#endif
                    return_value &= false;
                }
                break;
            default:
#if CHINET_VERBOSE
                std::clog << "-- Updating existing object in DB." << std::endl;
#endif
                // option 0 - write as a update
                update = BCON_NEW ("$set", BCON_DOCUMENT(&doc));
                if (!mongoc_collection_find_and_modify(
                        collection,
                        query,
                        nullptr,
                        update,
                        nullptr,
                        false,
                        true, // update by upsert: if the oid is not in the DB create a new document
                        false,
                        &reply, &error)
                        ) {
#if CHINET_VERBOSE
                    std::cerr << error.message;
#endif
                    return_value &= false;
                }
                break;
        }
        // destroy
        bson_destroy(update);
        bson_destroy(query);
        bson_destroy(&reply);
    } else {
        std::cerr << "ERROR: Not connected to DB - cannot write!" << std::endl;
    }
    return return_value;
}

bool MongoObject::write_to_db()
{
#if CHINET_VERBOSE
    std::clog << "WRITING MONGOOBJECT TO DB" << std::endl;
    std::clog << "-- MongoObject OID: " << get_own_oid() << std::endl;
#endif
    return write_to_db(get_bson(), 0);
}

bool MongoObject::read_from_db(const std::string &oid_string)
{
#if CHINET_VERBOSE
    std::clog << "READ MONGOOBJECT FROM DB" << std::endl;
    auto str = std::string(oid_string.c_str(), oid_string.size());
    std::clog << "-- Requested MongoObject OID:"<< str << std::endl;
#endif
    bson_oid_t oid;
    if (string_to_oid(oid_string, &oid)) {
        if (!is_connected_to_db()) {
#if CHINET_VERBOSE
            std::cerr << "-- Not connected to a DB." << std::endl;
#endif
            return false;
        } else {
            // find the oid in the DB collection
            bson_t *query = nullptr;
            query = BCON_NEW ("_id", BCON_OID(&oid));
            size_t len;
#if CHINET_VERBOSE
            std::clog << "-- Query result: " << bson_as_json(query, &len) << std::endl;
#endif
            mongoc_cursor_t *cursor; // cursor pointing to the new document
            cursor = mongoc_collection_find_with_opts(
                    collection,
                    query,
                    nullptr, // the opts
                    nullptr  // the read_prefs
            );
            const bson_t *doc;
            while (mongoc_cursor_next(cursor, &doc)) {
#if CHINET_VERBOSE
                std::clog << "-- Read content from DB: " << bson_as_json(doc, &len) << std::endl;
#endif
#if CHINET_VERBOSE
                std::clog << "-- Document content before reinint:" << bson_as_json(&document, &len) << std::endl;
#endif
#if CHINET_VERBOSE
                std::clog << "-- Reinit local document" << std::endl;
#endif
                bson_reinit(&document);
#if CHINET_VERBOSE
                std::clog << "-- Document content after reinint:" << bson_as_json(&document, &len) << std::endl;
#endif
#if CHINET_VERBOSE
                std::clog << "-- Copying document of query to the document of the node" << std::endl;
#endif
                bson_copy_to(doc, &document);
#if CHINET_VERBOSE
                std::clog << "-- Document content after copy: " << bson_as_json(&document, &len) << std::endl;
#endif
                bson_oid_copy(&oid, &oid_document);
#if CHINET_VERBOSE
                std::clog << "-- Setting the name and the precursor OID" << std::endl;
#endif
                bson_iter_t iter;
                if (bson_iter_init(&iter, &document) &&
                    bson_iter_find(&iter, "precursor") &&
                    BSON_ITER_HOLDS_OID(&iter)) {
                    bson_oid_copy(bson_iter_oid(&iter), &oid_precursor);
#if CHINET_VERBOSE
                    char oid_str[25];
                    bson_oid_to_string(&oid_precursor, oid_str);
                    std::clog << "-- Object's precursor OID set to the OID DB: "<< oid_str << std::endl;
#endif
                } else {
#if CHINET_VERBOSE
                    char oid_str[25];
                    bson_oid_to_string(&oid_document, oid_str);
                    std::clog << "-- Object's precursor OID set to own OID: "<< oid_str << std::endl;
#endif
                    bson_oid_copy(&oid_document, &oid_precursor);
                }
#if CHINET_VERBOSE
                std::clog << "-- Updating the time of death" << std::endl;
#endif
                if (bson_iter_init(&iter, &document) &&
                    bson_iter_find(&iter, "death") &&
                    BSON_ITER_HOLDS_INT64(&iter)) {
                    time_of_death = bson_iter_int64(&iter);
#if CHINET_VERBOSE
                    std::clog << "-- Set time of death: " << time_of_death << std::endl;
#endif
                } else {
                    time_of_death = 0;
                }
                if (bson_iter_init(&iter, &document) &&
                    bson_iter_find(&iter, "name") &&
                    BSON_ITER_HOLDS_UTF8(&iter)) {
                    uint32_t length; const char* text;
                    text = bson_iter_utf8(&iter, &length);
                    std::string name = std::string(text, length);
                    set_name(name);
#if CHINET_VERBOSE
                    std::clog << "-- Set name to:" << name << std::endl;
#endif
                }
            }
            if (mongoc_cursor_error(cursor, &error)) {
#if CHINET_VERBOSE
                std::cerr << "-- An error occurred. " << error.message << std::endl;
#endif
                return false;
            }
            // Clean up
            bson_destroy(query);
            mongoc_cursor_destroy(cursor);
            return true;
        }
    } else {
#if CHINET_VERBOSE
        std::cerr << "-- Error: OID string not valid." << std::endl;
#endif
        return false;
    }
}


bool MongoObject::read_from_db()
{
    return read_from_db(oid_to_string(oid_document));
}

std::string MongoObject::get_json_of_key(std::string key){
    std::string re;
    bson_iter_t iter, desc;
    bson_iter_init (&iter, &document);
    if(bson_iter_find_descendant (&iter, key.c_str(), &desc)){
        if (BSON_ITER_HOLDS_DOCUMENT (&desc)) {
            char *str = NULL;
            bson_t *arr;
            const uint8_t *data = NULL;
            uint32_t len = 0;
            bson_iter_document (&desc, &len, &data);
            arr = bson_new_from_data (data, len);
            str = bson_as_json (arr, NULL);
            re.assign(str);
            bson_free (str);
            bson_destroy (arr);
        }
    }
    return re;
}

std::string MongoObject::get_json(int indent)
{
    size_t len;
    bson_t doc = get_bson();
    char *str = bson_as_json(&doc, &len);
    if(indent == 0){
        return std::string(str, len);
    } else{
        // use nlohmann json to make pretty
        auto j = json::parse(str);
        return j.dump(indent);
    }
}


bson_t MongoObject::get_bson()
{

    bson_iter_t iter;
    bson_t doc;
    bson_init(&doc);
    bson_copy_to(&document, &doc);

    // oid_document
    if (bson_iter_init(&iter, &doc) &&
        bson_iter_find(&iter, "_id") &&
        BSON_ITER_HOLDS_OID(&iter)) {
        bson_iter_overwrite_oid(&iter, &oid_document);
    } else {
        bson_append_oid(&doc, "_id", 3, &oid_document);
    }

    // oid_precursor
    if (bson_iter_init(&iter, &doc) &&
        bson_iter_find(&iter, "precursor") &&
        BSON_ITER_HOLDS_OID(&iter)) {
        bson_iter_overwrite_oid(&iter, &oid_precursor);
    } else {
        bson_append_oid(&doc, "precursor", 9, &oid_precursor);
    }

    // time of death
    if (bson_iter_init(&iter, &doc) &&
        bson_iter_find(&iter, "death") &&
        BSON_ITER_HOLDS_INT64(&iter)) {
        bson_iter_overwrite_int64(&iter, time_of_death);
    } else {
        bson_append_int64(&doc, "death", 5, time_of_death);
    }

    // name
    append_string(
            &doc, "name",
            object_name.c_str(), object_name.size()
            );

    return doc;
}

bson_t MongoObject::get_bson_excluding(const char *first, ...)
{
    bson_t src = MongoObject::get_bson();
    bson_t dst;
    bson_init(&dst);
    va_list va;
    va_start(va, first);
    bson_copy_to_excluding_noinit_va(&src, &dst, "", va);
    va_end(va);
    return dst;
}

const bson_t *MongoObject::get_document()
{
    return &document;
}

std::string MongoObject::create_copy_in_db()
{
    bson_t document_copy;
    // update oid of copy
    bson_oid_t oid_copy;
    bson_oid_init(&oid_copy, nullptr);
    bson_copy_to(get_document(), &document_copy);
    bson_iter_t iter;
    if (bson_iter_init(&iter, &document_copy) &&
        bson_iter_find(&iter, "_id") &&
        BSON_ITER_HOLDS_OID(&iter)) {
        bson_iter_overwrite_oid(&iter, &oid_copy);
    }
    // set precursor of copy to current document
    if (bson_iter_init(&iter, &document_copy) &&
        bson_iter_find(&iter, "precursor") &&
        BSON_ITER_HOLDS_OID(&iter)) {
        bson_iter_overwrite_oid(&iter, &oid_document);
    } else {
        bson_append_oid(&document_copy, "precursor", 9, &oid_document);
    }
    size_t len;
#if CHINET_VERBOSE
    std::clog << "created copy: " << bson_as_json(&document_copy, &len) << std::endl;
#endif
    write_to_db(document_copy, 2);
    return oid_to_string(oid_copy);
}

bool MongoObject::is_connected_to_db()
{
    if(is_connected_to_db_){
        // Double check connection by pinging the DB
        bson_t *b = BCON_NEW ("ping", BCON_INT32 (1));
        bson_error_t error;
        bool r;
        mongoc_server_description_t **sds;
        size_t i, n;

        /* ensure client has connected */
        r = mongoc_client_command_simple (client, "db", b, NULL, NULL, &error);
        if (!r) {
            MONGOC_ERROR ("could not connect: %s\n", error.message);
            return false;
        }
    }
    return is_connected_to_db_;
}

void MongoObject::set_oid(const char *key, bson_oid_t value)
{
    bson_iter_t iter;
    if (bson_iter_init_find(&iter, &document, key) &&
        BSON_ITER_HOLDS_OID(&iter)) {
        bson_iter_overwrite_oid(&iter, &value);
    }
}

bool MongoObject::string_to_oid(const std::string &oid_string, bson_oid_t *oid)
{
    if (bson_oid_is_valid(oid_string.c_str(), oid_string.size())) {
        // convert the oid string to an oid
        bson_oid_init_from_string(oid, oid_string.c_str());
        return true;
    } else {
        bson_oid_init(oid, nullptr);
#if CHINET_VERBOSE
        std::cerr << "OID string not valid." << std::endl;
#endif
        return false;
    }
}

void MongoObject::append_string(
        bson_t *dst,
        const std::string key,
        const std::string content,
        size_t size
        )
{
    if (size != 0) {
        bson_append_utf8(
                dst,
                key.c_str(), key.size(),
                content.c_str(), size
                );
    } else {
        bson_append_utf8(
                dst,
                key.c_str(), key.size(),
                content.c_str(), content.size()
                );
    }
}

void MongoObject::set_string(
        std::string key,
        std::string str
){
    bson_t dst; bson_init(&dst);
    bson_copy_to_excluding_noinit(
            &document, &dst,
            key.c_str(),
            NULL
    );
    bson_append_utf8(
            &dst,
            key.c_str(), key.size(),
            str.c_str(), str.size()
    );
    bson_reinit(&document);
    bson_copy_to(&dst, &document);
    bson_destroy(&dst);
}


const std::string MongoObject::get_string_by_key(bson_t *doc, const std::string key)
{
    bson_iter_t iter;

    if (bson_iter_init(&iter, doc) &&
        bson_iter_find(&iter, key.c_str()) &&
        BSON_ITER_HOLDS_UTF8(&iter)) {
        const char *str;
        uint32_t len;
        str = bson_iter_utf8(&iter, &len);
        return std::string(str, len);
    }
#if CHINET_VERBOSE
    std::cerr << "Error: the key does not contain an string" << std::endl;
#endif

    return "";
}

bool MongoObject::read_json(std::string json_string)
{
    bson_t b;
    bson_error_t error;
    if (!bson_init_from_json(&b, json_string.c_str(), json_string.size(), &error)) {
#if CHINET_VERBOSE
        std::cerr << "Error reading JSON: " << error.message << std::endl;
#endif
        return false;
    } else {
        bson_reinit(&document);
        bson_copy_to(&b, &document);
        return true;
    }
}

std::shared_ptr<MongoObject> MongoObject::operator[](std::string key)
{
    auto mo = std::make_shared<MongoObject>();
    mo->read_json(get_json_of_key(key.c_str()));
    return mo;
}

