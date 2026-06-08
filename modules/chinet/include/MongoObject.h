#ifndef CHINET_MONGOOBJECT_H
#define CHINET_MONGOOBJECT_H

#include <iostream>     // std::cout, std::ios
#include <map>
#include <set>
#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <iterator>
#include <string>
#include <sstream>      // std::ostringstream
#include <mongoc.h>

#include "json.hpp"
using json = nlohmann::json;


#include "Functions.h"

class MongoObject : public std::enable_shared_from_this<MongoObject>{

private:

    static std::list<std::shared_ptr<MongoObject>> registered_objects;

    bool is_connected_to_db_ = false;
    mongoc_uri_t *uri;
    mongoc_client_t *client;
    bson_error_t error;
    mongoc_collection_t *collection;

protected:

    std::string object_name;
    bson_t document;
    std::string uri_string;
    std::string db_string;
    std::string app_string;
    std::string collection_string;
    bson_oid_t oid_document;
    bson_oid_t oid_precursor;
    uint64_t time_of_death;

    // Getter & Setter
    //--------------------------------------------------------------------

    //! The object identification number of the @class MongoObject instance
    /*!
     *  get_own_bson_oid() Returns the object's ObjectId value (see:
     *  https://docs.mongodb.com/manual/reference/bson-types/#objectid)
     *
     */
    //! \return
    bson_oid_t get_bson_oid()
    {
        return oid_document;
    }

    /// The BSON document of the @class MongoObject instance
    /// \return
    virtual bson_t get_bson();

    /// A BSON document containing the @class MongoObject instance BSON document
    /// excluding a set of keys
    /// \param first is the first key to exclude.
    /// \param ... more keys to exclude
    /// \return a bson_t document
    bson_t get_bson_excluding(const char* first, ...);

    /// Pointer to the BSON document of the MongoObject
    /// \return
    const bson_t* get_document();


    void set_document(bson_t *doc){
        bson_init(&document);
        bson_copy_to(doc, &document);
    }

    // Methods
    //--------------------------------------------------------------------
    //! Writes a BSON document to the connected MongoDB
    /*!
     *
     * @param doc a pointer to the BSON document that is written to the MongoDB
     * @param write_option integer specifying the write mode - 1: replaces the document (no upsert), 2: insert
     * the document das a new document, default: updates an existing document.
     *
     * @return true in case of a successful write.
     */
    bool write_to_db(const bson_t &doc, int write_option = 0);
    bool read_from_db();

    template <typename T>
    void create_oid_dict_in_doc(
            bson_t *doc,
            std::string key,
            const std::map<std::string, std::shared_ptr<T>> &mongo_obj_array
            ){

        bson_t child;
        bson_append_document_begin(doc, key.c_str(), key.size(), &child);
        for(auto &v : mongo_obj_array){
            const bson_oid_t b = v.second->get_bson_oid();
            bson_append_oid(&child, v.first.c_str(), v.first.size(), &b);
        }
        bson_append_document_end(doc, &child);
    }

    template <typename T>
    void append_number_array(bson_t *doc, std::string key, T &values){
        bson_t child;
        bson_append_array_begin(doc, key.c_str(), key.size(), &child);
        if(
                (std::is_same<T, std::vector<long>>::value) ||
                (std::is_same<T, std::vector<int>>::value)
                ){
            for(auto &v : values){
                bson_append_int64(&child, "", 0, v);
            }
        }
        else if(std::is_same<T, std::vector<double>>::value){
            for(auto &v : values){
                bson_append_double(&child, "", 0, v);
            }
        }
        else if(std::is_same<T, std::vector<bool>>::value){
            for(auto &v : values){
                bson_append_bool(&child, "", 0, v);
            }
        }
        bson_append_array_end(doc, &child);
    }

    template <typename T>
    void create_oid_array_in_doc(
            bson_t *doc,
            std::string target_field_name,
            const std::map<std::string, std::shared_ptr<T>> &mongo_obj_array){

        bson_t child;
        bson_append_array_begin(
                doc,
                target_field_name.c_str(), target_field_name.size(),
                &child
                );
        for(auto &v : mongo_obj_array){
            const bson_oid_t b = v.second->get_bson_oid();
            bson_append_oid(&child, "", 0, &b);
        }
        bson_append_array_end(doc, &child);
    }

    template <typename T>
    bool create_and_connect_objects_from_oid_doc(
            const bson_t *doc,
            const char *document_name,
            std::map<std::string, std::shared_ptr<T>> *target_map){
        bool return_value = true;
        bson_iter_t iter; bson_iter_t child;
        if (bson_iter_init_find (&iter, doc, document_name) &&
            BSON_ITER_HOLDS_DOCUMENT (&iter) &&
            bson_iter_recurse (&iter, &child)) {
            while (bson_iter_next (&child)) {
                if (BSON_ITER_HOLDS_OID(&child)){
                    // read oid
                    bson_oid_t oid;
                    bson_oid_copy(bson_iter_oid(&child), &oid);
                    // create new obj
                    auto o = new T();
                    // connect obj to db
                    return_value &= connect_object_to_db(o);
                    // read obj from db
#if CHINET_VERBOSE
                    std::cout << oid_to_string(oid) << std::endl;
#endif
                    o->read_from_db(oid_to_string(oid));
                    std::string key = bson_iter_key(&child);
                    // add obj to the target map
                    target_map->insert(std::pair<std::string, T*>(key, o));
                }
            }
        } else{
#if CHINET_VERBOSE
            std::cerr << "Error: no nodes section in Session" << std::endl;
#endif
            return_value &= false;
        }
        return return_value;
    }

    template <typename T>
    bool create_and_connect_objects_from_oid_array(
            const bson_t *doc,
            const char *array_name,
            std::map<std::string, std::shared_ptr<T>> *target_map)
    {
        bool return_value = true;
        bson_iter_t iter;
        bson_iter_t child;
        if (bson_iter_init_find (&iter, doc, array_name) &&
            BSON_ITER_HOLDS_ARRAY (&iter) &&
            bson_iter_recurse (&iter, &child)) {
            while (bson_iter_next (&child)) {
                if (BSON_ITER_HOLDS_OID(&child)){
                    // read oid
                    bson_oid_t new_oid;
                    bson_oid_copy(bson_iter_oid(&child), &new_oid);
                    // create new obj
                    std::shared_ptr<T> o = std::make_shared<T>();
                    // connect obj to db
                    return_value &= connect_object_to_db(o);
                    // read obj from db
                    o->read_from_db(oid_to_string(new_oid));
                    // add obj to the target set
                    target_map->insert(std::make_pair(o->get_own_oid(), o));
                }
            }
        } else{
#if CHINET_VERBOSE
            std::cerr << "Error: no nodes section in Session" << std::endl;
#endif
            return_value &= false;
        }
        return return_value;
    }

    /// Append a string to a BSON document
    /// \param dst pointer to the target BSON document
    /// \param key the key of the string int the target BSON document
    /// \param content the string that will be written to the BSON document
    /// \param size optional parameter for the string size in the BSON document.
    static void append_string(
            bson_t *dst, std::string key,
            std::string content,
            size_t size=0
                    );


    /// The string contained in a @class bson_t document with the @param key
    /// \param doc BSON @class bson_t document which is inspected
    /// \param key
    /// \return string contained under the @param key
    static const std::string get_string_by_key(
            bson_t *doc, std::string key
            );


    /// Converts a @class bson_oid_t oid into a @class std::string
    /// \param oid
    /// \return
    static std::string oid_to_string(bson_oid_t oid){
        char oid_str[25];
        bson_oid_to_string(&oid, oid_str);
        return std::string(oid_str, 25).substr(0, 24);
    }


    /// Converts a @class std::string into a @class bson_oid_t
    /// \param oid_string
    /// \param oid
    /// \return
    static bool string_to_oid(
            const std::string &oid_string,
            bson_oid_t *oid
            );


public:

    ~MongoObject();

    MongoObject(std::string name="");

    //! Connects the instance of @class MongoObject to a database
    /*!
     *
     * @param uri_string
     * @param db_string
     * @param app_string
     * @param collection_string
     * @return True if connected successfully
     */
    bool connect_to_db(
            const std::string &uri_string,
            const std::string &db_string,
            const std::string &app_string,
            const std::string &collection_string
    );

    /// Connects other object to the same MongoDB
    /// \tparam T
    /// \param o The object that is connected to the same MongoDB
    /// \return True if connected successfully
    template <typename T>
    bool connect_object_to_db(T o){
        return o->connect_to_db(
                uri_string,
                db_string,
                app_string,
                collection_string
        );
    }

    /// Disconnects the @class MongoObject instance from the DB
    void disconnect_from_db();

    /// Returns true if the instance of the @class MongoObject is connected
    /// to the DB
    bool is_connected_to_db();

    void register_instance(std::shared_ptr<MongoObject>);

    void unregister_instance(std::shared_ptr<MongoObject>);

    ///
    static std::list<std::shared_ptr<MongoObject>> get_instances();

    /// Writes @class MongoObject to the connected MongoDB
    /// \return
    virtual bool write_to_db();

    /// Creates a copy of the object in the connected MongoDB with a new OID.
    /// \return The OID of the copy
    std::string create_copy_in_db();

    /// Read the content of an existing BSON document into the current object
    /// \param oid_string Object identifier of the queried document in the DB
    /// \return True if successful otherwise false
    virtual bool read_from_db(const std::string &oid_string);

    /// Read the content of a JSON string into a MongoObject
    /// \param json_string
    /// \return
    bool read_json(std::string json_string);

    /// The own object identifier
    /// \return
    std::string get_own_oid()
    {
        return oid_to_string(oid_document);
    }

    /// Set the own object identifier without duplicate check
    /// \param oid_str
    void set_own_oid(std::string oid_str)
    {
        MongoObject::string_to_oid(oid_str, &oid_document);
    }

    void set_name(std::string name){
        object_name = name;
    }

    virtual std::string get_name(){
        return object_name;
    }

    std::shared_ptr<MongoObject> get_ptr();

    /// Create and / or set a string in the MongoObject accessed by @param key
    /// \param key the key to access the content
    /// \param str The content
    void set_string(std::string key, std::string str);

    // Seems WRONG
    virtual std::string get_string(){
        return object_name;
    }

    template<typename T>
    T get_singleton(const char *key){
        T v(0);
        bson_iter_t iter;
        if(std::is_same<T, int>::value || std::is_same<T, long>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_INT64(&iter))) {
                return (T) bson_iter_int64(&iter);
            }
        }
        else if(std::is_same<T, double>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_DOUBLE(&iter))) {
                return bson_iter_double(&iter);
            }
        }
        else if(std::is_same<T, bool>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_BOOL(&iter))) {
                return bson_iter_bool(&iter);
            }
        }
        return v;
    }

    template <typename T>
    void set_singleton(const char *key, T value){
        bson_iter_t iter;
        if(std::is_same<T, bool>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                BSON_ITER_HOLDS_BOOL(&iter)) {
                bson_iter_overwrite_bool(&iter, value);
            } else{
                bson_append_bool(&document, key, -1, value);
            }
        }
        else if(std::is_same<T, double>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                BSON_ITER_HOLDS_DOUBLE(&iter)) {
                bson_iter_overwrite_double(&iter, value);
            } else{
                bson_append_double(&document, key, -1, value);
            }
        }
        else if(std::is_same<T, int>::value) {
            if (bson_iter_init_find(&iter, &document, key) &&
                BSON_ITER_HOLDS_INT64(&iter)) {
                bson_iter_overwrite_int64(&iter, value);
            } else{
                bson_append_int64(&document, key, -1, value);
            }
        }
    }

    void set_oid(const char* key, bson_oid_t value);

    template <typename T>
    std::vector<T> get_array(const char* key){
        bson_iter_t iter;
        std::vector<T> v{};
        if(std::is_same<T, double>::value){
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_ARRAY(&iter))){
                bson_iter_t iter_array;
                bson_iter_recurse (&iter, &iter_array);
                while (bson_iter_next (&iter_array) &&
                       BSON_ITER_HOLDS_DOUBLE(&iter_array)) {
                    v.push_back(bson_iter_double(&iter_array));
                }
            }
            return v;
        }
        else if(std::is_same<T, int>::value || std::is_same<T, long>::value){
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_ARRAY(&iter))) {
                bson_iter_t iter_array;
                bson_iter_recurse(&iter, &iter_array);
                while (bson_iter_next(&iter_array) &&
                       BSON_ITER_HOLDS_INT64(&iter_array)) {
                    v.push_back((T) bson_iter_int64(&iter_array));
                }
            }
            return v;
        }
        else if(std::is_same<T, bool>::value){
            if (bson_iter_init_find(&iter, &document, key) &&
                (BSON_ITER_HOLDS_ARRAY(&iter))) {
                bson_iter_t iter_array;
                bson_iter_recurse(&iter, &iter_array);
                while (bson_iter_next(&iter_array) &&
                       BSON_ITER_HOLDS_BOOL(&iter_array)) {
                    v.push_back((int) bson_iter_bool(&iter_array));
                }
            }
            return v;
        }
        return v;
    }

    template <typename T>
    void set_array(const char* key, std::vector<T> value){
        bson_t dst; bson_init(&dst);
        bson_copy_to_excluding_noinit(
                &document, &dst,
                key,
                NULL
        );
        append_number_array(&dst, key, value);
        bson_copy_to(&dst, &document);
    }

    std::string get_json(int indent=4);

    std::string get_json_of_key(std::string key);

    std::string show(){
        std::ostringstream os;
        os << this->get_json();
        os << std::endl;
        return os.str();
    }

    // Operators
    //--------------------------------------------------------------------
    virtual std::shared_ptr<MongoObject> operator[](std::string key);

    bool operator==(MongoObject const& b){
        return (
                bson_oid_equal(&b.oid_document, &oid_document) &&
                (uri_string == b.uri_string)
        );
    };

    friend std::ostream& operator<<(std::ostream &out, MongoObject& o)
    {
        out << o.get_json();
        return out;
    }

};

#endif //CHINET_MONGOOBJECT_H
