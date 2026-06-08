#ifndef CHINET_MONGOOBJECT_INMEMORY_H
#define CHINET_MONGOOBJECT_INMEMORY_H

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <iterator>
#include <string>
#include <sstream>
#include <random>

#include "json.hpp"
#include "InMemoryStorage.h"
#include "Functions.h"

using json = nlohmann::json;

// Drop-in replacement for MongoObject using in-memory storage
class MongoObject : public std::enable_shared_from_this<MongoObject> {
private:
    static std::list<std::shared_ptr<MongoObject>> registered_objects;
    static std::shared_ptr<MongoReplacement> storage_;
    
    bool is_connected_to_db_ = false;
    json document_;
    std::string object_id_;
    std::string object_name_;
    uint64_t time_of_death_;
    
    // Generate unique object ID
    std::string generateObjectId() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        
        const char* hex_chars = "0123456789abcdef";
        std::string id;
        for (int i = 0; i < 24; ++i) {
            id += hex_chars[dis(gen)];
        }
        return id;
    }
    
protected:
    // Compatibility layer for BSON-like operations
    void append_string(json* doc, const std::string& key, const std::string& value) {
        if (!doc) doc = &document_;
        (*doc)[key] = value;
    }
    
    void bson_append_bool(json* doc, const std::string& key, int key_len, bool value) {
        if (!doc) doc = &document_;
        (*doc)[key] = value;
    }
    
    void bson_append_oid(json* doc, const std::string& key, int key_len, const std::string& oid) {
        if (!doc) doc = &document_;
        (*doc)[key] = oid;
    }
    
    void bson_init(json* doc) {
        *doc = json::object();
    }
    
    void bson_copy_to(const json* src, json* dst) {
        *dst = *src;
    }
    
    std::string get_string_by_key(const json* doc, const std::string& key) {
        if (doc->contains(key) && (*doc)[key].is_string()) {
            return (*doc)[key].get<std::string>();
        }
        return "";
    }
    
    // Legacy BSON type definitions for compatibility
    using bson_t = json;
    using bson_oid_t = std::string;
    using bson_error_t = std::string;
    
    bson_t document;
    bson_oid_t oid_document;
    bson_oid_t oid_precursor;
    
    // Get object ID
    bson_oid_t get_bson_oid() {
        return object_id_;
    }
    
    virtual bson_t get_bson() {
        document = document_;
        return document;
    }
    
    bson_t get_bson_excluding(const char* first, ...) {
        json result = document_;
        
        va_list args;
        va_start(args, first);
        
        const char* key = first;
        while (key != nullptr) {
            result.erase(key);
            key = va_arg(args, const char*);
        }
        
        va_end(args);
        return result;
    }
    
    const bson_t* get_document() {
        document = document_;
        return &document;
    }
    
    void set_document(bson_t* doc) {
        document_ = *doc;
        document = document_;
    }
    
    // Storage operations
    bool write_to_db(const bson_t& doc, int write_option = 0) {
        json doc_to_write = doc;
        doc_to_write["_id"] = object_id_;
        
        if (write_option == 2) {  // Insert new
            object_id_ = storage_->insert(doc_to_write);
        } else {  // Update existing
            storage_->update(object_id_, doc_to_write);
        }
        
        is_connected_to_db_ = true;
        return true;
    }
    
    bool read_from_db() {
        auto result = storage_->findOne(object_id_);
        if (result) {
            document_ = *result;
            document = document_;
            return true;
        }
        return false;
    }
    
    template <typename T>
    void create_oid_dict_in_doc(
        bson_t* doc,
        std::string key,
        const std::map<std::string, std::shared_ptr<T>>& mongo_obj_array
    ) {
        json dict;
        for (const auto& [k, v] : mongo_obj_array) {
            dict[k] = v->get_bson_oid();
        }
        (*doc)[key] = dict;
    }
    
    template <typename T>
    void create_oid_array_in_doc(
        bson_t* doc,
        std::string key,
        const std::map<std::string, std::shared_ptr<T>>& mongo_obj_array
    ) {
        json array = json::array();
        for (const auto& [k, v] : mongo_obj_array) {
            array.push_back(v->get_bson_oid());
        }
        (*doc)[key] = array;
    }
    
    template <typename T>
    bool create_and_connect_objects_from_oid_doc(
        bson_t* doc,
        const std::string& key,
        std::map<std::string, std::shared_ptr<T>>* target_map
    ) {
        if (!doc->contains(key)) return false;
        
        const json& dict = (*doc)[key];
        for (const auto& [k, oid] : dict.items()) {
            auto obj = std::make_shared<T>();
            if (obj->read_from_db(oid.get<std::string>())) {
                (*target_map)[k] = obj;
            }
        }
        return true;
    }
    
    template <typename T>
    bool create_and_connect_objects_from_oid_array(
        bson_t* doc,
        const std::string& key,
        std::map<std::string, std::shared_ptr<T>>* target_map
    ) {
        if (!doc->contains(key)) return false;
        
        const json& array = (*doc)[key];
        int index = 0;
        for (const auto& oid : array) {
            auto obj = std::make_shared<T>();
            if (obj->read_from_db(oid.get<std::string>())) {
                (*target_map)[std::to_string(index++)] = obj;
            }
        }
        return true;
    }
    
    template <typename T>
    bool connect_object_to_db(std::shared_ptr<T> obj) {
        obj->is_connected_to_db_ = true;
        return true;
    }
    
public:
    MongoObject(const std::string& name = "") : object_name_(name) {
        object_id_ = generateObjectId();
        oid_document = object_id_;
        document_ = json::object();
        document_["_id"] = object_id_;
        document_["name"] = name;
        document = document_;
        
        if (!storage_) {
            storage_ = std::make_shared<MongoReplacement>("chinet_objects");
        }
    }
    
    virtual ~MongoObject() = default;
    
    // Public interface
    void set_name(const std::string& name) {
        object_name_ = name;
        document_["name"] = name;
    }
    
    std::string get_name() const {
        return object_name_;
    }
    
    bool is_connected_to_db() const {
        return is_connected_to_db_;
    }
    
    // Virtual methods to be overridden
    virtual bool write_to_db() {
        return write_to_db(get_bson());
    }
    
    virtual bool read_from_db(const std::string& oid_string) {
        object_id_ = oid_string;
        oid_document = object_id_;
        return read_from_db();
    }
    
    // Static storage management
    static void setStorage(std::shared_ptr<MongoReplacement> storage) {
        storage_ = storage;
    }
    
    static std::shared_ptr<MongoReplacement> getStorage() {
        return storage_;
    }
    
    // Persistence
    static bool persistAllObjects(const std::string& filename) {
        return StorageManager::getInstance()->persistAll(filename);
    }
    
    static bool loadAllObjects(const std::string& filename) {
        return StorageManager::getInstance()->loadAll(filename);
    }
    
    // Statistics
    static json getStorageStatistics() {
        return StorageManager::getInstance()->getGlobalStatistics();
    }
};

// Initialize static member
std::list<std::shared_ptr<MongoObject>> MongoObject::registered_objects;
std::shared_ptr<MongoReplacement> MongoObject::storage_ = nullptr;

#endif // CHINET_MONGOOBJECT_INMEMORY_H
