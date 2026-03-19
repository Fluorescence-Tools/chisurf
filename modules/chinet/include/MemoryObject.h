#ifndef CHINET_MEMORYOBJECT_H
#define CHINET_MEMORYOBJECT_H

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <ctime>
#include <iterator>
#include <string>
#include <sstream>
#include <unordered_map>
#include <mutex>

#include "info.h"
#include "json.hpp"

using json = nlohmann::json;

#include "Functions.h"

// Forward declaration
class MongoObject;

/**
 * @class MemoryObject
 * @brief In-memory implementation of MongoObject functionality
 * 
 * This class provides the same interface as MongoObject but stores data in memory
 * instead of using MongoDB. It's designed to be a drop-in replacement for MongoObject
 * when MongoDB is not available or not desired.
 */
class MemoryObject : public std::enable_shared_from_this<MemoryObject> {
private:
    static std::list<std::shared_ptr<MemoryObject>> registered_objects;
    static std::unordered_map<std::string, json> object_store;
    static std::mutex object_store_mutex;

    bool is_connected_to_db_ = false;

protected:
    std::string object_name;
    json document;
    std::string oid_document;
    std::string oid_precursor;
    uint64_t time_of_death;

    // Set the document from a JSON object
    void set_document(json doc);

    // Helper methods for JSON manipulation
    template <typename T>
    void create_oid_dict_in_doc(
            json& doc,
            std::string key,
            const std::map<std::string, std::shared_ptr<T>>& obj_array) {
        json child = json::object();
        for (auto& v : obj_array) {
            child[v.first] = v.second->get_own_oid();
        }
        doc[key] = child;
    }

    template <typename T>
    void append_number_array(json& doc, std::string key, T& values) {
        json array = json::array();
        for (auto& v : values) {
            array.push_back(v);
        }
        doc[key] = array;
    }

    template <typename T>
    void create_oid_array_in_doc(
            json& doc,
            std::string target_field_name,
            const std::map<std::string, std::shared_ptr<T>>& obj_array) {
        json array = json::array();
        for (auto& v : obj_array) {
            array.push_back(v.second->get_own_oid());
        }
        doc[target_field_name] = array;
    }

    template <typename T>
    bool create_and_connect_objects_from_oid_doc(
            const json& doc,
            const char* document_name,
            std::map<std::string, std::shared_ptr<T>>* target_map) {
        bool return_value = true;
        if (doc.contains(document_name) && doc[document_name].is_object()) {
            for (auto& [key, value] : doc[document_name].items()) {
                if (value.is_string()) {
                    std::string oid = value.template get<std::string>();
                    auto o = std::make_shared<T>();
                    return_value &= connect_object_to_db(o);
                    o->read_from_db(oid);
                    target_map->insert(std::make_pair(key, o));
                }
            }
        } else {
            return_value = false;
        }
        return return_value;
    }

    template <typename T>
    bool create_and_connect_objects_from_oid_array(
            const json& doc,
            const char* array_name,
            std::map<std::string, std::shared_ptr<T>>* target_map) {
        bool return_value = true;
        if (doc.contains(array_name) && doc[array_name].is_array()) {
            for (auto& value : doc[array_name]) {
                if (value.is_string()) {
                    std::string oid = value.template get<std::string>();
                    auto o = std::make_shared<T>();
                    return_value &= connect_object_to_db(o);
                    o->read_from_db(oid);
                    target_map->insert(std::make_pair(o->get_own_oid(), o));
                }
            }
        } else {
            return_value = false;
        }
        return return_value;
    }

    // Generate a unique OID
    static std::string generate_oid() {
        static int counter = 0;
        std::stringstream ss;
        ss << std::hex << std::time(nullptr) << "-" << ++counter;
        return ss.str();
    }

public:
    ~MemoryObject();

    MemoryObject(std::string name="");

    bool connect_to_db(
            const std::string& uri_string,
            const std::string& db_string,
            const std::string& app_string,
            const std::string& collection_string);

    template <typename T>
    bool connect_object_to_db(T o) {
        return o->connect_to_db("memory", "memory", "memory", "memory");
    }

    void disconnect_from_db();

    bool is_connected_to_db();

    void register_instance(std::shared_ptr<MemoryObject>);

    void unregister_instance(std::shared_ptr<MemoryObject>);

    static std::list<std::shared_ptr<MemoryObject>> get_instances();

    virtual bool write_to_db();

    std::string create_copy_in_db();

    virtual bool read_from_db(const std::string& oid_string);

    bool read_json(std::string json_string);

    std::string get_own_oid() {
        return oid_document;
    }

    void set_own_oid(std::string oid_str) {
        oid_document = oid_str;
    }

    void set_name(std::string name) {
        object_name = name;
    }

    virtual std::string get_name() {
        return object_name;
    }

    std::shared_ptr<MemoryObject> get_ptr();

    void set_string(std::string key, std::string str) {
        document[key] = str;
    }

    virtual std::string get_string() {
        return object_name;
    }

    template<typename T>
    T get_singleton(const char* key) {
        T v{};
        if (document.contains(key)) {
            if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value) {
                if (document[key].is_number_integer()) {
                    return document[key].get<T>();
                }
            }
            else if constexpr (std::is_same<T, double>::value) {
                if (document[key].is_number_float()) {
                    return document[key].get<T>();
                }
            }
            else if constexpr (std::is_same<T, bool>::value) {
                if (document[key].is_boolean()) {
                    return document[key].get<T>();
                }
            }
        }
        return v;
    }

    template <typename T>
    void set_singleton(const char* key, T value) {
        document[key] = value;
    }

    void set_oid(const char* key, std::string value) {
        document[key] = value;
    }

    template <typename T>
    std::vector<T> get_array(const char* key) {
        std::vector<T> v{};
        if (document.contains(key) && document[key].is_array()) {
            for (auto& item : document[key]) {
                if constexpr (std::is_same<T, double>::value) {
                    if (item.is_number_float()) {
                        v.push_back(item.get<T>());
                    }
                }
                else if constexpr (std::is_same<T, int>::value || std::is_same<T, long>::value) {
                    if (item.is_number_integer()) {
                        v.push_back(item.get<T>());
                    }
                }
                else if constexpr (std::is_same<T, bool>::value) {
                    if (item.is_boolean()) {
                        v.push_back(item.get<T>());
                    }
                }
            }
        }
        return v;
    }

    template <typename T>
    void set_array(const char* key, std::vector<T> value) {
        json array = json::array();
        for (auto& v : value) {
            array.push_back(v);
        }
        document[key] = array;
    }

    virtual std::string get_json(int indent=0);

    std::string get_json_of_key(std::string key);

    std::string show() {
        std::ostringstream os;
        os << this->get_json();
        os << std::endl;
        return os.str();
    }

    virtual std::shared_ptr<MemoryObject> operator[](std::string key);

    bool operator==(MemoryObject const& b) {
        return (oid_document == b.oid_document);
    }

    friend std::ostream& operator<<(std::ostream& out, MemoryObject& o) {
        out << o.get_json();
        return out;
    }
};

#endif // CHINET_MEMORYOBJECT_H
