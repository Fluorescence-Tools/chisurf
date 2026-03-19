#include "MemoryObject.h"

using json = nlohmann::json;  // If not already aliased


// Initialize static members
std::list<std::shared_ptr<MemoryObject>> MemoryObject::registered_objects = std::list<std::shared_ptr<MemoryObject>>();
std::unordered_map<std::string, json> MemoryObject::object_store = std::unordered_map<std::string, json>();
std::mutex MemoryObject::object_store_mutex;

MemoryObject::MemoryObject(std::string name) :
    object_name(name),
    document(json::object()),
    time_of_death(0) {

    // Generate a unique OID for this object
    oid_document = generate_oid();
    oid_precursor = oid_document;

    // Initialize the document with basic fields
    document["_id"] = oid_document;
    document["precursor"] = oid_precursor;
    document["death"] = time_of_death;
    document["name"] = name;

    if (is_chinet_verbose()) {
        std::clog << "NEW MEMORYOBJECT" << std::endl;
    }
}

void MemoryObject::set_document(json doc) {
    // Set the document to the input JSON
    document = doc;

    // Update the object fields from the document
    if (document.contains("_id") && document["_id"].is_string()) {
        oid_document = document["_id"].get<std::string>();
    }

    if (document.contains("precursor") && document["precursor"].is_string()) {
        oid_precursor = document["precursor"].get<std::string>();
    }

    if (document.contains("death") && document["death"].is_number()) {
        time_of_death = document["death"].get<uint64_t>();
    }

    if (document.contains("name") && document["name"].is_string()) {
        object_name = document["name"].get<std::string>();
    }

    // Handle the "value" field if it exists
    // This is important for Port objects that have a value field
    // The actual handling of the value field is done in the derived classes
}

MemoryObject::~MemoryObject() {
    if (is_chinet_verbose()) {
        std::clog << "DESTROYING MEMORYOBJECT" << std::endl;
    }

    // Remove this object from the registered objects list
    // Don't use shared_from_this() in the destructor as it can throw std::bad_weak_ptr
    unregister_instance(nullptr);

    // Set time of death
    time_of_death = std::time(nullptr);

    // Update the death field in the document
    document["death"] = time_of_death;

    // If connected to the database, write the updated document
    if (is_connected_to_db_) {
        write_to_db();
    }
}

bool MemoryObject::connect_to_db(
        const std::string& uri_string,
        const std::string& db_string,
        const std::string& app_string,
        const std::string& collection_string) {

    // In the memory implementation, we don't actually connect to a database
    // but we'll set the flag to indicate we're "connected"
    is_connected_to_db_ = true;

    // Register this instance
    try {
        register_instance(shared_from_this());
    } catch (const std::bad_weak_ptr&) {
        // If shared_from_this() fails, log a warning but continue
        // This can happen if the object wasn't created with make_shared
        if (is_chinet_verbose()) {
            std::clog << "Warning: Could not register instance in connect_to_db (shared_from_this() failed)" << std::endl;
        }
    }

    return true;
}

void MemoryObject::disconnect_from_db() {
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::disconnect_from_db] disconnecting object '" << object_name << "'" << std::endl;
    }
    is_connected_to_db_ = false;
}

bool MemoryObject::is_connected_to_db() {
    return is_connected_to_db_;
}

void MemoryObject::register_instance(std::shared_ptr<MemoryObject> x) {
    if (x == nullptr) {
        // If x is null, try to use shared_from_this() to get a valid shared_ptr
        try {
            x = shared_from_this();
        } catch (const std::bad_weak_ptr&) {
            // If shared_from_this() fails, the object is already being destroyed
            // and we can't get a valid shared_ptr, so just return
            if (is_chinet_verbose()) {
                std::clog << "Warning: Could not register instance (null shared_ptr and shared_from_this() failed)" << std::endl;
            }
            return;
        }
    }

    // Only add the object if it's not already in the list
    if (std::find(registered_objects.begin(), registered_objects.end(), x) == registered_objects.end()) {
        registered_objects.push_back(x);
    }

    if (is_chinet_verbose()) {
        std::clog << "-- Total number of MemoryObject instances: " << registered_objects.size() << std::endl;
    }
}

void MemoryObject::unregister_instance(std::shared_ptr<MemoryObject> x) {
    if (x) {
        registered_objects.remove(x);
    } else {
        // If x is null, try to use shared_from_this() to get a valid shared_ptr
        try {
            registered_objects.remove(shared_from_this());
        } catch (const std::bad_weak_ptr&) {
            // If shared_from_this() fails, the object is already being destroyed
            // and we can't get a valid shared_ptr, so just ignore it
            if (is_chinet_verbose()) {
                std::clog << "Warning: Could not unregister instance (null shared_ptr and shared_from_this() failed)" << std::endl;
            }
        }
    }
}

std::list<std::shared_ptr<MemoryObject>> MemoryObject::get_instances() {
    return registered_objects;
}

bool MemoryObject::write_to_db() {
    if (!is_connected_to_db_) {
        std::cerr << "Error: Not connected to database" << std::endl;
        return false;
    }
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::write_to_db] oid=" << oid_document << ", name='" << object_name << "'" << std::endl;
    }

    // Update the document with the latest field values
    document["_id"] = oid_document;
    document["precursor"] = oid_precursor;
    document["death"] = time_of_death;
    document["name"] = object_name;

    // Store the document in the object store
    std::lock_guard<std::mutex> lock(object_store_mutex);
    object_store[oid_document] = document;

    return true;
}

std::string MemoryObject::create_copy_in_db() {
    // Create a new OID for the copy
    std::string new_oid = generate_oid();
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::create_copy_in_db] new_oid=" << new_oid << " from precursor=" << oid_document << std::endl;
    }

    // Create a copy of the document with the new OID
    json copy = document;
    copy["_id"] = new_oid;

    // Set the precursor of the copy to the current document's OID
    copy["precursor"] = oid_document;

    // Make sure all required fields are in the copy
    if (!copy.contains("death")) {
        copy["death"] = time_of_death;
    }

    if (!copy.contains("name")) {
        copy["name"] = object_name;
    }

    // Store the copy in the object store
    std::lock_guard<std::mutex> lock(object_store_mutex);
    object_store[new_oid] = copy;

    return new_oid;
}

bool MemoryObject::read_from_db(const std::string& oid_string) {
    if (!is_connected_to_db_) {
        std::cerr << "Error: Not connected to database" << std::endl;
        return false;
    }
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::read_from_db] oid=" << oid_string << std::endl;
    }

    // Look up the document in the object store
    std::lock_guard<std::mutex> lock(object_store_mutex);
    auto it = object_store.find(oid_string);
    if (it == object_store.end()) {
        std::cerr << "Error: Document not found with OID " << oid_string << std::endl;
        return false;
    }

    // Copy the document
    document = it->second;
    oid_document = oid_string;

    // Update the object fields from the document
    if (document.contains("precursor") && document["precursor"].is_string()) {
        oid_precursor = document["precursor"].get<std::string>();
    }

    if (document.contains("death") && document["death"].is_number()) {
        time_of_death = document["death"].get<uint64_t>();
    }

    if (document.contains("name") && document["name"].is_string()) {
        object_name = document["name"].get<std::string>();
    }

    return true;
}

bool MemoryObject::read_json(std::string json_string) {
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::read_json] input_size=" << json_string.size() << std::endl;
    }
    try {
        document = json::parse(json_string);

        // Update the fields from the document
        if (document.contains("_id") && document["_id"].is_string()) {
            oid_document = document["_id"].get<std::string>();
        }

        if (document.contains("precursor") && document["precursor"].is_string()) {
            oid_precursor = document["precursor"].get<std::string>();
        }

        if (document.contains("death") && document["death"].is_number()) {
            time_of_death = document["death"].get<uint64_t>();
        }

        if (document.contains("name") && document["name"].is_string()) {
            object_name = document["name"].get<std::string>();
        }

        return true;
    } catch (const json::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return false;
    }
}

std::shared_ptr<MemoryObject> MemoryObject::get_ptr() {
    try {
        return shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        // If shared_from_this() fails, log a warning and return nullptr
        if (is_chinet_verbose()) {
            std::clog << "Warning: get_ptr() failed (shared_from_this() threw bad_weak_ptr)" << std::endl;
        }
        return nullptr;
    }
}

std::string MemoryObject::get_json(int indent) {
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::get_json] indent=" << indent << std::endl;
    }
    // Make sure all required fields are in the document
    json doc = document;

    // Ensure _id is in the document
    if (!doc.contains("_id")) {
        doc["_id"] = oid_document;
    }

    // Ensure precursor is in the document
    if (!doc.contains("precursor")) {
        doc["precursor"] = oid_precursor;
    }

    // Ensure death is in the document
    if (!doc.contains("death")) {
        doc["death"] = time_of_death;
    }

    // Ensure name is in the document
    if (!doc.contains("name")) {
        doc["name"] = object_name;
    }

    // Ensure value is in the document if it exists in the original document
    // Note: For Port objects, the value might be dynamic and need to be updated
    // The actual handling of the value field is done in the derived classes (e.g., Port)
    if (document.contains("value")) {
        doc["value"] = document["value"];
    }

    return doc.dump(indent);
}

std::string MemoryObject::get_json_of_key(std::string key) {
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::get_json_of_key] key='" << key << "'" << std::endl;
    }
    if (document.contains(key)) {
        return document[key].dump();
    }
    return "{}";
}

std::shared_ptr<MemoryObject> MemoryObject::operator[](std::string key) {
    if (is_chinet_verbose()) {
        std::clog << "[MemoryObject::operator[]] key='" << key << "'" << std::endl;
    }
    // This is a placeholder implementation
    // In a real implementation, this would return a child object
    return nullptr;
}
