#include "InMemoryStorage.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/md5.h>
#include <random>

// Helper function to compute MD5 checksum
std::string computeMD5(const std::string& data) {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(data.c_str()), data.length(), digest);
    
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return ss.str();
}

// StorageEntry implementation
void InMemoryStorage::StorageEntry::updateChecksum() {
    checksum = computeMD5(data.dump());
}

bool InMemoryStorage::StorageEntry::validateChecksum() const {
    return checksum == computeMD5(data.dump());
}

// InMemoryStorage implementation
InMemoryStorage::InMemoryStorage() {}

bool InMemoryStorage::store(const std::string& key, const json& value) {
    std::unique_lock lock(mutex_);
    
    write_count_++;
    
    // Handle versioning
    if (enable_versioning_ && storage_.find(key) != storage_.end()) {
        auto& current = storage_[key];
        if (history_[key].size() >= max_versions_per_key_) {
            history_[key].erase(history_[key].begin());
        }
        history_[key].push_back(current);
    }
    
    StorageEntry entry(value);
    if (storage_.find(key) != storage_.end()) {
        entry.version = storage_[key].version + 1;
    }
    
    storage_[key] = entry;
    return true;
}

std::optional<json> InMemoryStorage::retrieve(const std::string& key) {
    std::shared_lock lock(mutex_);
    
    read_count_++;
    
    auto it = storage_.find(key);
    if (it != storage_.end()) {
        hit_count_++;
        if (!it->second.validateChecksum()) {
            // Data corruption detected
            return std::nullopt;
        }
        return it->second.data;
    }
    
    miss_count_++;
    return std::nullopt;
}

bool InMemoryStorage::remove(const std::string& key) {
    std::unique_lock lock(mutex_);
    
    auto it = storage_.find(key);
    if (it != storage_.end()) {
        if (enable_versioning_) {
            history_[key].push_back(it->second);
        }
        storage_.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> InMemoryStorage::listKeys(const std::string& prefix) {
    std::shared_lock lock(mutex_);
    
    std::vector<std::string> keys;
    for (const auto& [key, _] : storage_) {
        if (prefix.empty() || key.find(prefix) == 0) {
            keys.push_back(key);
        }
    }
    return keys;
}

bool InMemoryStorage::persist(const std::string& filename) {
    std::shared_lock lock(mutex_);
    
    try {
        json output;
        output["version"] = 1;
        output["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        
        json data;
        for (const auto& [key, entry] : storage_) {
            json entry_json;
            entry_json["data"] = entry.data;
            entry_json["timestamp"] = entry.timestamp;
            entry_json["version"] = entry.version;
            entry_json["checksum"] = entry.checksum;
            entry_json["tags"] = entry.tags;
            data[key] = entry_json;
        }
        output["storage"] = data;
        
        if (enable_versioning_) {
            json history_json;
            for (const auto& [key, versions] : history_) {
                json versions_array = json::array();
                for (const auto& ver : versions) {
                    json ver_json;
                    ver_json["data"] = ver.data;
                    ver_json["timestamp"] = ver.timestamp;
                    ver_json["version"] = ver.version;
                    ver_json["checksum"] = ver.checksum;
                    versions_array.push_back(ver_json);
                }
                history_json[key] = versions_array;
            }
            output["history"] = history_json;
        }
        
        std::ofstream file(filename);
        file << output.dump(2);
        return true;
    } catch (...) {
        return false;
    }
}

bool InMemoryStorage::load(const std::string& filename) {
    std::unique_lock lock(mutex_);
    
    try {
        std::ifstream file(filename);
        json input;
        file >> input;
        
        clear();
        
        if (input.contains("storage")) {
            for (const auto& [key, entry_json] : input["storage"].items()) {
                StorageEntry entry;
                entry.data = entry_json["data"];
                entry.timestamp = entry_json["timestamp"];
                entry.version = entry_json["version"];
                entry.checksum = entry_json["checksum"];
                if (entry_json.contains("tags")) {
                    entry.tags = entry_json["tags"].get<std::vector<std::string>>();
                }
                storage_[key] = entry;
            }
        }
        
        if (enable_versioning_ && input.contains("history")) {
            for (const auto& [key, versions_array] : input["history"].items()) {
                std::vector<StorageEntry> versions;
                for (const auto& ver_json : versions_array) {
                    StorageEntry ver;
                    ver.data = ver_json["data"];
                    ver.timestamp = ver_json["timestamp"];
                    ver.version = ver_json["version"];
                    ver.checksum = ver_json["checksum"];
                    versions.push_back(ver);
                }
                history_[key] = versions;
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void InMemoryStorage::clear() {
    storage_.clear();
    history_.clear();
}

size_t InMemoryStorage::size() const {
    std::shared_lock lock(mutex_);
    return storage_.size();
}

size_t InMemoryStorage::memoryUsage() const {
    std::shared_lock lock(mutex_);
    size_t total = 0;
    
    for (const auto& [key, entry] : storage_) {
        total += key.size();
        total += entry.data.dump().size();
        total += entry.checksum.size();
        total += sizeof(entry.timestamp) + sizeof(entry.version);
    }
    
    for (const auto& [key, versions] : history_) {
        total += key.size();
        for (const auto& ver : versions) {
            total += ver.data.dump().size();
            total += ver.checksum.size();
            total += sizeof(ver.timestamp) + sizeof(ver.version);
        }
    }
    
    return total;
}

std::shared_ptr<StorageTransaction> InMemoryStorage::beginTransaction() {
    return std::make_shared<StorageTransaction>();
}

bool InMemoryStorage::commitTransaction(std::shared_ptr<StorageTransaction> tx) {
    if (tx->is_committed_ || tx->is_rolled_back_) {
        return false;
    }
    
    std::unique_lock lock(mutex_);
    
    // Apply all pending writes
    for (const auto& [key, value] : tx->pending_writes_) {
        store(key, value);
    }
    
    // Apply all pending deletes
    for (const auto& key : tx->pending_deletes_) {
        remove(key);
    }
    
    tx->is_committed_ = true;
    return true;
}

void InMemoryStorage::rollbackTransaction(std::shared_ptr<StorageTransaction> tx) {
    tx->is_rolled_back_ = true;
    tx->pending_writes_.clear();
    tx->pending_deletes_.clear();
}

json InMemoryStorage::getStatistics() const {
    json stats;
    stats["size"] = size();
    stats["memory_usage"] = memoryUsage();
    stats["read_count"] = read_count_.load();
    stats["write_count"] = write_count_.load();
    stats["hit_count"] = hit_count_.load();
    stats["miss_count"] = miss_count_.load();
    
    double hit_rate = 0.0;
    if (read_count_ > 0) {
        hit_rate = static_cast<double>(hit_count_) / read_count_;
    }
    stats["hit_rate"] = hit_rate;
    
    return stats;
}

void InMemoryStorage::resetStatistics() {
    read_count_ = 0;
    write_count_ = 0;
    hit_count_ = 0;
    miss_count_ = 0;
}

// StorageTransaction implementation
void StorageTransaction::set(const std::string& key, const json& value) {
    if (is_committed_ || is_rolled_back_) return;
    
    pending_writes_[key] = value;
    pending_deletes_.erase(key);
    operations_log_.push_back({key, value});
}

std::optional<json> StorageTransaction::get(const std::string& key) {
    if (pending_writes_.find(key) != pending_writes_.end()) {
        return pending_writes_[key];
    }
    return std::nullopt;
}

void StorageTransaction::remove(const std::string& key) {
    if (is_committed_ || is_rolled_back_) return;
    
    pending_deletes_.insert(key);
    pending_writes_.erase(key);
}

bool StorageTransaction::hasChanges() const {
    return !pending_writes_.empty() || !pending_deletes_.empty();
}

std::vector<std::string> StorageTransaction::getModifiedKeys() const {
    std::vector<std::string> keys;
    for (const auto& [key, _] : pending_writes_) {
        keys.push_back(key);
    }
    for (const auto& key : pending_deletes_) {
        keys.push_back(key);
    }
    return keys;
}

// ObjectStorage implementation
bool ObjectStorage::store(const std::string& id, const json& object) {
    return storage_->store(type_prefix_ + id, object);
}

std::optional<json> ObjectStorage::retrieve(const std::string& id) {
    return storage_->retrieve(type_prefix_ + id);
}

bool ObjectStorage::remove(const std::string& id) {
    return storage_->remove(type_prefix_ + id);
}

std::vector<std::string> ObjectStorage::listIds() {
    auto keys = storage_->listKeys(type_prefix_);
    std::vector<std::string> ids;
    for (const auto& key : keys) {
        ids.push_back(key.substr(type_prefix_.length()));
    }
    return ids;
}

std::vector<json> ObjectStorage::getAll() {
    std::vector<json> objects;
    auto ids = listIds();
    for (const auto& id : ids) {
        auto obj = retrieve(id);
        if (obj) {
            objects.push_back(*obj);
        }
    }
    return objects;
}

// StorageManager implementation
std::shared_ptr<StorageManager> StorageManager::instance_ = nullptr;

StorageManager::StorageManager() {
    main_storage_ = std::make_shared<InMemoryStorage>();
}

std::shared_ptr<StorageManager> StorageManager::getInstance() {
    if (!instance_) {
        instance_ = std::shared_ptr<StorageManager>(new StorageManager());
    }
    return instance_;
}

std::shared_ptr<ObjectStorage> StorageManager::getStorage(const std::string& type) {
    if (type_stores_.find(type) == type_stores_.end()) {
        registerType(type);
    }
    return type_stores_[type];
}

void StorageManager::registerType(const std::string& type) {
    type_stores_[type] = std::make_shared<ObjectStorage>(main_storage_, type);
}

bool StorageManager::persistAll(const std::string& directory) {
    return main_storage_->persist(directory + "/chinet_storage.json");
}

bool StorageManager::loadAll(const std::string& directory) {
    return main_storage_->load(directory + "/chinet_storage.json");
}

void StorageManager::clearAll() {
    main_storage_->clear();
    type_stores_.clear();
}

json StorageManager::getGlobalStatistics() {
    json stats;
    stats["main_storage"] = main_storage_->getStatistics();
    stats["registered_types"] = type_stores_.size();
    
    json type_stats;
    for (const auto& [type, storage] : type_stores_) {
        type_stats[type] = storage->listIds().size();
    }
    stats["objects_by_type"] = type_stats;
    
    return stats;
}

// MongoReplacement implementation
MongoReplacement::MongoReplacement(const std::string& collection_name) {
    storage_ = StorageManager::getInstance()->getStorage(collection_name);
}

std::string MongoReplacement::generateId() {
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

std::string MongoReplacement::insert(const json& document) {
    std::string id = generateId();
    json doc_with_id = document;
    doc_with_id["_id"] = id;
    storage_->store(id, doc_with_id);
    return id;
}

bool MongoReplacement::update(const std::string& id, const json& document) {
    json doc_with_id = document;
    doc_with_id["_id"] = id;
    return storage_->store(id, doc_with_id);
}

std::optional<json> MongoReplacement::findOne(const std::string& id) {
    return storage_->retrieve(id);
}

std::optional<json> MongoReplacement::findOne(const json& query) {
    auto all = storage_->getAll();
    for (const auto& doc : all) {
        if (matchesQuery(doc, query)) {
            return doc;
        }
    }
    return std::nullopt;
}

std::vector<json> MongoReplacement::find(const json& query) {
    std::vector<json> results;
    auto all = storage_->getAll();
    
    for (const auto& doc : all) {
        if (query.empty() || matchesQuery(doc, query)) {
            results.push_back(doc);
        }
    }
    
    return results;
}

bool MongoReplacement::deleteOne(const std::string& id) {
    return storage_->remove(id);
}

bool MongoReplacement::deleteMany(const json& query) {
    auto matching = find(query);
    bool success = true;
    
    for (const auto& doc : matching) {
        if (doc.contains("_id")) {
            success &= storage_->remove(doc["_id"].get<std::string>());
        }
    }
    
    return success;
}

size_t MongoReplacement::count(const json& query) {
    if (query.empty()) {
        return storage_->listIds().size();
    }
    return find(query).size();
}

bool MongoReplacement::matchesQuery(const json& document, const json& query) {
    for (const auto& [key, value] : query.items()) {
        if (!document.contains(key) || document[key] != value) {
            return false;
        }
    }
    return true;
}
