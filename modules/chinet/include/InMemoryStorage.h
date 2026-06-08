#ifndef CHINET_INMEMORYSTORAGE_H
#define CHINET_INMEMORYSTORAGE_H

#include <unordered_map>
#include <shared_mutex>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <optional>
#include "json.hpp"

using nlohmann::json;

// Forward declarations
class StorageObject;
class StorageTransaction;

// Storage backend interface
class IStorageBackend {
public:
    virtual ~IStorageBackend() = default;
    virtual bool store(const std::string& key, const json& value) = 0;
    virtual std::optional<json> retrieve(const std::string& key) = 0;
    virtual bool remove(const std::string& key) = 0;
    virtual std::vector<std::string> listKeys(const std::string& prefix = "") = 0;
    virtual bool persist(const std::string& filename) = 0;
    virtual bool load(const std::string& filename) = 0;
    virtual void clear() = 0;
    virtual size_t size() const = 0;
};

// Thread-safe in-memory storage implementation
class InMemoryStorage : public IStorageBackend {
private:
    struct StorageEntry {
        json data;
        uint64_t timestamp;
        uint64_t version;
        std::string checksum;
        std::vector<std::string> tags;
        
        StorageEntry() : timestamp(0), version(1) {}
        explicit StorageEntry(const json& d) 
            : data(d), 
              timestamp(std::chrono::system_clock::now().time_since_epoch().count()),
              version(1) {
            updateChecksum();
        }
        
        void updateChecksum();
        bool validateChecksum() const;
    };
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, StorageEntry> storage_;
    std::unordered_map<std::string, std::vector<StorageEntry>> history_; // Key -> version history
    
    // Configuration
    bool enable_versioning_ = true;
    size_t max_versions_per_key_ = 100;
    bool enable_compression_ = false;
    
    // Statistics
    mutable std::atomic<uint64_t> read_count_{0};
    mutable std::atomic<uint64_t> write_count_{0};
    mutable std::atomic<uint64_t> hit_count_{0};
    mutable std::atomic<uint64_t> miss_count_{0};
    
public:
    InMemoryStorage();
    ~InMemoryStorage() override = default;
    
    // Basic operations
    bool store(const std::string& key, const json& value) override;
    std::optional<json> retrieve(const std::string& key) override;
    bool remove(const std::string& key) override;
    std::vector<std::string> listKeys(const std::string& prefix = "") override;
    
    // Persistence
    bool persist(const std::string& filename) override;
    bool load(const std::string& filename) override;
    
    // Memory management
    void clear() override;
    size_t size() const override;
    size_t memoryUsage() const;
    
    // Versioning
    bool storeWithVersion(const std::string& key, const json& value, uint64_t version);
    std::optional<json> retrieveVersion(const std::string& key, uint64_t version);
    std::vector<uint64_t> listVersions(const std::string& key);
    bool rollbackToVersion(const std::string& key, uint64_t version);
    
    // Transactions
    std::shared_ptr<StorageTransaction> beginTransaction();
    bool commitTransaction(std::shared_ptr<StorageTransaction> tx);
    void rollbackTransaction(std::shared_ptr<StorageTransaction> tx);
    
    // Querying
    std::vector<std::string> findByTag(const std::string& tag);
    std::vector<std::string> query(const json& criteria);
    
    // Configuration
    void setVersioning(bool enable) { enable_versioning_ = enable; }
    void setMaxVersions(size_t max) { max_versions_per_key_ = max; }
    void setCompression(bool enable) { enable_compression_ = enable; }
    
    // Statistics
    json getStatistics() const;
    void resetStatistics();
};

// Transaction support
class StorageTransaction {
private:
    friend class InMemoryStorage;
    
    std::unordered_map<std::string, json> pending_writes_;
    std::unordered_set<std::string> pending_deletes_;
    std::vector<std::pair<std::string, json>> operations_log_;
    bool is_committed_ = false;
    bool is_rolled_back_ = false;
    
public:
    void set(const std::string& key, const json& value);
    std::optional<json> get(const std::string& key);
    void remove(const std::string& key);
    bool hasChanges() const;
    std::vector<std::string> getModifiedKeys() const;
};

// Specialized storage for different object types
class ObjectStorage {
private:
    std::shared_ptr<InMemoryStorage> storage_;
    std::string type_prefix_;
    
public:
    ObjectStorage(std::shared_ptr<InMemoryStorage> storage, const std::string& type)
        : storage_(storage), type_prefix_(type + ":") {}
    
    bool store(const std::string& id, const json& object);
    std::optional<json> retrieve(const std::string& id);
    bool remove(const std::string& id);
    std::vector<std::string> listIds();
    std::vector<json> getAll();
};

// Storage manager for the entire application
class StorageManager {
private:
    static std::shared_ptr<StorageManager> instance_;
    std::shared_ptr<InMemoryStorage> main_storage_;
    std::unordered_map<std::string, std::shared_ptr<ObjectStorage>> type_stores_;
    
    StorageManager();
    
public:
    static std::shared_ptr<StorageManager> getInstance();
    
    // Object type management
    std::shared_ptr<ObjectStorage> getStorage(const std::string& type);
    void registerType(const std::string& type);
    
    // Global operations
    bool persistAll(const std::string& directory);
    bool loadAll(const std::string& directory);
    void clearAll();
    
    // Direct access to main storage
    std::shared_ptr<InMemoryStorage> getMainStorage() { return main_storage_; }
    
    // Statistics
    json getGlobalStatistics();
};

// Drop-in replacement for MongoDB operations
class MongoReplacement {
private:
    std::shared_ptr<ObjectStorage> storage_;
    
public:
    explicit MongoReplacement(const std::string& collection_name);
    
    // MongoDB-like interface
    std::string insert(const json& document);
    bool update(const std::string& id, const json& document);
    std::optional<json> findOne(const std::string& id);
    std::optional<json> findOne(const json& query);
    std::vector<json> find(const json& query = {});
    bool deleteOne(const std::string& id);
    bool deleteMany(const json& query);
    size_t count(const json& query = {});
    
    // Indexing (simplified)
    void createIndex(const std::string& field);
    void dropIndex(const std::string& field);
    
private:
    bool matchesQuery(const json& document, const json& query);
    std::string generateId();
};

#endif // CHINET_INMEMORYSTORAGE_H
