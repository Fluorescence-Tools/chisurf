#ifndef CHINET_STORAGEREPLICATION_H
#define CHINET_STORAGEREPLICATION_H

#include <vector>
#include <thread>
#include <atomic>
#include "InMemoryStorage.h"
#include "json.hpp"

using nlohmann::json;

// Distributed storage with master-slave replication
class ReplicatedStorage {
private:
    struct ReplicaNode {
        std::string address;
        std::shared_ptr<InMemoryStorage> storage;
        bool is_master;
        bool is_alive;
        uint64_t last_heartbeat;
        uint64_t lag_ms;
    };
    
    std::vector<ReplicaNode> replicas_;
    std::atomic<int> current_master_{0};
    std::thread replication_thread_;
    std::thread heartbeat_thread_;
    std::atomic<bool> running_{true};
    
    // Write-ahead log for replication
    struct WALEntry {
        uint64_t sequence_num;
        std::string operation;
        std::string key;
        json value;
        uint64_t timestamp;
    };
    
    std::queue<WALEntry> wal_queue_;
    std::mutex wal_mutex_;
    std::condition_variable wal_cv_;
    uint64_t wal_sequence_ = 0;
    
public:
    // Master-slave replication
    void replicateWrite(const std::string& key, const json& value) {
        WALEntry entry{
            ++wal_sequence_,
            "WRITE",
            key,
            value,
            std::chrono::system_clock::now().time_since_epoch().count()
        };
        
        {
            std::lock_guard<std::mutex> lock(wal_mutex_);
            wal_queue_.push(entry);
        }
        wal_cv_.notify_one();
        
        // Synchronous replication to at least one slave
        int replicated_count = 0;
        for (auto& replica : replicas_) {
            if (!replica.is_master && replica.is_alive) {
                if (replica.storage->store(key, value)) {
                    replicated_count++;
                    if (replicated_count >= 1) break; // At least one replica
                }
            }
        }
    }
    
    // Automatic failover
    void performFailover() {
        int new_master = -1;
        uint64_t min_lag = UINT64_MAX;
        
        for (size_t i = 0; i < replicas_.size(); i++) {
            if (!replicas_[i].is_master && replicas_[i].is_alive) {
                if (replicas_[i].lag_ms < min_lag) {
                    min_lag = replicas_[i].lag_ms;
                    new_master = i;
                }
            }
        }
        
        if (new_master != -1) {
            replicas_[current_master_].is_master = false;
            replicas_[new_master].is_master = true;
            current_master_ = new_master;
        }
    }
    
    // Read from slaves for load balancing
    std::optional<json> readWithLoadBalancing(const std::string& key, bool allow_stale = true) {
        if (!allow_stale) {
            // Read from master only
            return replicas_[current_master_].storage->retrieve(key);
        }
        
        // Round-robin read from alive slaves
        static std::atomic<size_t> read_index{0};
        size_t attempts = 0;
        
        while (attempts < replicas_.size()) {
            size_t idx = (read_index++ % replicas_.size());
            if (replicas_[idx].is_alive) {
                auto result = replicas_[idx].storage->retrieve(key);
                if (result) return result;
            }
            attempts++;
        }
        
        return std::nullopt;
    }
};

// Sharding for horizontal scaling
class ShardedStorage {
private:
    struct Shard {
        std::string id;
        std::shared_ptr<InMemoryStorage> storage;
        std::pair<std::string, std::string> key_range;
        size_t size;
    };
    
    std::vector<Shard> shards_;
    std::hash<std::string> hasher_;
    
public:
    // Consistent hashing for shard selection
    size_t selectShard(const std::string& key) {
        return hasher_(key) % shards_.size();
    }
    
    // Auto-resharding when shard grows too large
    void reshard(size_t shard_idx, size_t max_shard_size) {
        if (shards_[shard_idx].size <= max_shard_size) return;
        
        // Create new shard
        Shard new_shard;
        new_shard.id = "shard_" + std::to_string(shards_.size());
        new_shard.storage = std::make_shared<InMemoryStorage>();
        
        // Move half of the data
        auto keys = shards_[shard_idx].storage->listKeys();
        size_t move_count = keys.size() / 2;
        
        for (size_t i = 0; i < move_count; i++) {
            auto value = shards_[shard_idx].storage->retrieve(keys[i]);
            if (value) {
                new_shard.storage->store(keys[i], *value);
                shards_[shard_idx].storage->remove(keys[i]);
            }
        }
        
        shards_.push_back(new_shard);
    }
    
    // Parallel query across shards
    std::vector<json> parallelQuery(const json& query) {
        std::vector<std::future<std::vector<json>>> futures;
        
        for (auto& shard : shards_) {
            futures.push_back(std::async(std::launch::async, [&]() {
                // Execute query on shard
                std::vector<json> results;
                auto keys = shard.storage->listKeys();
                for (const auto& key : keys) {
                    auto doc = shard.storage->retrieve(key);
                    if (doc && QueryEngine::match(*doc, query)) {
                        results.push_back(*doc);
                    }
                }
                return results;
            }));
        }
        
        // Collect results
        std::vector<json> all_results;
        for (auto& future : futures) {
            auto shard_results = future.get();
            all_results.insert(all_results.end(), 
                             shard_results.begin(), 
                             shard_results.end());
        }
        
        return all_results;
    }
};

#endif // CHINET_STORAGEREPLICATION_H
