#ifndef CHINET_INDEXMANAGER_H
#define CHINET_INDEXMANAGER_H

#include <unordered_map>
#include <map>
#include <set>
#include <shared_mutex>
#include "json.hpp"

using nlohmann::json;

class IndexManager {
private:
    struct Index {
        std::multimap<json, std::string> btree_index;  // For range queries
        std::unordered_map<std::string, std::set<std::string>> hash_index;  // For equality
        bool is_unique = false;
        bool is_sparse = false;
    };
    
    std::unordered_map<std::string, Index> indices_;
    mutable std::shared_mutex mutex_;
    
public:
    // Create different types of indices
    void createIndex(const std::string& field, bool unique = false, bool sparse = false) {
        std::unique_lock lock(mutex_);
        indices_[field] = Index{};
        indices_[field].is_unique = unique;
        indices_[field].is_sparse = sparse;
    }
    
    // Add document to indices
    void indexDocument(const std::string& doc_id, const json& document) {
        std::unique_lock lock(mutex_);
        
        for (auto& [field, index] : indices_) {
            if (document.contains(field)) {
                const auto& value = document[field];
                
                // Add to btree index for range queries
                index.btree_index.emplace(value, doc_id);
                
                // Add to hash index for equality queries
                std::string value_str = value.dump();
                index.hash_index[value_str].insert(doc_id);
            } else if (!index.is_sparse) {
                // Index null values if not sparse
                index.btree_index.emplace(nullptr, doc_id);
                index.hash_index["null"].insert(doc_id);
            }
        }
    }
    
    // Remove document from indices
    void removeDocument(const std::string& doc_id, const json& document) {
        std::unique_lock lock(mutex_);
        
        for (auto& [field, index] : indices_) {
            if (document.contains(field)) {
                const auto& value = document[field];
                
                // Remove from btree
                auto range = index.btree_index.equal_range(value);
                for (auto it = range.first; it != range.second; ) {
                    if (it->second == doc_id) {
                        it = index.btree_index.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Remove from hash index
                std::string value_str = value.dump();
                index.hash_index[value_str].erase(doc_id);
                if (index.hash_index[value_str].empty()) {
                    index.hash_index.erase(value_str);
                }
            }
        }
    }
    
    // Find documents by indexed field
    std::set<std::string> findByIndex(const std::string& field, const json& value) {
        std::shared_lock lock(mutex_);
        
        if (indices_.find(field) == indices_.end()) {
            return {};
        }
        
        auto& index = indices_[field];
        std::string value_str = value.dump();
        
        if (index.hash_index.find(value_str) != index.hash_index.end()) {
            return index.hash_index[value_str];
        }
        
        return {};
    }
    
    // Range queries using btree index
    std::set<std::string> findInRange(const std::string& field, 
                                      const json& min_value, 
                                      const json& max_value,
                                      bool include_min = true,
                                      bool include_max = true) {
        std::shared_lock lock(mutex_);
        
        if (indices_.find(field) == indices_.end()) {
            return {};
        }
        
        std::set<std::string> results;
        auto& btree = indices_[field].btree_index;
        
        auto start = include_min ? btree.lower_bound(min_value) : btree.upper_bound(min_value);
        auto end = include_max ? btree.upper_bound(max_value) : btree.lower_bound(max_value);
        
        for (auto it = start; it != end; ++it) {
            results.insert(it->second);
        }
        
        return results;
    }
    
    // Get index statistics
    json getIndexStats() {
        std::shared_lock lock(mutex_);
        json stats;
        
        for (const auto& [field, index] : indices_) {
            json field_stats;
            field_stats["btree_size"] = index.btree_index.size();
            field_stats["hash_buckets"] = index.hash_index.size();
            field_stats["is_unique"] = index.is_unique;
            field_stats["is_sparse"] = index.is_sparse;
            stats[field] = field_stats;
        }
        
        return stats;
    }
};

#endif // CHINET_INDEXMANAGER_H
