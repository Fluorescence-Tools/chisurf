#ifndef CHINET_BACKUPMANAGER_H
#define CHINET_BACKUPMANAGER_H

#include <filesystem>
#include <chrono>
#include <thread>
#include <atomic>
#include "InMemoryStorage.h"
#include "json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

class BackupManager {
private:
    struct BackupMetadata {
        std::string backup_id;
        std::string backup_path;
        uint64_t timestamp;
        size_t size_bytes;
        std::string type;  // "full" or "incremental"
        std::string parent_backup_id;  // For incremental backups
        json checksum_map;  // Document checksums for verification
        bool is_compressed;
        bool is_encrypted;
    };
    
    std::shared_ptr<InMemoryStorage> storage_;
    std::string backup_directory_;
    std::vector<BackupMetadata> backup_history_;
    std::thread auto_backup_thread_;
    std::atomic<bool> auto_backup_enabled_{false};
    std::chrono::minutes backup_interval_{60};
    
    // Point-in-time recovery log
    struct PITREntry {
        uint64_t timestamp;
        std::string operation;
        std::string key;
        json old_value;
        json new_value;
    };
    
    std::vector<PITREntry> pitr_log_;
    std::mutex pitr_mutex_;
    
public:
    BackupManager(std::shared_ptr<InMemoryStorage> storage, const std::string& backup_dir)
        : storage_(storage), backup_directory_(backup_dir) {
        fs::create_directories(backup_directory_);
    }
    
    ~BackupManager() {
        stopAutoBackup();
    }
    
    // Full backup
    std::string createFullBackup() {
        std::string backup_id = generateBackupId();
        std::string backup_path = backup_directory_ + "/" + backup_id + "_full.json";
        
        BackupMetadata metadata;
        metadata.backup_id = backup_id;
        metadata.backup_path = backup_path;
        metadata.timestamp = getCurrentTimestamp();
        metadata.type = "full";
        
        // Export all data with checksums
        json backup_data;
        backup_data["metadata"] = {
            {"backup_id", backup_id},
            {"timestamp", metadata.timestamp},
            {"type", "full"}
        };
        
        json documents;
        json checksums;
        
        for (const auto& key : storage_->listKeys()) {
            auto value = storage_->retrieve(key);
            if (value) {
                documents[key] = *value;
                checksums[key] = computeChecksum(*value);
            }
        }
        
        backup_data["documents"] = documents;
        backup_data["checksums"] = checksums;
        
        // Compress if large
        if (documents.dump().size() > 1024 * 1024) {  // > 1MB
            backup_data = compressBackup(backup_data);
            metadata.is_compressed = true;
        }
        
        // Write to file
        std::ofstream file(backup_path);
        file << backup_data.dump(2);
        file.close();
        
        metadata.size_bytes = fs::file_size(backup_path);
        metadata.checksum_map = checksums;
        
        backup_history_.push_back(metadata);
        
        return backup_id;
    }
    
    // Incremental backup (only changes since last backup)
    std::string createIncrementalBackup() {
        if (backup_history_.empty()) {
            return createFullBackup();  // First backup must be full
        }
        
        std::string backup_id = generateBackupId();
        std::string backup_path = backup_directory_ + "/" + backup_id + "_incr.json";
        
        BackupMetadata metadata;
        metadata.backup_id = backup_id;
        metadata.backup_path = backup_path;
        metadata.timestamp = getCurrentTimestamp();
        metadata.type = "incremental";
        metadata.parent_backup_id = backup_history_.back().backup_id;
        
        // Detect changes since last backup
        json changes;
        json checksums;
        const auto& last_checksums = backup_history_.back().checksum_map;
        
        for (const auto& key : storage_->listKeys()) {
            auto value = storage_->retrieve(key);
            if (value) {
                std::string current_checksum = computeChecksum(*value);
                
                // Check if document is new or modified
                if (!last_checksums.contains(key) || 
                    last_checksums[key] != current_checksum) {
                    changes[key] = *value;
                    checksums[key] = current_checksum;
                }
            }
        }
        
        // Check for deleted documents
        json deleted_keys = json::array();
        for (const auto& [key, _] : last_checksums.items()) {
            if (!storage_->retrieve(key).has_value()) {
                deleted_keys.push_back(key);
            }
        }
        
        json backup_data;
        backup_data["metadata"] = {
            {"backup_id", backup_id},
            {"timestamp", metadata.timestamp},
            {"type", "incremental"},
            {"parent_id", metadata.parent_backup_id}
        };
        backup_data["changes"] = changes;
        backup_data["deleted"] = deleted_keys;
        backup_data["checksums"] = checksums;
        
        std::ofstream file(backup_path);
        file << backup_data.dump(2);
        
        metadata.size_bytes = fs::file_size(backup_path);
        metadata.checksum_map = checksums;
        
        backup_history_.push_back(metadata);
        
        return backup_id;
    }
    
    // Restore from backup
    bool restoreFromBackup(const std::string& backup_id) {
        auto metadata = findBackup(backup_id);
        if (!metadata) return false;
        
        if (metadata->type == "full") {
            return restoreFullBackup(*metadata);
        } else {
            // For incremental, need to restore the chain
            return restoreIncrementalChain(*metadata);
        }
    }
    
    // Point-in-time recovery
    bool restoreToPointInTime(uint64_t timestamp) {
        // Find the nearest backup before the timestamp
        BackupMetadata* nearest_backup = nullptr;
        for (auto& backup : backup_history_) {
            if (backup.timestamp <= timestamp) {
                nearest_backup = &backup;
            } else {
                break;
            }
        }
        
        if (!nearest_backup) return false;
        
        // Restore the backup
        if (!restoreFromBackup(nearest_backup->backup_id)) {
            return false;
        }
        
        // Apply PITR log entries up to the timestamp
        std::lock_guard<std::mutex> lock(pitr_mutex_);
        for (const auto& entry : pitr_log_) {
            if (entry.timestamp > nearest_backup->timestamp && 
                entry.timestamp <= timestamp) {
                applyPITREntry(entry);
            }
        }
        
        return true;
    }
    
    // Auto backup management
    void startAutoBackup(std::chrono::minutes interval) {
        backup_interval_ = interval;
        auto_backup_enabled_ = true;
        
        auto_backup_thread_ = std::thread([this]() {
            while (auto_backup_enabled_) {
                std::this_thread::sleep_for(backup_interval_);
                if (auto_backup_enabled_) {
                    createIncrementalBackup();
                }
            }
        });
    }
    
    void stopAutoBackup() {
        auto_backup_enabled_ = false;
        if (auto_backup_thread_.joinable()) {
            auto_backup_thread_.join();
        }
    }
    
    // Backup verification
    bool verifyBackup(const std::string& backup_id) {
        auto metadata = findBackup(backup_id);
        if (!metadata) return false;
        
        std::ifstream file(metadata->backup_path);
        json backup_data;
        file >> backup_data;
        
        if (!backup_data.contains("checksums")) return false;
        
        // Verify each document checksum
        for (const auto& [key, expected_checksum] : backup_data["checksums"].items()) {
            if (backup_data["documents"].contains(key)) {
                std::string actual_checksum = computeChecksum(backup_data["documents"][key]);
                if (actual_checksum != expected_checksum.get<std::string>()) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    // Backup cleanup
    void cleanupOldBackups(size_t max_backups) {
        if (backup_history_.size() <= max_backups) return;
        
        // Keep the most recent backups
        size_t to_delete = backup_history_.size() - max_backups;
        
        for (size_t i = 0; i < to_delete; i++) {
            fs::remove(backup_history_[i].backup_path);
        }
        
        backup_history_.erase(backup_history_.begin(), 
                             backup_history_.begin() + to_delete);
    }
    
    // Log operation for PITR
    void logOperation(const std::string& op, const std::string& key,
                      const json& old_val, const json& new_val) {
        std::lock_guard<std::mutex> lock(pitr_mutex_);
        
        PITREntry entry{
            getCurrentTimestamp(),
            op,
            key,
            old_val,
            new_val
        };
        
        pitr_log_.push_back(entry);
        
        // Trim old PITR entries
        if (pitr_log_.size() > 10000) {
            pitr_log_.erase(pitr_log_.begin(), 
                           pitr_log_.begin() + 1000);
        }
    }
    
    // Get backup statistics
    json getBackupStatistics() {
        json stats;
        stats["total_backups"] = backup_history_.size();
        
        size_t full_count = 0;
        size_t incr_count = 0;
        size_t total_size = 0;
        
        for (const auto& backup : backup_history_) {
            if (backup.type == "full") {
                full_count++;
            } else {
                incr_count++;
            }
            total_size += backup.size_bytes;
        }
        
        stats["full_backups"] = full_count;
        stats["incremental_backups"] = incr_count;
        stats["total_size_bytes"] = total_size;
        stats["auto_backup_enabled"] = auto_backup_enabled_.load();
        stats["backup_interval_minutes"] = backup_interval_.count();
        
        return stats;
    }
    
private:
    std::string generateBackupId() {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        return "backup_" + std::to_string(timestamp);
    }
    
    uint64_t getCurrentTimestamp() {
        return std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    std::string computeChecksum(const json& data) {
        std::hash<std::string> hasher;
        return std::to_string(hasher(data.dump()));
    }
    
    json compressBackup(const json& data) {
        // Implementation would use zlib or similar
        return data;
    }
    
    std::optional<BackupMetadata> findBackup(const std::string& backup_id) {
        for (const auto& backup : backup_history_) {
            if (backup.backup_id == backup_id) {
                return backup;
            }
        }
        return std::nullopt;
    }
    
    bool restoreFullBackup(const BackupMetadata& metadata) {
        std::ifstream file(metadata.backup_path);
        json backup_data;
        file >> backup_data;
        
        storage_->clear();
        
        for (const auto& [key, value] : backup_data["documents"].items()) {
            storage_->store(key, value);
        }
        
        return true;
    }
    
    bool restoreIncrementalChain(const BackupMetadata& metadata) {
        // Build the chain of backups
        std::vector<BackupMetadata> chain;
        
        const BackupMetadata* current = &metadata;
        chain.push_back(*current);
        
        while (current->type == "incremental") {
            auto parent = findBackup(current->parent_backup_id);
            if (!parent) return false;
            
            chain.push_back(*parent);
            current = &chain.back();
        }
        
        // Apply backups in reverse order (oldest first)
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            if (it->type == "full") {
                if (!restoreFullBackup(*it)) return false;
            } else {
                if (!applyIncrementalBackup(*it)) return false;
            }
        }
        
        return true;
    }
    
    bool applyIncrementalBackup(const BackupMetadata& metadata) {
        std::ifstream file(metadata.backup_path);
        json backup_data;
        file >> backup_data;
        
        // Apply changes
        for (const auto& [key, value] : backup_data["changes"].items()) {
            storage_->store(key, value);
        }
        
        // Apply deletions
        for (const auto& key : backup_data["deleted"]) {
            storage_->remove(key.get<std::string>());
        }
        
        return true;
    }
    
    void applyPITREntry(const PITREntry& entry) {
        if (entry.operation == "CREATE" || entry.operation == "UPDATE") {
            storage_->store(entry.key, entry.new_value);
        } else if (entry.operation == "DELETE") {
            storage_->remove(entry.key);
        }
    }
};

#endif // CHINET_BACKUPMANAGER_H
