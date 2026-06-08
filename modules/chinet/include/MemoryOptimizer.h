#ifndef CHINET_MEMORYOPTIMIZER_H
#define CHINET_MEMORYOPTIMIZER_H

#include <memory>
#include <chrono>
#include <zlib.h>
#include "json.hpp"

using nlohmann::json;

class MemoryOptimizer {
private:
    struct CacheEntry {
        json data;
        uint64_t last_access;
        uint64_t access_count;
        size_t size_bytes;
        bool is_compressed;
        std::vector<uint8_t> compressed_data;
    };
    
    size_t max_memory_bytes_;
    size_t current_memory_bytes_ = 0;
    size_t compression_threshold_ = 1024; // Compress if larger than 1KB
    
public:
    // LRU eviction policy
    std::string selectEvictionCandidate(const std::unordered_map<std::string, CacheEntry>& cache) {
        std::string oldest_key;
        uint64_t oldest_time = UINT64_MAX;
        
        for (const auto& [key, entry] : cache) {
            if (entry.last_access < oldest_time) {
                oldest_time = entry.last_access;
                oldest_key = key;
            }
        }
        
        return oldest_key;
    }
    
    // LFU eviction policy  
    std::string selectLFUCandidate(const std::unordered_map<std::string, CacheEntry>& cache) {
        std::string least_used_key;
        uint64_t min_count = UINT64_MAX;
        
        for (const auto& [key, entry] : cache) {
            if (entry.access_count < min_count) {
                min_count = entry.access_count;
                least_used_key = key;
            }
        }
        
        return least_used_key;
    }
    
    // Compress JSON using zlib
    std::vector<uint8_t> compress(const json& data) {
        std::string str = data.dump();
        
        uLongf compressed_size = compressBound(str.size());
        std::vector<uint8_t> compressed(compressed_size);
        
        int result = compress2(
            compressed.data(), &compressed_size,
            reinterpret_cast<const Bytef*>(str.data()), str.size(),
            Z_BEST_COMPRESSION
        );
        
        if (result == Z_OK) {
            compressed.resize(compressed_size);
            return compressed;
        }
        
        return {};
    }
    
    // Decompress data
    json decompress(const std::vector<uint8_t>& compressed_data, size_t original_size) {
        std::vector<uint8_t> decompressed(original_size);
        uLongf decompressed_size = original_size;
        
        int result = uncompress(
            decompressed.data(), &decompressed_size,
            compressed_data.data(), compressed_data.size()
        );
        
        if (result == Z_OK) {
            std::string str(decompressed.begin(), decompressed.begin() + decompressed_size);
            return json::parse(str);
        }
        
        return {};
    }
    
    // Memory-mapped file support for large datasets
    class MappedStorage {
    private:
        int fd_;
        void* mapped_memory_;
        size_t file_size_;
        
    public:
        bool mapFile(const std::string& filename, size_t size) {
            fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0644);
            if (fd_ == -1) return false;
            
            if (ftruncate(fd_, size) == -1) {
                close(fd_);
                return false;
            }
            
            mapped_memory_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                                 MAP_SHARED, fd_, 0);
            
            if (mapped_memory_ == MAP_FAILED) {
                close(fd_);
                return false;
            }
            
            file_size_ = size;
            return true;
        }
        
        void unmap() {
            if (mapped_memory_ && mapped_memory_ != MAP_FAILED) {
                munmap(mapped_memory_, file_size_);
                mapped_memory_ = nullptr;
            }
            if (fd_ != -1) {
                close(fd_);
                fd_ = -1;
            }
        }
        
        void* getMemory() { return mapped_memory_; }
        size_t getSize() { return file_size_; }
    };
};

// Smart pointer pool to reduce allocations
template<typename T>
class ObjectPool {
private:
    std::queue<std::unique_ptr<T>> pool_;
    std::mutex mutex_;
    size_t max_pool_size_ = 100;
    
public:
    std::shared_ptr<T> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!pool_.empty()) {
            std::unique_ptr<T> obj = std::move(pool_.front());
            pool_.pop();
            return std::shared_ptr<T>(obj.release(), [this](T* ptr) {
                release(std::unique_ptr<T>(ptr));
            });
        }
        
        return std::shared_ptr<T>(new T(), [this](T* ptr) {
            release(std::unique_ptr<T>(ptr));
        });
    }
    
private:
    void release(std::unique_ptr<T> obj) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (pool_.size() < max_pool_size_) {
            pool_.push(std::move(obj));
        }
    }
};

#endif // CHINET_MEMORYOPTIMIZER_H
