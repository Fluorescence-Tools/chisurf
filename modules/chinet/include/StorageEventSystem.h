#ifndef CHINET_STORAGEEVENTSYSTEM_H
#define CHINET_STORAGEEVENTSYSTEM_H

#include <functional>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include "json.hpp"

using nlohmann::json;

enum class StorageEventType {
    DOCUMENT_CREATED,
    DOCUMENT_UPDATED,
    DOCUMENT_DELETED,
    TRANSACTION_COMMIT,
    TRANSACTION_ROLLBACK,
    STORAGE_PERSISTED,
    STORAGE_LOADED,
    MEMORY_THRESHOLD_EXCEEDED
};

struct StorageEvent {
    StorageEventType type;
    std::string key;
    json old_value;
    json new_value;
    uint64_t timestamp;
    std::string transaction_id;
    json metadata;
};

class StorageEventSystem {
private:
    using EventHandler = std::function<void(const StorageEvent&)>;
    using EventFilter = std::function<bool(const StorageEvent&)>;
    
    struct Subscription {
        std::string id;
        EventHandler handler;
        EventFilter filter;
        bool is_async;
    };
    
    std::unordered_map<StorageEventType, std::vector<Subscription>> subscriptions_;
    std::queue<std::pair<StorageEvent, EventHandler>> async_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    bool running_ = true;
    
    void processAsyncEvents() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !async_queue_.empty() || !running_; });
            
            while (!async_queue_.empty()) {
                auto [event, handler] = async_queue_.front();
                async_queue_.pop();
                lock.unlock();
                
                handler(event);
                
                lock.lock();
            }
        }
    }
    
public:
    StorageEventSystem() : worker_thread_(&StorageEventSystem::processAsyncEvents, this) {}
    
    ~StorageEventSystem() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        worker_thread_.join();
    }
    
    // Subscribe to events
    std::string subscribe(StorageEventType type, 
                          EventHandler handler,
                          EventFilter filter = nullptr,
                          bool is_async = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string id = generateSubscriptionId();
        Subscription sub{id, handler, filter, is_async};
        subscriptions_[type].push_back(sub);
        
        return id;
    }
    
    // Unsubscribe
    void unsubscribe(const std::string& subscription_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& [type, subs] : subscriptions_) {
            subs.erase(
                std::remove_if(subs.begin(), subs.end(),
                    [&](const Subscription& s) { return s.id == subscription_id; }),
                subs.end()
            );
        }
    }
    
    // Emit event
    void emit(const StorageEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (subscriptions_.find(event.type) == subscriptions_.end()) {
            return;
        }
        
        for (const auto& sub : subscriptions_[event.type]) {
            if (!sub.filter || sub.filter(event)) {
                if (sub.is_async) {
                    async_queue_.push({event, sub.handler});
                    cv_.notify_one();
                } else {
                    sub.handler(event);
                }
            }
        }
    }
    
    // Watch specific keys
    std::string watchKey(const std::string& key, EventHandler handler) {
        return subscribe(StorageEventType::DOCUMENT_UPDATED,
            handler,
            [key](const StorageEvent& e) { return e.key == key; });
    }
    
    // Watch pattern
    std::string watchPattern(const std::string& pattern, EventHandler handler) {
        std::regex regex_pattern(pattern);
        return subscribe(StorageEventType::DOCUMENT_UPDATED,
            handler,
            [regex_pattern](const StorageEvent& e) { 
                return std::regex_match(e.key, regex_pattern); 
            });
    }
    
private:
    std::string generateSubscriptionId() {
        static std::atomic<uint64_t> counter{0};
        return "sub_" + std::to_string(counter++);
    }
};

// Integration with InMemoryStorage
class ReactiveStorage {
private:
    std::shared_ptr<InMemoryStorage> storage_;
    std::shared_ptr<StorageEventSystem> event_system_;
    
public:
    ReactiveStorage(std::shared_ptr<InMemoryStorage> storage)
        : storage_(storage), event_system_(std::make_shared<StorageEventSystem>()) {}
    
    bool store(const std::string& key, const json& value) {
        auto old_value = storage_->retrieve(key);
        bool result = storage_->store(key, value);
        
        if (result) {
            StorageEvent event{
                old_value ? StorageEventType::DOCUMENT_UPDATED 
                         : StorageEventType::DOCUMENT_CREATED,
                key,
                old_value.value_or(json{}),
                value,
                std::chrono::system_clock::now().time_since_epoch().count(),
                "",
                json{}
            };
            event_system_->emit(event);
        }
        
        return result;
    }
    
    // Computed properties that auto-update
    void createComputedProperty(const std::string& target_key,
                                const std::vector<std::string>& source_keys,
                                std::function<json(const std::vector<json>&)> compute) {
        for (const auto& source : source_keys) {
            event_system_->watchKey(source, [=](const StorageEvent& e) {
                std::vector<json> values;
                for (const auto& key : source_keys) {
                    auto val = storage_->retrieve(key);
                    if (val) values.push_back(*val);
                }
                
                if (values.size() == source_keys.size()) {
                    json computed = compute(values);
                    storage_->store(target_key, computed);
                }
            });
        }
    }
};

#endif // CHINET_STORAGEEVENTSYSTEM_H
