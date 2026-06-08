#ifndef CHINET_PERFORMANCEMONITOR_H
#define CHINET_PERFORMANCEMONITOR_H

#include <chrono>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <queue>
#include "json.hpp"

using nlohmann::json;

class PerformanceMonitor {
private:
    struct OperationStats {
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> total_time_ns{0};
        std::atomic<uint64_t> min_time_ns{UINT64_MAX};
        std::atomic<uint64_t> max_time_ns{0};
        std::atomic<uint64_t> errors{0};
        
        // Percentile tracking
        std::priority_queue<uint64_t> recent_times;
        std::mutex queue_mutex;
        static constexpr size_t MAX_SAMPLES = 1000;
        
        void recordTime(uint64_t time_ns) {
            count++;
            total_time_ns += time_ns;
            
            uint64_t current_min = min_time_ns.load();
            while (time_ns < current_min && 
                   !min_time_ns.compare_exchange_weak(current_min, time_ns));
            
            uint64_t current_max = max_time_ns.load();
            while (time_ns > current_max && 
                   !max_time_ns.compare_exchange_weak(current_max, time_ns));
            
            // Track recent times for percentile calculation
            std::lock_guard<std::mutex> lock(queue_mutex);
            recent_times.push(time_ns);
            if (recent_times.size() > MAX_SAMPLES) {
                // Keep only recent samples
                std::priority_queue<uint64_t> new_queue;
                size_t keep = MAX_SAMPLES / 2;
                for (size_t i = 0; i < keep && !recent_times.empty(); i++) {
                    new_queue.push(recent_times.top());
                    recent_times.pop();
                }
                recent_times = std::move(new_queue);
            }
        }
        
        double getAverage() const {
            if (count == 0) return 0;
            return static_cast<double>(total_time_ns) / count;
        }
        
        double getPercentile(double p) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (recent_times.empty()) return 0;
            
            std::vector<uint64_t> times;
            auto temp_queue = recent_times;
            while (!temp_queue.empty()) {
                times.push_back(temp_queue.top());
                temp_queue.pop();
            }
            
            std::sort(times.begin(), times.end());
            size_t index = static_cast<size_t>(times.size() * p / 100.0);
            return times[std::min(index, times.size() - 1)];
        }
    };
    
    std::unordered_map<std::string, OperationStats> operation_stats_;
    std::mutex stats_mutex_;
    
    // Resource tracking
    std::atomic<size_t> memory_usage_{0};
    std::atomic<size_t> peak_memory_{0};
    std::atomic<size_t> active_connections_{0};
    std::atomic<size_t> peak_connections_{0};
    
    // Throughput tracking
    struct ThroughputWindow {
        std::atomic<uint64_t> operations{0};
        std::atomic<uint64_t> bytes_read{0};
        std::atomic<uint64_t> bytes_written{0};
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
    };
    
    ThroughputWindow current_window_;
    std::vector<ThroughputWindow> historical_windows_;
    std::chrono::seconds window_duration_{60};  // 1-minute windows
    
public:
    // RAII timer for automatic measurement
    class Timer {
    private:
        PerformanceMonitor* monitor_;
        std::string operation_;
        std::chrono::high_resolution_clock::time_point start_;
        bool completed_ = false;
        
    public:
        Timer(PerformanceMonitor* monitor, const std::string& operation)
            : monitor_(monitor), operation_(operation),
              start_(std::chrono::high_resolution_clock::now()) {}
        
        ~Timer() {
            if (!completed_) {
                complete();
            }
        }
        
        void complete() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start_).count();
            monitor_->recordOperation(operation_, duration, true);
            completed_ = true;
        }
        
        void error() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start_).count();
            monitor_->recordOperation(operation_, duration, false);
            completed_ = true;
        }
    };
    
    // Record operation performance
    void recordOperation(const std::string& operation, uint64_t time_ns, bool success) {
        operation_stats_[operation].recordTime(time_ns);
        if (!success) {
            operation_stats_[operation].errors++;
        }
        
        current_window_.operations++;
        checkWindowRotation();
    }
    
    // Record data throughput
    void recordDataTransfer(size_t bytes, bool is_read) {
        if (is_read) {
            current_window_.bytes_read += bytes;
        } else {
            current_window_.bytes_written += bytes;
        }
    }
    
    // Resource tracking
    void updateMemoryUsage(size_t bytes) {
        memory_usage_ = bytes;
        size_t current_peak = peak_memory_.load();
        while (bytes > current_peak && 
               !peak_memory_.compare_exchange_weak(current_peak, bytes));
    }
    
    void updateConnectionCount(int delta) {
        if (delta > 0) {
            active_connections_ += delta;
            size_t current = active_connections_.load();
            size_t current_peak = peak_connections_.load();
            while (current > current_peak && 
                   !peak_connections_.compare_exchange_weak(current_peak, current));
        } else {
            active_connections_ -= (-delta);
        }
    }
    
    // Get comprehensive statistics
    json getStatistics() {
        json stats;
        
        // Operation statistics
        json op_stats;
        for (const auto& [op, stat] : operation_stats_) {
            json op_json;
            op_json["count"] = stat.count.load();
            op_json["avg_time_ms"] = stat.getAverage() / 1'000'000.0;
            op_json["min_time_ms"] = stat.min_time_ns.load() / 1'000'000.0;
            op_json["max_time_ms"] = stat.max_time_ns.load() / 1'000'000.0;
            op_json["p50_ms"] = stat.getPercentile(50) / 1'000'000.0;
            op_json["p95_ms"] = stat.getPercentile(95) / 1'000'000.0;
            op_json["p99_ms"] = stat.getPercentile(99) / 1'000'000.0;
            op_json["errors"] = stat.errors.load();
            op_json["error_rate"] = stat.count > 0 ? 
                static_cast<double>(stat.errors) / stat.count : 0;
            
            op_stats[op] = op_json;
        }
        stats["operations"] = op_stats;
        
        // Resource statistics
        json resources;
        resources["memory_usage_mb"] = memory_usage_.load() / (1024.0 * 1024.0);
        resources["peak_memory_mb"] = peak_memory_.load() / (1024.0 * 1024.0);
        resources["active_connections"] = active_connections_.load();
        resources["peak_connections"] = peak_connections_.load();
        stats["resources"] = resources;
        
        // Throughput statistics
        json throughput;
        auto now = std::chrono::steady_clock::now();
        auto window_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - current_window_.start_time).count();
        
        if (window_elapsed > 0) {
            throughput["ops_per_second"] = 
                static_cast<double>(current_window_.operations) / window_elapsed;
            throughput["read_mb_per_second"] = 
                (current_window_.bytes_read.load() / (1024.0 * 1024.0)) / window_elapsed;
            throughput["write_mb_per_second"] = 
                (current_window_.bytes_written.load() / (1024.0 * 1024.0)) / window_elapsed;
        }
        
        // Historical throughput
        json historical = json::array();
        for (const auto& window : historical_windows_) {
            json window_json;
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                window.end_time - window.start_time).count();
            
            window_json["start_time"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                window.start_time.time_since_epoch()).count();
            window_json["duration_seconds"] = duration;
            window_json["operations"] = window.operations.load();
            window_json["bytes_read"] = window.bytes_read.load();
            window_json["bytes_written"] = window.bytes_written.load();
            
            if (duration > 0) {
                window_json["ops_per_second"] = 
                    static_cast<double>(window.operations) / duration;
            }
            
            historical.push_back(window_json);
        }
        throughput["historical"] = historical;
        stats["throughput"] = throughput;
        
        return stats;
    }
    
    // Health check
    json getHealthStatus() {
        json health;
        
        // Calculate health score (0-100)
        int score = 100;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        
        // Check error rates
        for (const auto& [op, stat] : operation_stats_) {
            if (stat.count > 0) {
                double error_rate = static_cast<double>(stat.errors) / stat.count;
                if (error_rate > 0.1) {
                    errors.push_back(op + " has high error rate: " + 
                                   std::to_string(error_rate * 100) + "%");
                    score -= 20;
                } else if (error_rate > 0.01) {
                    warnings.push_back(op + " has elevated error rate: " + 
                                     std::to_string(error_rate * 100) + "%");
                    score -= 10;
                }
            }
        }
        
        // Check response times
        if (operation_stats_.count("read") > 0) {
            double avg_read = operation_stats_["read"].getAverage() / 1'000'000.0;
            if (avg_read > 100) {  // > 100ms
                warnings.push_back("Slow read performance: " + 
                                 std::to_string(avg_read) + "ms average");
                score -= 10;
            }
        }
        
        // Check memory usage
        double memory_usage_mb = memory_usage_.load() / (1024.0 * 1024.0);
        if (memory_usage_mb > 1024) {  // > 1GB
            warnings.push_back("High memory usage: " + 
                             std::to_string(memory_usage_mb) + "MB");
            score -= 5;
        }
        
        health["score"] = std::max(0, score);
        health["status"] = score >= 80 ? "healthy" : (score >= 50 ? "degraded" : "unhealthy");
        health["warnings"] = warnings;
        health["errors"] = errors;
        
        return health;
    }
    
    // Reset statistics
    void reset() {
        operation_stats_.clear();
        historical_windows_.clear();
        current_window_ = ThroughputWindow{};
        current_window_.start_time = std::chrono::steady_clock::now();
    }
    
    // Export metrics in Prometheus format
    std::string exportPrometheusMetrics() {
        std::stringstream ss;
        
        // Operation metrics
        for (const auto& [op, stat] : operation_stats_) {
            std::string op_safe = op;
            std::replace(op_safe.begin(), op_safe.end(), '.', '_');
            
            ss << "chinet_operation_count{operation=\"" << op_safe << "\"} " 
               << stat.count.load() << "\n";
            ss << "chinet_operation_duration_ms{operation=\"" << op_safe 
               << "\",quantile=\"0.5\"} " << stat.getPercentile(50) / 1'000'000.0 << "\n";
            ss << "chinet_operation_duration_ms{operation=\"" << op_safe 
               << "\",quantile=\"0.95\"} " << stat.getPercentile(95) / 1'000'000.0 << "\n";
            ss << "chinet_operation_duration_ms{operation=\"" << op_safe 
               << "\",quantile=\"0.99\"} " << stat.getPercentile(99) / 1'000'000.0 << "\n";
            ss << "chinet_operation_errors{operation=\"" << op_safe << "\"} " 
               << stat.errors.load() << "\n";
        }
        
        // Resource metrics
        ss << "chinet_memory_usage_bytes " << memory_usage_.load() << "\n";
        ss << "chinet_peak_memory_bytes " << peak_memory_.load() << "\n";
        ss << "chinet_active_connections " << active_connections_.load() << "\n";
        
        return ss.str();
    }
    
private:
    void checkWindowRotation() {
        auto now = std::chrono::steady_clock::now();
        if (now - current_window_.start_time > window_duration_) {
            current_window_.end_time = now;
            historical_windows_.push_back(current_window_);
            
            // Keep only last 60 windows (1 hour of 1-minute windows)
            if (historical_windows_.size() > 60) {
                historical_windows_.erase(historical_windows_.begin());
            }
            
            current_window_ = ThroughputWindow{};
            current_window_.start_time = now;
        }
    }
};

#endif // CHINET_PERFORMANCEMONITOR_H
