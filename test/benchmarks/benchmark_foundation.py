#!/usr/bin/env python3
"""
Performance Benchmarking Suite for ChiSurf Foundation
Measures current performance baselines for history and action operations
"""

import time
import tracemalloc
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chisurf
from chisurf.controllers.action_controller import ActionController
from chisurf.runtime.actions import record_action


def benchmark_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function call"""
    tracemalloc.start()
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Take final snapshot
    snapshot2 = tracemalloc.take_snapshot()
    
    tracemalloc.stop()
    
    # Calculate memory difference
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_allocated = sum(stat.size_diff for stat in top_stats)
    
    return result, total_allocated


def benchmark_execution_time(func, *args, **kwargs):
    """Measure execution time of a function call"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def run_action_benchmark(action_name, payload, iterations=100):
    """Benchmark action execution performance"""
    controller = ActionController()
    
    # Warm-up
    for _ in range(5):
        controller.execute(name=action_name, payload=payload)
    
    # Benchmark execution time
    total_time = 0
    for _ in range(iterations):
        _, exec_time = benchmark_execution_time(
            controller.execute, 
            name=action_name, 
            payload=payload
        )
        total_time += exec_time
    
    avg_time = total_time / iterations
    
    # Benchmark memory usage
    try:
        _, memory_used = benchmark_memory_usage(
            controller.execute, 
            name=action_name, 
            payload=payload
        )
    except Exception as e:
        memory_used = 0
        print(f"Memory benchmark failed for {action_name}: {e}")
    
    return {
        'action': action_name,
        'iterations': iterations,
        'avg_execution_time_ms': avg_time * 1000,
        'memory_used_bytes': memory_used,
        'executions_per_second': 1000 / (avg_time * 1000) if avg_time > 0 else 0
    }


def run_history_benchmark(operations=1000):
    """Benchmark history recording performance"""
    # Clear existing history
    if hasattr(chisurf, 'history') and chisurf.history:
        initial_count = len(chisurf.history.events)
    else:
        initial_count = 0
    
    # Benchmark recording performance
    start_time = time.time()
    
    for i in range(operations):
        record_action(
            action_type="test_operation",
            summary=f"Test operation {i}",
            payload={"iteration": i, "data": f"test_data_{i}"},
            source_uid=f"test_source_{i}"
        )
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    operations_per_second = operations / total_time if total_time > 0 else 0
    avg_time_per_operation_ms = (total_time / operations) * 1000 if operations > 0 else 0
    
    # Check final history state
    if hasattr(chisurf, 'history') and chisurf.history:
        # Use the correct attribute name for history events
        if hasattr(chisurf.history, 'events'):
            final_count = len(chisurf.history.events)
        elif hasattr(chisurf.history, '_events'):
            final_count = len(chisurf.history._events)
        else:
            final_count = 0
        operations_recorded = final_count - initial_count
    else:
        operations_recorded = 0
    
    return {
        'operations_attempted': operations,
        'operations_recorded': operations_recorded,
        'total_time_seconds': total_time,
        'operations_per_second': operations_per_second,
        'avg_time_per_operation_ms': avg_time_per_operation_ms,
        'success_rate': (operations_recorded / operations) * 100 if operations > 0 else 0
    }


def setup_test_environment():
    """Set up minimal test environment for benchmarking"""
    # Initialize history if not present
    if not hasattr(chisurf, 'history'):
        from chisurf.history import OperationHistory
        chisurf.history = OperationHistory()
    
    # Add a minimal fit for testing
    if not hasattr(chisurf, 'fits'):
        chisurf.fits = []
    
    if len(chisurf.fits) == 0:
        # Create a minimal mock fit object
        class MockFit:
            def __init__(self):
                self.name = "test_fit"
                self.unique_identifier = "test_fit_001"
                self.model = MockModel()
        
        class MockModel:
            def __init__(self):
                self.parameters_all_dict = {}
                # Add a test parameter
                class MockParameter:
                    def __init__(self):
                        self.name = "tau1"
                        self.value = 2.5
                        self.unique_identifier = "param_tau1_001"
                    
                    def scan(self, fit, scan_range, n_steps):
                        # Mock scan method
                        pass
                
                self.parameters_all_dict["tau1"] = MockParameter()
            
            def update(self):
                pass
        
        chisurf.fits.append(MockFit())
    
    # Initialize action controller if not present
    if not hasattr(chisurf, 'action_controller'):
        chisurf.action_controller = ActionController()


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmarking"""
    print("=== ChiSurf Foundation Performance Benchmark ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"ChiSurf version: {getattr(chisurf, '__version__', 'unknown')}")
    print()
    
    # Set up test environment
    setup_test_environment()
    
    results = {}
    
    # Test common actions
    common_actions = [
        ("parameter.value", {"parameter_name": "tau1", "value": 2.5, "fit_index": 0}),
        ("fit.update", {"fit_index": 0}),
        ("model.add_component", {"component_name": "lifetime"}),
        ("model.remove_component", {"component_name": "lifetime"}),
        ("parameter.scan", {"parameter_name": "tau1", "fit_index": 0, "scan_range": (1.0, 5.0), "n_steps": 10}),
    ]
    
    print("--- Action Performance Benchmarks ---")
    for action_name, payload in common_actions:
        try:
            benchmark_result = run_action_benchmark(action_name, payload)
            results[f"action_{action_name}"] = benchmark_result
            print(f"{action_name:25} | {benchmark_result['avg_execution_time_ms']:6.2f} ms | {benchmark_result['memory_used_bytes']:8} bytes | {benchmark_result['executions_per_second']:6.1f} ops/sec")
        except Exception as e:
            print(f"{action_name:25} | ERROR: {str(e)}")
            results[f"action_{action_name}"] = {'error': str(e)}
    
    print()
    print("--- History Performance Benchmarks ---")
    
    # Test history recording performance
    for ops in [100, 500, 1000]:
        try:
            history_result = run_history_benchmark(ops)
            results[f"history_{ops}_ops"] = history_result
            print(f"{ops:4} operations | {history_result['total_time_seconds']:6.2f} s | {history_result['operations_per_second']:8.1f} ops/sec | {history_result['avg_time_per_operation_ms']:6.2f} ms/op | {history_result['success_rate']:5.1f}% success")
        except Exception as e:
            print(f"{ops:4} operations | ERROR: {str(e)}")
            results[f"history_{ops}_ops"] = {'error': str(e)}
    
    print()
    print("=== Benchmark Complete ===")
    
    # Save results
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'python_version': sys.version,
                'platform': sys.platform,
                'chisurf_version': getattr(chisurf, '__version__', 'unknown')
            },
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return results


if __name__ == "__main__":
    # Initialize ChiSurf if needed
    if not hasattr(chisurf, 'history'):
        from chisurf.history import OperationHistory
        chisurf.history = OperationHistory()
    
    run_comprehensive_benchmark()