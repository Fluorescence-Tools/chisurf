#!/usr/bin/env python3
"""
Simplified ChiSurf Foundation Benchmark
Tests core functionality without complex dependencies
"""

import time
import sys
import os

# Ensure we're using the right environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chisurf
from chisurf.controllers.action_controller import ActionController

def benchmark_action_controller():
    """Benchmark the action controller performance"""
    print("=== Simplified ChiSurf Foundation Benchmark ===")
    print(f"Python version: {sys.version}")
    print(f"ChiSurf version: {getattr(chisurf, '__version__', 'unknown')}")
    print()
    
    controller = ActionController()
    results = {}
    
    # Test action routing performance (without execution)
    test_actions = [
        "parameter.value",
        "fit.update", 
        "model.add_component",
        "model.remove_component",
        "parameter.scan",
        "fit.range_set",
        "model.update",
        "project.save",
        "project.load",
        "dataset.add"
    ]
    
    print("--- Action Controller Routing Performance ---")
    for action_name in test_actions:
        try:
            # Test routing speed (without actual execution)
            start_time = time.time()
            
            # Just test the routing, not the actual execution
            handler_exists = action_name.replace(".", "_") in controller._handlers
            
            end_time = time.time()
            routing_time = (end_time - start_time) * 1000  # Convert to ms
            
            results[action_name] = {
                'routing_time_ms': routing_time,
                'handler_exists': handler_exists
            }
            
            status = "OK" if handler_exists else "MISSING"
            print(f"{action_name:25} | {routing_time:6.3f} ms | {status}")
            
        except Exception as e:
            print(f"{action_name:25} | ERROR: {str(e)}")
            results[action_name] = {'error': str(e)}
    
    print()
    
    # Test action registry performance
    print("--- Action Registry Performance ---")
    try:
        registry = getattr(chisurf, 'action_registry', None)
        if registry:
            start_time = time.time()
            
            # Test registry operations
            action_list = registry.list_actions()
            catalog = registry.catalog()
            
            end_time = time.time()
            registry_time = (end_time - start_time) * 1000
            
            results['registry'] = {
                'actions_count': len(action_list),
                'catalog_size': len(catalog),
                'registry_time_ms': registry_time
            }
            
            print(f"Registry operations      | {registry_time:6.3f} ms | {len(action_list)} actions | {len(catalog)} catalog entries")
        else:
            print("Registry operations      | NOT AVAILABLE")
    except Exception as e:
        print(f"Registry operations      | ERROR: {str(e)}")
        results['registry'] = {'error': str(e)}
    
    print()
    
    # Test history system (if available)
    print("--- History System Status ---")
    try:
        history = getattr(chisurf, 'history', None)
        if history:
            # Check history attributes
            history_attrs = []
            if hasattr(history, '_events'):
                history_attrs.append(f"events: {len(history._events)}")
            if hasattr(history, 'cursor'):
                history_attrs.append(f"cursor: {history.cursor}")
            if hasattr(history, 'checkpoints'):
                history_attrs.append(f"checkpoints: {len(history.checkpoints)}")
            
            print(f"History system           | ACTIVE | {', '.join(history_attrs)}")
            results['history'] = {'status': 'active', 'attrs': history_attrs}
        else:
            print("History system           | NOT INITIALIZED")
            results['history'] = {'status': 'not_initialized'}
    except Exception as e:
        print(f"History system           | ERROR: {str(e)}")
        results['history'] = {'error': str(e)}
    
    print()
    print("=== Benchmark Complete ===")
    
    # Save results
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_simple_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'python_version': sys.version,
                'chisurf_version': getattr(chisurf, '__version__', 'unknown')
            },
            'results': results
        }, f, indent=2)
    
    print(f"Results saved to: {filename}")
    
    # Summary
    print()
    print("=== Summary ===")
    active_actions = sum(1 for v in results.values() if isinstance(v, dict) and v.get('handler_exists'))
    total_actions = len([v for v in results.values() if isinstance(v, dict) and 'handler_exists' in v])
    
    print(f"Action handlers available: {active_actions}/{total_actions}")
    
    if 'registry' in results and 'actions_count' in results['registry']:
        print(f"Registered actions: {results['registry']['actions_count']}")
    
    if 'history' in results:
        print(f"History status: {results['history'].get('status', 'unknown')}")
    
    return results

if __name__ == "__main__":
    benchmark_action_controller()