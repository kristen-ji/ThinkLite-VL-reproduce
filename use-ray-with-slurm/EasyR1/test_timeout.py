#!/usr/bin/env python3
"""
Simple timeout test script to prevent hanging in Ray worker creation
"""

import threading
import time
import signal

def timeout_handler(signum, frame):
    print("ERROR: Operation timed out!")
    raise TimeoutError("Operation timed out")

def long_running_operation():
    """Simulate a long-running operation that might hang"""
    print("Starting long-running operation...")
    time.sleep(10)  # Simulate hanging
    print("Operation completed")
    return "success"

def test_signal_timeout():
    """Test signal-based timeout"""
    print("Testing signal-based timeout...")
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        result = long_running_operation()
        signal.alarm(0)  # Cancel alarm
        print(f"Result: {result}")
        
    except TimeoutError as e:
        print(f"Caught timeout: {e}")
    except Exception as e:
        print(f"Other error: {e}")

def test_threading_timeout():
    """Test threading-based timeout"""
    print("Testing threading-based timeout...")
    
    result = {'success': False, 'data': None, 'error': None}
    
    def worker():
        try:
            data = long_running_operation()
            result['success'] = True
            result['data'] = data
        except Exception as e:
            result['error'] = e
    
    # Start worker in separate thread
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    
    # Wait for completion or timeout
    thread.join(5)  # 5 second timeout
    
    if thread.is_alive():
        print("ERROR: Operation timed out after 5 seconds")
        return None
    
    if result['success']:
        print(f"Result: {result['data']}")
        return result['data']
    else:
        print(f"Error: {result['error']}")
        return None

if __name__ == "__main__":
    print("Testing timeout mechanisms...")
    
    # Test 1: Signal-based timeout
    test_signal_timeout()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Threading-based timeout
    test_threading_timeout()
    
    print("\nTimeout tests completed!")
