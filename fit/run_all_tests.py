"""
Run all tests for the FIT library.
This script will test all major components and report what's working.
"""

import sys
import os

def run_test_file(filename):
    """Run a test file and capture results."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {filename}")
    print('='*60)
    
    try:
        # Import and run the test
        if filename == "test_basic_tensor.py":
            import test_basic_tensor
        elif filename == "test_neural_network.py":
            import test_neural_network
        elif filename == "test_training.py":
            import test_training
        elif filename == "test_kaggle_features.py":
            import test_kaggle_features
        
        return True
    except Exception as e:
        print(f"❌ {filename} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests and summarize results."""
    
    print("FIT LIBRARY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # List of test files to run
    test_files = [
        "test_basic_tensor.py",
        "test_neural_network.py", 
        "test_training.py",
        "test_kaggle_features.py"
    ]
    
    results = {}
    
    # Run each test
    for test_file in test_files:
        success = run_test_file(test_file)
        results[test_file] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your FIT library is working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)