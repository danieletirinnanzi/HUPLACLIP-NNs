import os
import unittest
import sys


def run_all_tests():
    # Define the directory where test files are located
    test_dir = os.path.join(os.path.dirname(__file__))

    # Discover and run all tests in the `tests` directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=test_dir, pattern="test_*.py")
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests and capture the result
    result = test_runner.run(test_suite)

    # Stop the main script if tests fail
    if not result.wasSuccessful():
        print("Some tests failed. Aborting experiment.")
        sys.exit(1)  # Exit with error code if tests failed
    else:
        print("All tests passed. Proceeding with the experiment.")
