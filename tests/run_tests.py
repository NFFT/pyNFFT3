import subprocess
import sys
import os


# Script to run all tests and return any failures
def run_tests():
    test_files = [
        "tests/NFFT_test.py",
        "tests/NFCT_test.py",
        "tests/NFST_test.py",
        "tests/fastsum_test.py",
    ]

    for test_file in test_files:
        print("Testing ", test_file, "...")
        result = subprocess.run(
            [sys.executable, test_file], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Test failed for ", test_file)
            print(result.stderr)
            sys.exit(result.returncode)


if __name__ == "__main__":
    run_tests()
