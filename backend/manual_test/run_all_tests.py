"""Master test runner for the backend manual checks."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime


class TestRunner:
    """Run backend manual tests sequentially."""

    def __init__(self) -> None:
        self.results = {}
        self.start_time = datetime.now()

    def run_test(self, test_name: str, test_file: str) -> bool:
        print(f"\n{'=' * 70}")
        print(f"Running: {test_name}")
        print(f"File: {test_file}")
        print(f"{'=' * 70}")

        try:
            result = subprocess.run(
                [sys.executable, test_file],
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            )
            success = result.returncode == 0
            self.results[test_name] = {
                "status": "PASS" if success else "FAIL",
                "returncode": result.returncode,
            }
            return success
        except Exception as exc:
            self.results[test_name] = {
                "status": "ERROR",
                "error": str(exc),
            }
            return False

    def run_all(self) -> bool:
        print("\n" + "=" * 70)
        print("MASTER TEST SUITE")
        print("=" * 70)
        print(f"Started: {self.start_time}")

        tests = [
            ("TEST 1: RAG Module", "backend/manual_test/01_test_rag.py"),
            ("TEST 2: Retriever Agent", "backend/manual_test/02_test_retriever.py"),
            ("TEST 3: Graph Operations", "backend/manual_test/03_test_graph.py"),
            ("TEST 4: Full System", "backend/manual_test/04_test_full_system.py"),
        ]

        passed = 0
        failed = 0
        for test_name, test_file in tests:
            if self.run_test(test_name, test_file):
                passed += 1
            else:
                failed += 1

        return self._print_final_summary(passed, failed, len(tests))

    def _print_final_summary(self, passed: int, failed: int, total: int) -> bool:
        duration = (datetime.now() - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f} seconds")
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            print(f"  {test_name}: {result['status']}")
        print("=" * 70 + "\n")
        return failed == 0


if __name__ == "__main__":
    runner = TestRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)



