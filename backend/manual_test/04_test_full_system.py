"""
TEST 4: Full AGCT System Integration
Location: manual_tests/04_test_full_system.py
Run: python manual_tests/04_test_full_system.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.agct_system import AGCTSystem
import json

class TestFullSystem:
    """Test full AGCT system"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.system = None
    
    def test(self, name, func):
        """Run a test"""
        try:
            print(f"\n   {name}...", end=" ")
            result = func()
            print("")
            self.passed += 1
            return result
        except Exception as e:
            print("")
            self.failed += 1
            self.errors.append((name, str(e)))
            return None
    
    def run_all(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("TEST 4: FULL AGCT SYSTEM")
        print("="*70)
        
        # Setup
        print("\n Initializing system...")
        self.test(
            "Initialize AGCT",
            self._init_system
        )
        
        if not self.system:
            print("\n Initialization failed")
            self._print_summary()
            return False
        
        print("\n Testing queries...")
        
        test_queries = [
            "What is backpropagation?",
            "Explain the chain rule",
            "How do neural networks work?",
            "What is gradient descent?",
        ]
        
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Query {i}/{ len(test_queries)}: {query[:40]}...")
            
            result = self.test(
                f"Process query {i}",
                lambda q=query: self.system.process_query(q)
            )
            
            if result:
                if result.get("status") == "success":
                    print(f"     Explanation: {len(result.get('explanation', ''))} chars")
                    print(f"     Path: {'  '.join(result['graph_path'])}")
                    results.append(result)
                else:
                    print(f"     {result.get('error')}")
        
        # Statistics
        print("\n Summary Statistics")
        successful = sum(1 for r in results if r.get("status") == "success")
        print(f"\n  Successful queries: {successful}/{len(results)}")
        
        if successful > 0:
            avg_explanation_len = sum(
                len(r.get('explanation', '')) for r in results
            ) / successful
            print(f"  Avg explanation length: {avg_explanation_len:.0f} chars")
        
        self._print_summary()
        return self.failed == 0
    
    def _init_system(self):
        """Initialize system"""
        self.system = AGCTSystem()
    
    def _print_summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {self.passed} ")
        print(f"Failed: {self.failed} ")
        
        if self.errors:
            print(f"\nErrors:")
            for name, error in self.errors[:5]:  # Show first 5
                print(f"   {name}")
                print(f"    {error[:60]}")
        
        total = self.passed + self.failed
        if total > 0:
            percentage = (self.passed / total) * 100
            print(f"\nSuccess Rate: {percentage:.1f}%")
        
        print("="*70 + "\n")

if __name__ == "__main__":
    tester = TestFullSystem()
    success = tester.run_all()
    sys.exit(0 if success else 1)



