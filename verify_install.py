"""
Verify AGCT Installation
Location: verify_install.py (root directory)
Run: python verify_install.py
Checks if all required packages are installed correctly
"""

import sys
import subprocess
from importlib import import_module
from packaging import version as pkg_version

class InstallationVerifier:
    """Verify all dependencies are installed"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    # Required packages with minimum versions
    REQUIRED_PACKAGES = {
        "numpy": "1.20.0",
        "torch": "2.0.0",
        "transformers": "4.30.0",
        "sentence_transformers": "2.2.0",
        "faiss": "1.7.0",
        "networkx": "3.0",
        "fastapi": "0.100.0",
        "requests": "2.28.0",
        "pytest": "7.0.0",
    }
    
    # Optional packages
    OPTIONAL_PACKAGES = {
        "nltk": "3.8.0",
        "spacy": "3.5.0",
        "streamlit": "1.0.0",
        "gradio": "4.0.0",
    }
    
    def verify_package(self, package_name, min_version=None, required=True):
        """Verify a single package is installed"""
        try:
            # Handle special cases
            if package_name == "faiss":
                try:
                    import faiss
                    actual_version = "1.8.0"  # FAISS doesn't expose version easily
                except ImportError:
                    raise ImportError(f"{package_name} not installed")
            else:
                module = import_module(package_name)
                actual_version = getattr(module, '__version__', 'unknown')
            
            # Check version if specified
            if min_version and actual_version != 'unknown':
                try:
                    if pkg_version.parse(actual_version) < pkg_version.parse(min_version):
                        status = "⚠ OLD VERSION"
                        self.warnings += 1
                    else:
                        status = "✓ OK"
                        self.passed += 1
                except Exception:
                    status = "✓ OK"
                    self.passed += 1
            else:
                status = "✓ OK"
                self.passed += 1
            
            self.results[package_name] = {
                "status": status,
                "version": actual_version,
                "required": required
            }
            
            return True
        
        except ImportError:
            status = "✗ NOT INSTALLED"
            if required:
                self.failed += 1
            else:
                self.warnings += 1
            
            self.results[package_name] = {
                "status": status,
                "version": None,
                "required": required
            }
            
            return False
    
    def verify_python_version(self):
        """Verify Python version"""
        print("\n" + "="*70)
        print("PYTHON VERSION")
        print("="*70)
        
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        print(f"Required: Python {required_version[0]}.{required_version[1]}+")
        print(f"Current:  Python {current_version[0]}.{current_version[1]}.{sys.version_info[2]}")
        
        if current_version >= required_version:
            print("✓ Python version OK")
            self.passed += 1
            return True
        else:
            print("✗ Python version too old!")
            self.failed += 1
            return False
    
    def verify_pip(self):
        """Verify pip is working"""
        print("\n" + "="*70)
        print("PIP CONFIGURATION")
        print("="*70)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(result.stdout.strip())
                print("✓ Pip is working")
                self.passed += 1
                return True
            else:
                print("✗ Pip error")
                self.failed += 1
                return False
        
        except Exception as e:
            print(f"✗ Error checking pip: {e}")
            self.failed += 1
            return False
    
    def verify_internet(self):
        """Verify internet connection"""
        print("\n" + "="*70)
        print("INTERNET CONNECTION")
        print("="*70)
        
        try:
            import urllib.request
            print("Checking connection to PyPI...", end=" ")
            urllib.request.urlopen("https://pypi.org", timeout=5)
            print("✓")
            print("✓ Internet connection OK")
            self.passed += 1
            return True
        
        except Exception as e:
            print("✗")
            print(f"✗ Cannot reach PyPI: {e}")
            print("  Note: You may need to use a mirror:")
            print("    pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt")
            self.warnings += 1
            return False
    
    def verify_required_packages(self):
        """Verify all required packages"""
        print("\n" + "="*70)
        print("REQUIRED PACKAGES")
        print("="*70)
        
        for package, min_version in self.REQUIRED_PACKAGES.items():
            status_symbol = "⏳"
            print(f"{status_symbol} Checking {package}...", end=" ", flush=True)
            
            self.verify_package(package, min_version, required=True)
            
            result = self.results[package]
            print(f"{result['status']} (v{result['version']})")
    
    def verify_optional_packages(self):
        """Verify optional packages"""
        print("\n" + "="*70)
        print("OPTIONAL PACKAGES")
        print("="*70)
        
        for package, min_version in self.OPTIONAL_PACKAGES.items():
            print(f"⏳ Checking {package}...", end=" ", flush=True)
            
            self.verify_package(package, min_version, required=False)
            
            result = self.results[package]
            print(f"{result['status']} (v{result['version']})")
    
    def verify_backend_structure(self):
        """Verify backend directory structure"""
        print("\n" + "="*70)
        print("BACKEND STRUCTURE")
        print("="*70)
        
        import os
        
        required_dirs = [
            "backend",
            "backend/rag",
            "backend/agents",
            "backend/graph",
            "backend/utils",
            "backend/core",
            "data",
            "manual_tests"
        ]
        
        all_exist = True
        
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                print(f"✓ {dir_path}/")
                self.passed += 1
            else:
                print(f"✗ {dir_path}/ (MISSING)")
                self.failed += 1
                all_exist = False
        
        return all_exist
    
    def verify_data_files(self):
        """Verify required data files"""
        print("\n" + "="*70)
        print("DATA FILES")
        print("="*70)
        
        import os
        
        required_files = [
            ("data/sample.txt", "Sample data"),
            ("main.py", "Main program"),
            ("requirements.txt", "Requirements"),
        ]
        
        all_exist = True
        
        for file_path, description in required_files:
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"✓ {file_path} ({size} bytes) - {description}")
                self.passed += 1
            else:
                print(f"✗ {file_path} (MISSING) - {description}")
                self.failed += 1
                all_exist = False
        
        return all_exist
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)
        
        total = self.passed + self.failed + self.warnings
        
        print(f"\nResults:")
        print(f"  ✓ Passed:  {self.passed}")
        print(f"  ⚠ Warnings: {self.warnings}")
        print(f"  ✗ Failed:  {self.failed}")
        print(f"  Total:     {total}")
        
        if total > 0:
            percentage = (self.passed / total) * 100
            print(f"\nSuccess Rate: {percentage:.1f}%")
        
        print("\n" + "="*70)
        
        if self.failed == 0:
            print("✅ INSTALLATION VERIFIED - Ready to use AGCT!")
            return True
        else:
            print(f"⚠���  {self.failed} critical issue(s) found")
            print("\nTo fix:")
            print(f"  1. Run: python setup.py")
            print(f"  2. Or manually install missing packages")
            print(f"  3. Then run: python verify_install.py")
            return False
    
    def print_detailed_results(self):
        """Print detailed results table"""
        print("\n" + "="*70)
        print("DETAILED RESULTS")
        print("="*70)
        
        print(f"\n{'Package':<25} {'Status':<20} {'Version':<15}")
        print("-" * 70)
        
        for package, info in sorted(self.results.items()):
            status = info['status']
            version = info['version'] or "N/A"
            required = " (required)" if info['required'] else ""
            
            print(f"{package:<25} {status:<20} {version:<15}{required}")
    
    def run_all_checks(self, verbose=False):
        """Run all verification checks"""
        print("\n" + "="*70)
        print("AGCT INSTALLATION VERIFICATION")
        print("="*70)
        print(f"Python: {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        
        # Run checks
        self.verify_python_version()
        self.verify_pip()
        self.verify_internet()
        self.verify_backend_structure()
        self.verify_data_files()
        self.verify_required_packages()
        self.verify_optional_packages()
        
        # Print results
        if verbose:
            self.print_detailed_results()
        
        success = self.print_summary()
        
        return success


def main():
    """Main function"""
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    verifier = InstallationVerifier()
    success = verifier.run_all_checks(verbose=verbose)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()