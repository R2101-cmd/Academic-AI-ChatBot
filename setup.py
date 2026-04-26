"""
Automated AGCT Setup Script
Location: setup.py (root directory)
Run: python setup.py
Handles installation, configuration, and initialization
"""

import subprocess
import sys
import os
from pathlib import Path

class AGCTSetup:
    """Setup and configuration for AGCT"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.success_count = 0
        self.error_count = 0
        self.errors = []
    
    def print_header(self):
        """Print setup header"""
        print("\n" + "="*70)
        print("AGCT SETUP WIZARD")
        print("="*70)
        print("Setting up Academic Graph-CoT Tutor")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Location: {self.root_dir}")
        print("="*70 + "\n")
    
    def step(self, step_num, description):
        """Print step header"""
        print(f"\n{'─'*70}")
        print(f"STEP {step_num}: {description}")
        print(f"{'─'*70}")
    
    def run_command(self, cmd, description, critical=False):
        """Run a shell command"""
        print(f"\n⏳ {description}...", end=" ", flush=True)
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.root_dir
            )
            
            if result.returncode == 0:
                print("✓")
                self.success_count += 1
                return True
            else:
                print("✗")
                error_msg = result.stderr or result.stdout
                self.error_count += 1
                self.errors.append((description, error_msg))
                
                if critical:
                    print(f"  Error: {error_msg[:100]}")
                    return False
                return False
        
        except subprocess.TimeoutExpired:
            print("✗ (Timeout)")
            self.error_count += 1
            self.errors.append((description, "Command timed out"))
            return False
        
        except Exception as e:
            print("✗")
            self.error_count += 1
            self.errors.append((description, str(e)))
            return False
    
    def create_directories(self):
        """Create required directories"""
        self.step(1, "Creating Directory Structure")
        
        directories = [
            "backend",
            "backend/rag",
            "backend/agents",
            "backend/graph",
            "backend/utils",
            "backend/core",
            "backend/data",
            "data",
            "tests",
            "manual_tests",
            "logs"
        ]
        
        print("\nCreating directories:")
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {directory}/")
        
        self.success_count += len(directories)
    
    def create_init_files(self):
        """Create __init__.py files"""
        self.step(2, "Creating Python Package Files")
        
        init_dirs = [
            "backend",
            "backend/rag",
            "backend/agents",
            "backend/graph",
            "backend/utils",
            "backend/core",
            "tests"
        ]
        
        print("\nCreating __init__.py files:")
        for directory in init_dirs:
            init_file = self.root_dir / directory / "__init__.py"
            init_file.touch(exist_ok=True)
            print(f"  ✓ {directory}/__init__.py")
        
        self.success_count += len(init_dirs)
    
    def upgrade_pip(self):
        """Upgrade pip"""
        self.step(3, "Upgrading Package Manager")
        
        self.run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip",
            critical=False
        )
    
    def clear_pip_cache(self):
        """Clear pip cache"""
        print("\n⏳ Clearing pip cache...", end=" ", flush=True)
        
        try:
            subprocess.run(
                f"{sys.executable} -m pip cache purge",
                shell=True,
                capture_output=True,
                timeout=30
            )
            print("✓")
            self.success_count += 1
        except Exception as e:
            print("⚠ (skipped)")
    
    def install_requirements(self):
        """Install requirements"""
        self.step(4, "Installing Dependencies")
        
        req_file = self.root_dir / "requirements.txt"
        
        if not req_file.exists():
            print(f"\n✗ requirements.txt not found at {req_file}")
            self.error_count += 1
            return False
        
        print(f"\nInstalling from {req_file}")
        
        success = self.run_command(
            f"{sys.executable} -m pip install --default-timeout=1000 -r requirements.txt",
            "Installing requirements",
            critical=True
        )
        
        if not success:
            print("\n⚠️ Installation failed. Trying minimal requirements...")
            
            minimal_packages = [
                "numpy==1.26.4",
                "scipy==1.13.0",
                "scikit-learn==1.4.2",
                "torch==2.0.1",
                "transformers==4.38.0",
                "sentence-transformers==5.2.0",
                "faiss-cpu==1.8.0",
                "networkx==3.3",
                "requests==2.31.0",
                "fastapi==0.109.0",
                "uvicorn==0.27.0",
                "pytest==7.4.4"
            ]
            
            for package in minimal_packages:
                self.run_command(
                    f"{sys.executable} -m pip install --default-timeout=1000 {package}",
                    f"Installing {package.split('==')[0]}",
                    critical=False
                )
        
        return True
    
    def download_nltk_data(self):
        """Download NLTK data"""
        self.step(5, "Downloading NLP Models")
        
        print("\n⏳ Downloading NLTK data...", end=" ", flush=True)
        
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("✓")
            self.success_count += 1
        except ImportError:
            print("⚠ (NLTK not installed)")
        except Exception as e:
            print(f"⚠ ({str(e)[:30]})")
    
    def verify_installation(self):
        """Verify installation"""
        self.step(6, "Verifying Installation")
        
        packages_to_check = [
            ("numpy", "NumPy"),
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("sentence_transformers", "Sentence Transformers"),
            ("faiss", "FAISS"),
            ("networkx", "NetworkX"),
            ("fastapi", "FastAPI"),
            ("requests", "Requests"),
            ("pytest", "pytest"),
        ]
        
        print("\nVerifying packages:")
        verified = 0
        
        for package, name in packages_to_check:
            try:
                __import__(package)
                print(f"  ✓ {name}")
                verified += 1
                self.success_count += 1
            except ImportError:
                print(f"  ✗ {name}")
                self.error_count += 1
        
        return verified == len(packages_to_check)
    
    def create_gitignore(self):
        """Create .gitignore file"""
        self.step(7, "Creating Git Configuration")
        
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data & Models
*.pkl
*.pth
*.bin
*.safetensors

# Logs
logs/
*.log

# Cache
.cache/
*.faiss
*.sqlite
*.db

# OS
.DS_Store
Thumbs.db
"""
        
        gitignore_file = self.root_dir / ".gitignore"
        gitignore_file.write_text(gitignore_content)
        print("\n✓ .gitignore created")
        self.success_count += 1
    
    def create_readme(self):
        """Create README.md"""
        self.step(8, "Creating Documentation")
        
        readme_content = """# Academic Graph-CoT Tutor (AGCT)

Advanced AI-powered academic tutoring system with Graph-Enhanced Chain-of-Thought reasoning.

## Quick Start

```bash
# 1. Verify installation
python verify_install.py

# 2. Run tests
python manual_tests/run_all_tests.py

# 3. Start the program
python main.py"""