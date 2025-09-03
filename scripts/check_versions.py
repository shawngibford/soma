#!/usr/bin/env python3
"""Script to check versions of quantum drug discovery package and dependencies."""

import sys
import re

def get_package_version():
    """Extract version from setup.py"""
    try:
        with open('setup.py', 'r') as f:
            content = f.read()
            version_match = re.search(r"version=['\"]([^'\"]+)['\"]", content)
            if version_match:
                return version_match.group(1)
    except Exception as e:
        return f"Error reading setup.py: {e}"
    return "Version not found"

def check_dependencies():
    """Check versions of key dependencies"""
    dependencies = {}
    
    # Check Python
    dependencies['Python'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    dependencies['Python Executable'] = sys.executable
    
    # Check key packages
    packages_to_check = [
        'numpy', 'pennylane', 'torch', 'rdkit', 'pandas', 'scipy'
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                dependencies[package] = module.__version__
            elif hasattr(module, 'version'):
                dependencies[package] = module.version
            else:
                dependencies[package] = "Version not available"
        except ImportError:
            dependencies[package] = "Not installed"
        except Exception as e:
            dependencies[package] = f"Error: {e}"
    
    return dependencies

def main():
    print("=" * 60)
    print("QUANTUM DRUG DISCOVERY - VERSION CHECK")
    print("=" * 60)
    
    # Package version
    print(f"\nPackage Version: {get_package_version()}")
    
    # Dependencies
    print("\nDependencies:")
    print("-" * 40)
    deps = check_dependencies()
    for name, version in deps.items():
        print(f"{name:20}: {version}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 