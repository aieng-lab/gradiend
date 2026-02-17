#!/usr/bin/env python
"""Check all dependencies by importing all modules."""

import sys
import importlib

# Add current directory to path
sys.path.insert(0, '.')

missing = []
errors = []

def try_import(module_name, package_name=None):
    """Try to import a module and report errors."""
    try:
        if package_name:
            mod = importlib.import_module(module_name, package=package_name)
        else:
            mod = importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        missing.append((module_name, str(e)))
        print(f"✗ {module_name}: {e}")
        return False
    except Exception as e:
        errors.append((module_name, str(e)))
        print(f"⚠ {module_name}: {e}")
        return False

print("Checking core dependencies...")
print("=" * 60)

# Standard library - should always work
print("\n1. Standard library:")
try_import("json")
try_import("os")
try_import("itertools")
try_import("collections")
try_import("typing")

# Core dependencies
print("\n2. Core dependencies:")
try_import("torch")
try_import("transformers")
try_import("numpy")
try_import("pandas")
try_import("sklearn")
try_import("tqdm")
try_import("datasets")
try_import("scipy")
try_import("matplotlib")
try_import("seaborn")
try_import("inflect")
try_import("sentencepiece")

# GRADIEND modules
print("\n3. GRADIEND modules:")
try_import("gradiend.logging_config")
try_import("gradiend.model")
try_import("gradiend.model.model")
try_import("gradiend.model.text_model")
try_import("gradiend.model.gradient_creator")
try_import("gradiend.training")
try_import("gradiend.training.training_args")
try_import("gradiend.training.factory")
try_import("gradiend.training.training")


print("\n" + "=" * 60)
if missing:
    print(f"\n❌ Missing {len(missing)} dependencies:")
    for mod, err in missing:
        print(f"  - {mod}: {err}")
    
    print("\nTo install missing dependencies, run:")
    print("  pip install " + " ".join([mod.split(".")[0] for mod, _ in missing if "." in mod]))
else:
    print("\n✅ All dependencies are available!")

if errors:
    print(f"\n⚠ {len(errors)} import errors (not missing deps):")
    for mod, err in errors:
        print(f"  - {mod}: {err}")

sys.exit(0 if not missing else 1)
