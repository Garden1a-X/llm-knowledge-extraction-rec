#!/usr/bin/env python3
"""
Test RecBole installation and import.
"""

import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

print("Trying to import recbole...")
try:
    import recbole
    print("✓ RecBole imported successfully!")
    print(f"  Version: {recbole.__version__}")
    print(f"  Location: {recbole.__file__}")
    print()

    print("Trying to import recbole.quick_start...")
    from recbole.quick_start import run_recbole
    print("✓ recbole.quick_start imported successfully!")

    print("Trying to import recbole.config...")
    from recbole.config import Config
    print("✓ recbole.config imported successfully!")

    print()
    print("="*60)
    print("RecBole is working correctly!")
    print("="*60)

except ImportError as e:
    print("✗ Import failed!")
    print(f"  Error: {e}")
    print()
    print("Full traceback:")
    import traceback
    traceback.print_exc()

except Exception as e:
    print("✗ Unexpected error!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
