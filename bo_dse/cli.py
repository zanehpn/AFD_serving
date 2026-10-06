#!/usr/bin/env python3
"""Run the model-assisted BO CLI independently of the legacy static_dse package."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    implementation = Path(__file__).resolve().parent / "scripts" / "afd"
    sys.path.insert(0, str(implementation))
    runpy.run_path(str(implementation / "static_dse_cli.py"), run_name="__main__")
