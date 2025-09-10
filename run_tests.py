#!/usr/bin/env python
"""Run tests for OpenEgo package."""

import sys
import subprocess

def main():
    """Run pytest with appropriate arguments."""
    cmd = [sys.executable, "-m", "pytest", "-v"]
    
    # Add any command line arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    # Run pytest
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()