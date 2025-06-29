"""
Configuration file for pytest.

This file ensures that the project root directory is added to the Python path
so that modules can be imported correctly during testing.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
