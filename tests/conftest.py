"""Pytest conftest — adds tests/ to sys.path so helpers module is importable."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
