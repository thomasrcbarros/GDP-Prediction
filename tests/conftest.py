"""Pytest configuration: puts ``src/`` and the project root on ``sys.path`` so the
tests can import the ``gdp_nowcast`` package and ``config``.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
