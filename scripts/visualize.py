#!/usr/bin/env python3
"""superGPT Neural Network Visualizer."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runpy
runpy.run_module("supergpt.tools.visualize", run_name="__main__")
