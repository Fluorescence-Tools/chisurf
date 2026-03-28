import sys
import os

# Add modules/chinet to sys.path so 'chinet' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add tests dir to sys.path so 'utils' and 'constants' can be imported
sys.path.insert(0, os.path.dirname(__file__))
