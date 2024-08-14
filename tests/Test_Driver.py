import sys
import os

# Add the parent directory of 'MatVecNew' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build.debug import MatVec

help(MatVec)