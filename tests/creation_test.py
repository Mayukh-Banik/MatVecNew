import sys
import os

# Add the parent directory of 'MatVecNew' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build.debug import MatVec
import numpy as np
import pytest
import time
# x = np.array([[0, 1, 2, 3, 4],
#               [5, 6, 7, 8, 9]], dtype=np.float64)
# b = MatVec.MatVec(x)
# print(b)

x = np.array([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]], dtype=np.float64)
b = MatVec.MatVec(x)

def test_basic_print(capsys):
    print(b)
    captured = capsys.readouterr()
    assert captured.out != ""

def test_len():
    assert len(b) == 10