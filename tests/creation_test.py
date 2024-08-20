import sys
import os

# Add the parent directory of 'MatVecNew' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.name == 'posix':  # For Linux and macOS
    from build.Release import MatVec as mv
elif os.name == 'nt':  # For Windows
    from build.Debug.Debug import MatVec as mv
else:
    raise ImportError("Unsupported operating system")

import numpy as np
import pytest
import time
from contextlib import nullcontext


class TestMatVecCreation:
	@pytest.fixture(autouse=True)
	def setup(self):
		# This fixture will run before each test method in this class
		self.scalar = np.array(1, dtype=np.float64)
		self.vector = np.array([1, 2, 3], dtype=np.float64)
		self.matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)
		self.Fscalar = np.array(1, dtype=np.int32)
		self.Fvector = np.array([1, 2, 3], dtype=np.int32)
		self.Fmatrix = np.array([[1, 2], [3, 4]], dtype=np.int32)

	def test_no_exception_float64(self):
		arrays = [self.scalar, self.vector, self.matrix]
		for arr in arrays:
			with nullcontext():
				mv_obj = mv.matvec(arr)
			assert isinstance(mv_obj, mv.matvec)

	# def test_exception_int32(self):
	# 	arrays = [self.Fscalar, self.Fvector, self.Fmatrix]
	# 	for arr in arrays:
	# 		with pytest.raises(Exception):
	# 			mv.matvec(arr)

	# def test_mixed_types(self):
	# 	# Should not raise exception
	# 	with nullcontext():
	# 		mv_obj = mv.matvec(self.matrix)
	# 	assert isinstance(mv_obj, mv.matvec)

	# 	# Should raise exception
	# 	with pytest.raises(Exception):
	# 		mv.matvec(self.Fmatrix)
	
	def test_return_back(self):
		arrays = [self.scalar, self.vector, self.matrix]
		for arr in arrays:
			a = mv.matvec(arr)
			b = a.toNumPy()
			

	def test_dimensions(self):
		arrays = [self.scalar, self.vector, self.matrix]
		for arr in arrays:
			b = mv.matvec(arr)
			assert b.shape == arr.shape
			assert b.strides == arr.strides
			assert b.ndim == arr.ndim

