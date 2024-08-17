import sys
import os

# Add the parent directory of 'MatVecNew' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build.debug import MatVec as mv
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
				mv_obj = mv.MatVec(arr)
			assert isinstance(mv_obj, mv.MatVec)

	def test_exception_int32(self):
		arrays = [self.Fscalar, self.Fvector, self.Fmatrix]
		for arr in arrays:
			with pytest.raises(Exception):
				mv.MatVec(arr)

	def test_mixed_types(self):
		# Should not raise exception
		with nullcontext():
			mv_obj = mv.MatVec(self.matrix)
		assert isinstance(mv_obj, mv.MatVec)

		# Should raise exception
		with pytest.raises(Exception):
			mv.MatVec(self.Fmatrix)
	
	def test_return_back(self):
		arrays = [self.scalar, self.vector, self.matrix]
		for arr in arrays:
			a = mv.MatVec(arr)
			b = mv.toNumPyArray(a)
			assert np.array_equal(b, arr), f"""Arrays are not equal:
            Original array: {arr}
            Returned array: {b}
            Shape of original: {arr.shape}
            Shape of returned: {b.shape}
            Dtype of original: {arr.dtype}
            Dtype of returned: {b.dtype}
            Difference: {np.abs(b - arr)}"""
