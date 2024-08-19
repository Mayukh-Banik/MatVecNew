#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "MatVecCore/MatVecClassDeclaration.h"
#include "Bindings/Bindings.h"

namespace py =  pybind11;

template <typename T>
void init_MatVecBindings(py::module_& m)
{
	py::class_<MatVec<T>>(m, "MatVec", R"(
		A class meant to recreate NumPy on the GPU as much as possible.

		Can only be created from existing NumPy arrays that are C style and are of dtype=np.float64.
	)")
		.def(py::init<py::array>(), R"(
			Initialize a MatVec object.

			Parameters:
			-----------
			array : numpy.ndarray
				Input NumPy array with dtype=np.float64 and C style ordering.

			Takes in any NumPy array of dtype=np.float64 and C style ordering in memory.
			Takes in raw memory data, so Fortran style will be compatible, and so will
			transposed matrices, but will be interpreted as the C style memory layout.
			
			Example:
			import numpy as np
			import MatVec as mv
			a = np.array([5], dtype=np.float64)
			b = mv.MatVec(a)
		)")
		.def("__str__", &MatVec<T>::toString)
		.def("__repr__", &MatVec<T>::toString)
		.def("__len__", &MatVec<T>::length)
		.def("__getitem__", &MatVec<T>::get, R"(
			Get an item from the MatVec object.

			Not recommended due to memory copy time from host to device.

			Parameters:
			-----------
			index : int
				Index of the item to retrieve.

			Returns:
			--------
			float
				Value at the specified index.
		)")
		.def("__setitem__", &MatVec<T>::set, R"(
			Set an item in the MatVec object.

			Not recommended due to memory copy time from host to device.

			Parameters:
			-----------
			index : int
				Index of the item to set.
			value : float
				Value to set at the specified index.
		)")
		.def_readonly("ndim", &MatVec<T>::ndim)
        .def_readonly("elementCount", &MatVec<T>::elementCount)
        .def_readonly("memSize", &MatVec<T>::memSize)
		.def_property_readonly("shape", &MatVec<T>::get_shape)
        .def_property_readonly("strides", &MatVec<T>::get_strides);
}

template void init_MatVecBindings<double>(pybind11::module_&);