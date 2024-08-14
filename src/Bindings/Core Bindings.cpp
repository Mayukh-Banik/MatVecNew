// #include "MatVecCore/MatVecClassDeclaration.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <iostream>

#include "MatVecCore/MatVecClassDeclaration.h"

void process_array(const pybind11::array& arr) {
    // Ensure the input array is of type float64 (double)
    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != sizeof(double)) {
        throw std::runtime_error("Input array must be of type np.float64 (double).");
    }

    // Safely access the array data
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Example processing (just printing the first element here)
    std::cout << "First element: " << ptr[0] << std::endl;
}

namespace py = pybind11;
PYBIND11_MODULE(MatVec, m) 
{
    m.def("process_array", &process_array, "Process a NumPy array of type np.float64 (double)");
    py::class_<MatVec>(m, "MatVec")
        .def(py::init<py::array>());
}
