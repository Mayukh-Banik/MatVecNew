#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "MatVecCore/MatVecClassDeclaration.h"

namespace py =  pybind11;
void init_MatVecBindings(py::module_& m)
{
    py::class_<MatVec>(m, "MatVec")
        .def(py::init<py::array>())
        .def("__str__", &MatVec::toString)
        .def("__repr__", &MatVec::toString);;
}