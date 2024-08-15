// #include "MatVecCore/MatVecClassDeclaration.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <iostream>

#include "MatVecCore/MatVecClassDeclaration.h"

namespace py = pybind11;

void init_MatVecBindings(py::module_& m);

PYBIND11_MODULE(MatVec, m) 
{
    init_MatVecBindings(m);
}
