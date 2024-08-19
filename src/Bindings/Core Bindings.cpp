// #include "MatVecCore/MatVecClassDeclaration.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <iostream>

#include "MatVecCore/MatVecClassDeclaration.h"
#include "MatVecCore/NumPyInteractions.h"

#include "Bindings/Bindings.h"
namespace py = pybind11;

PYBIND11_MODULE(MatVec, m) 
{
    init_MatVecBindings<double>(m);
	m.def("toNumPyArray", &toNumPyArray<double>);

}
