#pragma once

#include "MatVecCore/MatVecClassDeclaration.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

pybind11::array_t<double> toNumPyArray(const MatVec& m);
