#pragma once

#include "MatVecCore/MatVecClassDeclaration.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename T>
pybind11::array_t<T> toNumPyArray(const MatVec<T>& m);
