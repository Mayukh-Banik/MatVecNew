#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

template <typename T>
void init_MatVecBindings(pybind11::module_& m);

