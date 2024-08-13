#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include "temp.h"

PYBIND11_MODULE(MatVec, m)
{
    m.def("add", &add);
}