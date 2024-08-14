#pragma once

#include <vector>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <stdexcept>
#include <cstdint>

class MatVec
{
public:
    double* data;
    std::uint64_t ndim;
    std::uint64_t elementCount;
    std::uint64_t memSize;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;


    MatVec(const pybind11::array& array);

    ~MatVec();
private:
};