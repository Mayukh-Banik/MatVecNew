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

    std::string toString();

    double get(std::uint64_t index);
    void set(std::uint64_t index, double val);
    std::uint64_t length() {return this->elementCount;}
private:
};