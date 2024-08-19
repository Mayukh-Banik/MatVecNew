#pragma once

#include <vector>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <stdexcept>
#include <cstdint>
#include <pybind11/stl.h>

template <typename T>
class MatVec
{
public:
    T* data;
    std::uint64_t ndim;
    std::uint64_t elementCount;
    std::uint64_t memSize;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;


    MatVec(const pybind11::array& array);

	MatVec(T* data, std::uint64_t ndim, std::uint64_t elementCount, std::uint64_t memeSize, const std::vector<std::uint64_t> shape, const std::vector<std::uint64_t> strides);

    ~MatVec();

    std::string toString();

    T get(std::uint64_t index);
    void set(std::uint64_t index, T val);
    std::uint64_t length() {return this->elementCount;}
	pybind11::tuple get_shape() const {return pybind11::cast(shape);}
    pybind11::tuple get_strides() const {return pybind11::cast(strides);}
private:
};