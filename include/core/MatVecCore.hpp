#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "defs/Macros.hpp"
#include <vector>


template <typename T>
class MatVec
{
private:

public:
	T* data;
	std::uint64_t ndim;
	std::vector<std::uint64_t> shape;
	std::string device;
	int deviceId;
	std::vector<std::uint64_t> strides;

	MatVec(const nanobind::ndarray<T>& arr);
};