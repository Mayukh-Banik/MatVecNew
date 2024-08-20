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
	std::vector<std::uint64_t> strides;
	int elementSize;
	std::uint64_t nBytes;
	std::uint64_t elementCount;
	int deviceId;
	std::string device;


	MatVec(const nanobind::ndarray<T, nanobind::c_contig>& arr);

	std::string toStringVerbose() const;
	std::string toStringData() const;

	nanobind::ndarray<nanobind::numpy, T> toNumPy();
	~MatVec();
};