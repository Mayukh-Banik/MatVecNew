#include "MatVecCore/MatVecClassDeclaration.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

MatVec::MatVec(const pybind11::array& array)
{
	if (array.dtype().kind() != 'f' || array.dtype().itemsize() != sizeof(double)) 
	{
		throw std::runtime_error("Input array must be of type np.float64 (double).");
	}
	else if(!array.dtype().is(pybind11::dtype::of<double>()))
	{
		throw std::runtime_error("Input array must be of type np.float64 (double).");
	}
	pybind11::buffer_info buf = array.request();
	
	this->ndim = static_cast<std::uint64_t>(buf.ndim);
	this->elementCount = 1;
	this->shape.reserve(ndim);
	this->strides.reserve(ndim);
	
	for (pybind11::ssize_t i = 0; i < buf.ndim; i++) 
	{
		this->shape.push_back(static_cast<std::uint64_t>(buf.shape[i]));
		this->strides.push_back(static_cast<std::uint64_t>(buf.strides[i]) / sizeof(double));
		this->elementCount *= this->shape[i];
	}
	this->memSize = this->elementCount * sizeof(double);

	cudaError_t t = cudaMalloc((void**) &this->data, this->memSize);

	cudaMemcpy(this->data, buf.ptr, this->memSize, cudaMemcpyHostToDevice);
}

MatVec::~MatVec()
{
	cudaFree(this->data);
}