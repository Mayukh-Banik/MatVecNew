#include "MatVecCore/MatVecClassDeclaration.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#include "Constants/Common Defs.h"

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
    const std::uint64_t itemsize = sizeof(double);
    if (buf.ndim == 0)
    {
        this->elementCount = 1;
    }
    else if (buf.ndim == 1)
    {
        this->shape = {static_cast<std::uint64_t>(buf.shape[0])};
        this->strides = {itemsize};
        this->elementCount = buf.shape[0];
    }
    else
    {
        for (pybind11::ssize_t i = 0; i < buf.ndim; i++) 
        {
            this->shape.push_back(static_cast<std::uint64_t>(buf.shape[i]));
            this->strides.push_back(static_cast<std::uint64_t>(buf.strides[i]));
            this->elementCount *= this->shape[i];
        }
    }
    this->memSize = this->elementCount * itemsize;
    cudaError_t t = cudaMalloc((void**) &this->data, this->memSize);
    CUDA_CHECK_ERROR(t);
    t = cudaMemcpy(this->data, buf.ptr, this->memSize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(t);
}

MatVec::MatVec(double* data, std::uint64_t ndim, std::uint64_t elementCount, std::uint64_t memSize, const std::vector<std::uint64_t> shape, const std::vector<std::uint64_t> strides)
{
	this->ndim = ndim;
	this->elementCount = elementCount;
	this->memSize = memSize;
	this->shape = std::vector<std::uint64_t>(shape);
	this->strides = std::vector<std::uint64_t>(strides);
	cudaError_t t = cudaMemcpy((void**) &this->data, data, this->memSize, cudaMemcpyDeviceToDevice);
	CUDA_CHECK_ERROR(t);
}

MatVec::~MatVec()
{
	cudaFree(this->data);
}