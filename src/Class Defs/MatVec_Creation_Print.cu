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
    
    // Ensure at least 2 dimensions
    this->ndim = std::max(static_cast<std::uint64_t>(2), static_cast<std::uint64_t>(buf.ndim));
    this->elementCount = 1;
    this->shape.reserve(ndim);
    this->strides.reserve(ndim);
    
    const std::uint64_t itemsize = sizeof(double);

    if (buf.ndim == 0)  // Single element
    {
        this->shape = {1, 1};
        this->strides = {0, 0};
        this->elementCount = 1;
    }
    else if (buf.ndim == 1)  // Vector
    {
        this->shape = {1, static_cast<std::uint64_t>(buf.shape[0])};
        this->strides = {0, itemsize};
        this->elementCount = buf.shape[0];
    }
    else  // 2 or more dimensions
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


MatVec::~MatVec()
{
	cudaFree(this->data);
}

std::string MatVec::toString()
{
	std::ostringstream oss;
	oss << "Dimensions: " << ndim << "\n";
	oss << "Element Count: " << elementCount << "\n";
	oss << "Memory Size: " << memSize << " bytes\n";
	oss << "Shape: [";
	for (size_t i = 0; i < shape.size(); ++i) 
	{
		oss << shape[i];
		if (i < shape.size() - 1) oss << ", ";
	}
	oss << "]\n";
	oss << "Strides: [";
	for (size_t i = 0; i < strides.size(); ++i) 
	{
		oss << strides[i];
		if (i < strides.size() - 1) oss << ", ";
	}
	oss << "]\n";
	oss << "Data (first few elements): [";
	const int maxElementsToPrint = 10;
	std::vector<double> hostData(std::min(maxElementsToPrint, static_cast<int>(elementCount)));
	cudaError_t cudaStatus = cudaMemcpy(hostData.data(), data, hostData.size() * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		oss << "Error copying data from GPU: " << cudaGetErrorString(cudaStatus) << "]\n";
	} 
	else 
	{
		for (size_t i = 0; i < hostData.size(); ++i) 
		{
			oss << hostData[i];
			if (i < hostData.size() - 1) 
			{
				oss << ", ";
			}
		}
		if (elementCount > maxElementsToPrint) 
		{
			oss << ", ...";
		}
		oss << "]\n";
	}
	return oss.str();
}