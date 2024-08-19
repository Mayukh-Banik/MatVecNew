#include "MatVecCore/MatVecClassDeclaration.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

#include "Constants/Common Defs.h"

template <typename T>
std::string MatVec<T>::toString()
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
	std::vector<T> hostData(std::min(maxElementsToPrint, static_cast<int>(elementCount)));
	cudaError_t cudaStatus = cudaMemcpy(hostData.data(), data, hostData.size() * sizeof(T), cudaMemcpyDeviceToHost);
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

template <typename T>
T MatVec<T>::get(std::uint64_t index)
{
	T x;
	cudaError_t t = cudaMemcpy(&x, this->data + index, sizeof(T), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR(t);
	return x;
}


template <typename T>
void MatVec<T>::set(std::uint64_t index, T value)
{
	cudaError_t t = cudaMemcpy(this->data + index, &value, sizeof(T), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR(t);
}

template std::string MatVec<double>::toString();
template double MatVec<double>::get(std::uint64_t index);
template void MatVec<double>::set(std::uint64_t index, double value);