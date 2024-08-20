#include "core/MatVecCore.hpp"
#include "cuda_runtime_api.h"
#include <sstream>
#include <numeric>
#include <nanobind/stl/vector.h>
#include "cuda_runtime.h"
#include <cstring>
#include <nanobind/stl/complex.h>

namespace nb = nanobind;

template <typename T>
MatVec<T>::MatVec(const nanobind::ndarray<T, nanobind::c_contig>& arr)
{
    this->ndim = static_cast<std::uint64_t>(arr.ndim());
    this->shape.reserve(this->ndim);
    this->strides.reserve(this->ndim);
    this->elementCount = 1;
    for (std::uint64_t count = 0; count < this->ndim; count++)
    {
        this->shape.push_back(static_cast<std::uint64_t>(arr.shape(count)));
        this->strides.push_back(static_cast<std::uint64_t>(arr.stride(count)));
        this->elementCount *= static_cast<std::uint64_t>(arr.shape(count));
    }
    this->elementSize = static_cast<int>(sizeof(T));
    this->nBytes = this->elementCount * static_cast<std::uint64_t>(this->elementSize);
    cudaError_t t = cudaMalloc((void**) &data, this->nBytes);
    CUDA_CHECK_ERROR(t);
    t = cudaMemcpy(data, arr.data(), this->nBytes, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(t);
}

template <typename T>
std::string MatVec<T>::toStringVerbose() const
{
    std::ostringstream oss;
    oss << "MatVec<" << typeid(T).name() << "> {\n";
    oss << "  data: " << static_cast<const void*>(data) << "\n";
    oss << "  ndim: " << ndim << "\n";
    oss << "  shape: [";
    for (size_t i = 0; i < shape.size(); ++i) 
	{
        oss << shape[i];
        if (i != shape.size() - 1) oss << ", ";
    }
    oss << "]\n";
    oss << "  strides: [";
    for (size_t i = 0; i < strides.size(); ++i) 
	{
        oss << strides[i];
        if (i != strides.size() - 1) oss << ", ";
    }
    oss << "]\n";
    oss << "  elementSize: " << elementSize << " bytes\n";
    oss << "  nBytes: " << nBytes << " bytes\n";
    oss << "  elementCount: " << elementCount << "\n";
    oss << "}";
    return oss.str();
}

template <typename T>
std::string MatVec<T>::toStringData() const
{
    std::ostringstream oss;
    oss << "[";
    const int maxElements = 40;
    int totalElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::uint64_t>());
    int elementsToShow = std::min(maxElements, totalElements);
    std::vector<T> hostData(elementsToShow);
    cudaError_t err = cudaMemcpy(hostData.data(), data, elementsToShow * sizeof(T), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR(err);
    for (int i = 0; i < elementsToShow; ++i)
    {
        if (i > 0) oss << ", ";
        oss << hostData[i];
    }
    if (elementsToShow < totalElements)
    {
        oss << ", ...";
    }
    oss << "]";
    return oss.str();
}

template <typename T>
MatVec<T>::~MatVec()
{
	cudaFree((void*) data);
}

template <typename T>
nb::ndarray<nb::numpy, T> MatVec<T>::toNumPy() 
{
	T* data = nullptr;
	try
	{
		data = new T[this->elementCount];
	}
	catch(const std::exception& e)
	{
		throw std::runtime_error("Probably ran out of memory copying the memory to host");
	}
	cudaError_t t = cudaMemcpy(data_ptr, this->data, this->nBytes, cudaMemcpyDeviceToHost);
	if (t != cudaSuccess)
	{
		delete[] data_ptr;
		throw std::runtime_error(cudaGetErrorString(t));
	}
	nb::capsule owner(data_ptr, [](void* p) noexcept 
	{
		delete[] (T*) p;
	});
	return nb::ndarray<nb::numpy, T>(data_ptr, this->ndim, this->shape.data(), owner);
}














#define GENERATE_SPECIALIZATION(r, data, T) \
	template class MatVec<T>;
BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)
#undef GENERATE_SPECIALIZATION


