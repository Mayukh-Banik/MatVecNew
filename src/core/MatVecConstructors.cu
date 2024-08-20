#include "core/MatVecCore.hpp"
#include "cuda_runtime_api.h"
#include <sstream>
#include <numeric>
#include <nanobind/stl/vector.h>
#include <cstring>

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
nanobind::ndarray<T> MatVec<T>::toNumPy()
{
    std::vector<T> hostData(elementCount);
    cudaError_t err = cudaMemcpy(hostData.data(), data, nBytes, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR(err);
    T* data_ptr = new T[elementCount];
    memcpy(data_ptr, hostData.data(), nBytes);
    auto capsule = nanobind::capsule(data_ptr, [](void *p) noexcept { delete[] static_cast<T*>(p); });
    size_t* shape_ptr = new size_t[ndim];
    int64_t* strides_ptr = new int64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) 
	{
        shape_ptr[i] = static_cast<size_t>(shape[i]);
        strides_ptr[i] = static_cast<int64_t>(strides[i] * sizeof(T));
    }
    return nanobind::ndarray<T>(
        data_ptr,                // Data pointer
        ndim,                    // Number of dimensions
        shape_ptr,               // Shape
        capsule,                 // Capsule for memory management
        strides_ptr,             // Strides
        nanobind::dtype<T>(),    // Data type
        0                        // Device (0 for CPU)
    );
}










#define GENERATE_SPECIALIZATION(r, data, T) \
	template class MatVec<T>;
BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)
#undef GENERATE_SPECIALIZATION


