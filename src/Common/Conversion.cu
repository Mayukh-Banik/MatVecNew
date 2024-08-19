#include "MatVecCore/NumPyInteractions.h"
#include "Constants/Common Defs.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

namespace py = pybind11;

template <typename T>
py::array_t<T> toNumPyArray(const MatVec<T>& m)
{
    std::vector<py::ssize_t> shape(m.shape.begin(), m.shape.end());
    std::vector<py::ssize_t> strides(m.strides.begin(), m.strides.end());

    py::array_t<T> arr(shape, strides);

    // Get the buffer info
    py::buffer_info buf = arr.request();
    T* ptr = static_cast<T*>(buf.ptr);

    // Copy data from CUDA device to host
    cudaError_t t = cudaMemcpy(ptr, m.data, m.memSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(t);

    return arr;
}

#define GENERATE_SPECIALIZATION(r, data, T) \
    template py::array_t<T> toNumPyArray<T>(const MatVec<T>& m);

BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)

#undef GENERATE_SPECIALIZATION