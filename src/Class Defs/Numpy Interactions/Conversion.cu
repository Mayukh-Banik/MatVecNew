#include "MatVecCore/NumPyInteractions.h"
#include "Constants/Common Defs.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

namespace py = pybind11;

py::array_t<double> toNumPyArray(const MatVec& m)
{
    std::vector<py::ssize_t> shape(m.shape.begin(), m.shape.end());
    std::vector<py::ssize_t> strides(m.strides.begin(), m.strides.end());

    py::array_t<double> arr(shape, strides);

    // Get the buffer info
    py::buffer_info buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Copy data from CUDA device to host
    cudaError_t t = cudaMemcpy(ptr, m.data, m.memSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(t);

    return arr;
}

