#include "MatVecCore/NumPyInteractions.h"
#include "Constants/Common Defs.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

namespace py = pybind11;

pybind11::array_t<double> toNumPyArray(const MatVec& m)
{
    // Create vectors of dimensions and strides from MatVec
    std::vector<py::ssize_t> shape(m.shape.begin(), m.shape.end());
    std::vector<py::ssize_t> strides(m.strides.begin(), m.strides.end());

    // Remove trailing dimensions of size 1
    while (shape.size() > 1 && shape.back() == 1) {
        shape.pop_back();
        strides.pop_back();
    }

    // Handle scalar case
    if (shape.size() == 1 && shape[0] == 1) {
        shape.clear();
        strides.clear();
    }

    // Create the NumPy array with proper shape and strides
    py::array_t<double> arr(shape, strides);

    // Get the buffer info
    py::buffer_info buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);

    // Copy data from CUDA device to host
    cudaError_t t = cudaMemcpy(ptr, m.data, m.memSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(t);

    return arr;
}

