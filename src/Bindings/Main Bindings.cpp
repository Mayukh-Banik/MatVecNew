#include "core/MatVecCore.hpp"

#include <iostream>

namespace nb = nanobind;

template <typename T>
MatVec<T>::MatVec(const nanobind::ndarray<T>& arr) 
{
	// Set data pointer
	data = const_cast<T*>(arr.data());

	// Set number of dimensions
	ndim = arr.ndim();

	// Set shape
	shape.resize(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		shape[i] = arr.shape(i);
	}

	// Set device and device ID
	device = arr.device_type() == nanobind::device::cpu::value ? "cpu" : "cuda";
	deviceId = arr.device_id();

	// Set strides
	strides.resize(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		strides[i] = arr.stride(i);
	}
}

#define GENERATE_SPECIALIZATION(r, data, T) \
    template class MatVec<T>;

BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)

#undef GENERATE_SPECIALIZATION

// template class MatVec<double>;

template <typename T>
void process_array(nb::ndarray<T> input) {
    MatVec<T> mat_vec(input);
	std::cout << "MatVec object was created for array of type: ";
    if constexpr (std::is_same_v<T, std::int8_t>)
        std::cout << "int8_t";
    else if constexpr (std::is_same_v<T, std::uint8_t>)
        std::cout << "uint8_t";
    else if constexpr (std::is_same_v<T, std::int16_t>)
        std::cout << "int16_t";
    else if constexpr (std::is_same_v<T, std::uint16_t>)
        std::cout << "uint16_t";
    else if constexpr (std::is_same_v<T, std::int32_t>)
        std::cout << "int32_t";
    else if constexpr (std::is_same_v<T, std::uint32_t>)
        std::cout << "uint32_t";
    else if constexpr (std::is_same_v<T, std::int64_t>)
        std::cout << "int64_t";
    else if constexpr (std::is_same_v<T, std::uint64_t>)
        std::cout << "uint64_t";
    else if constexpr (std::is_same_v<T, float>)
        std::cout << "float";
    else if constexpr (std::is_same_v<T, double>)
        std::cout << "double";
    else if constexpr (std::is_same_v<T, long double>)
        std::cout << "long double";
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        std::cout << "complex<float>";
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        std::cout << "complex<double>";
    else if constexpr (std::is_same_v<T, bool>)
        std::cout << "bool";
    else
        std::cout << "unknown type";

    std::cout << std::endl;
}


// template void process_array<double>(nb::ndarray<double> input);

#define GENERATE_SPECIALIZATION(r, data, T) \
    template void process_array<T>(nb::ndarray<T> input);

BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)

#undef GENERATE_SPECIALIZATION




#define GENERATE_BINDING(r, data, T) \
    m.def("process_array", &process_array<T>, "Process a " BOOST_PP_STRINGIZE(T) " ndarray");


NB_MODULE(MatVec, m) 
{
	BOOST_PP_SEQ_FOR_EACH(GENERATE_BINDING, ~, NUMERIC_TYPES)
    // m.def("process_array", &process_array<double>, "Process a double ndarray");
}

#undef GENERATE_BINDING