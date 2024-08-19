#pragma once

#include <stdexcept>

/**
 * Throws runtime error for cudaError_t
 */
#define CUDA_CHECK_ERROR(x)   \
	if (x != cudaSuccess)   \
	{	\
		throw std::runtime_error(cudaGetErrorString(x));	\
	}

#define DOUBLE_SIZE sizeof(double)

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

/**
 * Usage:
 * 
 * 	#define NUMERIC_TYPES (std::int8_t)(std::uint8_t)(std::int16_t)(std::uint16_t)(std::int32_t)(std::uint32_t)(std::int64_t)(std::uint64_t)(float)(double)
 *	
	#define GENERATE_SPECIALIZATION(r, data, T) \
    	template py::array_t<T> toNumPyArray<T>(const MatVec<T>& m);

	BOOST_PP_SEQ_FOR_EACH(GENERATE_SPECIALIZATION, ~, NUMERIC_TYPES)

#undef GENERATE_SPECIALIZATION
 */
#define NUMERIC_TYPES (std::int8_t)(std::uint8_t)(std::int16_t)(std::uint16_t)(std::int32_t)(std::uint32_t)(std::int64_t)(std::uint64_t)(float)(double)
