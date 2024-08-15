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

