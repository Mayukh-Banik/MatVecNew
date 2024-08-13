#include "temp.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

int add(int a, int b)
{
    double* x;
    cudaMalloc((void**) &x, sizeof(double));
    std::cout << x << std::endl;
    cudaFree(x);
    return a + b;
}