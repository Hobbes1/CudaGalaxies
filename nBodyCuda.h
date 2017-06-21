#pragma once 
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "vector_types.h"
#include <cuda.h>
#include <stdio.h>
#include <device_functions.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
// probably some unecessary includes here

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}
