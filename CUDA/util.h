#ifndef __UTIL_H__
#define __UTIL_H__
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CEILING(a, b) (((a) + (b) - 1) / (b))
#define CUDA_SAFE_CALL(a) (cudaErrorCheck((a), __FILE__, __LINE__))

/* Swap references to type T*/
template <typename T> inline void swap(T& a, T& b){
        T temp = a;
        a = b;
        b = temp;
}

/* Assert style Cuda error handling */
inline void cudaErrorCheck(cudaError_t err, const char *file, const int line){
        if (cudaSuccess != err){
                fprintf(stderr, "Cuda error at %s:%d : %s\n", file, line, cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
}

#endif
