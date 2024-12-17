#pragma once
#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>
cudaError_t ErrorCheck(cudaError_t error_code, const char *filename ,int lineNumber)
{
    if(error_code !=cudaSuccess)
    {
        printf("\033[31mCUDA error:\r\ncode = %d ,name = %s ,description = %s \r\nfile=%s ,line%d\r\n\033[0m",
        error_code , cudaGetErrorName(error_code) , cudaGetErrorString(error_code) , filename , lineNumber);
    }
    return error_code;
}

void setGPU(){

    int iDeviceCount = 0;
    
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount),__FILE__,__LINE__);

    if(error != cudaSuccess || iDeviceCount == 0)
    {
        printf("\033[31mNo cuda compatable GPU found!\033[0m\n");
        exit(-1);
    }
    else
    {
        printf("this is %d GPU \n",iDeviceCount);
    }

    int iDev = 0;
    error = ErrorCheck(cudaSetDevice(iDev),__FILE__,__LINE__);

    if(error != cudaSuccess)
    {
        printf("\033[31mfail to set GPU 0 for computing.\033[0m\n");
        exit(-1);
    }
    else
    {
        printf("set GPU 0 for cumputing.\n");
    }

}

void printMatrix(float *matrix, int rows, int cols, const char *filename, int lineNumber) {
    if (matrix == NULL) {
        printf("\033[31mMatrix is NULL.\033[0m\n");
        return;
    }

    printf("Matrix (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

