#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
float tolerance =  0.1;
bool compareMatrices(float *matrix1, float *matrix2, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = fabs(matrix1[i * cols + j] - matrix2[i * cols + j]);
            if (diff > tolerance) {
                printf("\033[31mMatrices differ at (%d, %d): %f != %f\n\033[0m", i, j, matrix1[i * cols + j], matrix2[i * cols + j]);
                return false;
            }
        }
    }
    return true;
}
