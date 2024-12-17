#include<stdio.h>
#include"tool\common.cuh"
#include<math.h>
#include"tool\cudatime.cuh"
#include"tool\compare.cuh"
#include <time.h>
#include <windows.h>
// 矩阵乘实现
__global__
void CUDAmatrixMultiplication(float *A, float *B, float *C, int row_max, int line_max) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < row_max && col < line_max) {
        float sum = 0;
        for (int k = 0; k < row_max; k++) {
            sum += A[row * row_max + k] * B[k * line_max + col];
        }
        C[row * line_max + col] = sum;
    }
}
/**************************************************/

void BASEmatrixMultiplication(float *A, float *B, float *C, int row_max, int line_max, int MAX_num) {
    for (int row = 0; row < row_max; row++) {
        for (int col = 0; col < line_max; col++) {
            float sum = 0.0;
                for (int k = 0; k < line_max; k++) {
                    sum += A[row * line_max + k] * B[k * line_max + col];
                }
            C[row * line_max + col] = sum;
        }
    }
}

// 生成随机数，初始化ABC
__host__
void initdata(float *addr , int elemCount)
{
    for(int i = 0;i < elemCount;i++)
    {
        addr[i] = (float)(rand() & 0xff) / 10.f;
    }
}
/**************************************************/



int main()
{
    setGPU();
// 定义数据大小和矩阵大小
    int row_max;
    int line_max;
    printf("please write :row_max  line_max\n");
    scanf("%d %d",&row_max,&line_max);
    int MAX_num = row_max * line_max;
    int MAX_size = MAX_num * sizeof(float);
/**************************************************/


// 分配主机端内存
    float *A_H, *B_H , *C_H , *C_H1;
    A_H = (float *)malloc(MAX_size);
    B_H = (float *)malloc(MAX_size);
    C_H = (float *)malloc(MAX_size);
    C_H1 = (float *)malloc(MAX_size);
/**************************************************/


// 初始化主机端内存
    if(A_H != NULL && B_H != NULL && C_H != NULL){
        memset(A_H , 0 , MAX_size);
        memset(B_H , 0 , MAX_size);
        memset(C_H , 0 , MAX_size);
        memset(C_H1 , 0 , MAX_size);
    } else {
        printf("fail to allocate memory\n");
        free(A_H);
        free(B_H);
        free(C_H);
        free(C_H1);
        exit(-1);
    }
/**************************************************/

   srand(666);


// 初始化设备端内存
float *A_D, *B_D ,*C_D;
   cudaMalloc((void **)&A_D,MAX_size);
   cudaMalloc((void **)&B_D,MAX_size);
   cudaMalloc((void **)&C_D,MAX_size);

    initdata(A_H , MAX_num);
    initdata(B_H , MAX_num);
/**************************************************/


// 数据搬运
    cudaMemcpy(A_D,A_H,MAX_size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_D,B_H,MAX_size,cudaMemcpyHostToDevice);
    cudaMemcpy(C_D,C_H,MAX_size,cudaMemcpyHostToDevice);
/**************************************************/
float CUDAtotal_time = 0;
cudaEvent_t start, stop; 

dim3 block(16, 16);
dim3 grid((line_max + block.x - 1) / block.x, (row_max + block.y - 1) / block.y);

    time_start(&start , &stop ,__FILE__ , __LINE__);

    for(int i = 0;i<11;i++){

    CUDAmatrixMultiplication<<<grid, block>>>(A_D, B_D, C_D, row_max, line_max);
    cudaMemcpy(C_H, C_D, MAX_size, cudaMemcpyDeviceToHost);

    float single_time;
    time_end(&start, &stop, &single_time, __FILE__, __LINE__);
    CUDAtotal_time += single_time;
    if(i!=10)memset(C_H, 0, MAX_size);
    }

    CUDAtotal_time /= 10;
    
Sleep(2000);
    clock_t start1, end1;
    double total_cpu_time_used = 0;
    for (int i = 0; i < 10; ++i) {
        start1 = clock();
        BASEmatrixMultiplication(A_H, B_H, C_H1, row_max, line_max, MAX_num);
        end1 = clock();
        total_cpu_time_used += ((double)(end1 - start1)) * 1000 / CLOCKS_PER_SEC;
        // 每次迭代后重置C矩阵，以便下一次迭代
        if(i!=9)memset(C_H1, 0, MAX_size);
    }
    double average_cpu_time_used = total_cpu_time_used / 10;
// 比较结果
    printf("-------------------------------------------------\n");
    if(compareMatrices(C_H , C_H1 , row_max , line_max , tolerance)){
        printf("\033[34mTest Success\n");
        float ratio = 0.0;
        ratio = average_cpu_time_used / CUDAtotal_time;
        printf("ratio is :%.7f\n\033[0m",ratio);
    }else{
        printf("\033[31mTest False\n\033[0m");
    }
    printf("-------------------------------------------------\n");
/**************************************************/


// 释放内存
    free(A_H);
    free(B_H);
    free(C_H);
    free(C_H1);
    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(C_D);
/**************************************************/


return 0;
}