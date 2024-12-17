
#include"tool\common.cuh"

void time_start(cudaEvent_t *start, cudaEvent_t *stop, const char *filename, int lineNumber) {
    ErrorCheck(cudaEventCreate(start), filename, lineNumber);
    ErrorCheck(cudaEventCreate(stop), filename, lineNumber);
    ErrorCheck(cudaEventRecord(*start), filename, lineNumber);
// cudaEventQuery 函数的作用是查询指定 CUDA 事件的状态，检查它是否已经完成。
    cudaEventQuery(*start);
}

// 计时结束函数
void time_end(cudaEvent_t *start, cudaEvent_t *stop, float *time, const char *filename, int lineNumber) {
    // cudaEventRecord事件计时函数
    ErrorCheck(cudaEventRecord(*stop), filename, lineNumber);
    ErrorCheck(cudaEventSynchronize(*stop), filename, lineNumber);
    ErrorCheck(cudaEventElapsedTime(time, *start, *stop), filename, lineNumber);
}