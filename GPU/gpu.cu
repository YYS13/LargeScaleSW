#include "functions.cuh"


int main(){
    clock_t start, end;
    // 設定由哪個 gpu 執行
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);
    printf("GPU global memory space : %zu Bytes\n", prop.totalGlobalMem);
    // 讀取數據
    char *nDNA = read_from_file("./nDNA.txt");
    char *mtDNA = read_from_file("./mtDNA.txt");
    printf("nDNA length: %lld\nmtDNA length : %d\n", (long long)strlen(nDNA), (int)strlen(mtDNA));

    //分配 GPU 空間
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * strlen(mtDNA)), __FILE__, __LINE__);
    int *E;
    ErrorCheck(cudaMalloc((int**) &E, sizeof(int) * 6400), __FILE__, __LINE__);
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (strlen(mtDNA) + 1) * 6401), __FILE__, __LINE__);
    // copy mtDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA, strlen(mtDNA) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //利用 gpu multithreads 對 F 填初始值
    start = clock();
    int threadsPerBlock = 512;
    int blocksPerGrid = (strlen(mtDNA) + threadsPerBlock - 1) / threadsPerBlock;

    fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(F, INT_MIN - OPEN_GAP, strlen(mtDNA));
    cudaDeviceSynchronize();
    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("填充 F 花 %.6f 秒\n", elapsed_time);
    //根據nDNA長度切段去執行每個subMatrix
    int epoch = (int)(strlen(nDNA) / (size_t)6400);
    for(int i = 0; i < epoch; i++){
        printf("%d / %d ", i+1, epoch);
        char *slice = substring(nDNA, 6400 * i, 6400);
        ErrorCheck(cudaMemcpyToSymbol(device_slice_nDNA, slice, strlen(slice) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    }


    //free memory space
    cudaFree(F);
    cudaFree(E);
    cudaFree(H);
    free(mtDNA); 
    free(nDNA);

    return 0;
}