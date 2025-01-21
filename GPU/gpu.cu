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
    char *nDNA = read_from_file("../data/nDNA.txt");
    char *mtDNA = read_from_file("../data/mtDNA.txt");
    printf("nDNA length: %lld\nmtDNA length : %d\n", (long long)strlen(nDNA), (int)strlen(mtDNA));

    //設定 panalty 參數
    int *panalty_score = (int *)malloc(sizeof(int) * 4);
    for(int i = 0; i < 4; i++){
        switch (i)
        {
        case 0:
            panalty_score[i] = MATCH;   
            break;
        case 1:
            panalty_score[i] = MISMATCH;
            break;
        case 2:
            panalty_score[i] = EXTEND_GAP;
            break;
        case 3:
            panalty_score[i] = OPEN_GAP;
            break;
        default:
            break;
        }
    }

    //分配 GPU 空間
    cudaMemcpyToSymbol(panalty, panalty_score, 4 * sizeof(int));
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * strlen(mtDNA)), __FILE__, __LINE__);
    int *E;
    ErrorCheck(cudaMalloc((int**) &E, sizeof(int) * 6400), __FILE__, __LINE__);
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (strlen(mtDNA) + 1) * 6401), __FILE__, __LINE__);
    int *global_max_score;
    ErrorCheck(cudaMalloc((int**)&global_max_score, sizeof(int)), __FILE__, __LINE__);
    int *global_max_i;
    ErrorCheck(cudaMalloc((int**)&global_max_i, sizeof(int)), __FILE__, __LINE__);
    int *global_max_j;
    ErrorCheck(cudaMalloc((int**)&global_max_j, sizeof(int)), __FILE__, __LINE__);
    // copy mtDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA, strlen(mtDNA) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //fill F vector with -∞
    int threadsPerBlock = 512;
    int blocksPerGrid = (strlen(mtDNA) + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(F, INT_MIN - OPEN_GAP, strlen(mtDNA));
    cudaDeviceSynchronize();

    // fill first row and col with 0 in H
    start = clock();
    int maxLen = MAX(strlen(mtDNA), 6400);
    threadsPerBlock = 512;
    blocksPerGrid = (maxLen + threadsPerBlock - 1) / threadsPerBlock;
    initializeH<<<blocksPerGrid, threadsPerBlock>>>(H, strlen(mtDNA), 6400);
    cudaDeviceSynchronize();
    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("填充 H 花 %.6f 秒\n", elapsed_time);

    // proint_arrInfo<<<1, 1>>>(H, 6587);
    // cudaDeviceSynchronize();


    //根據nDNA長度切段去執行每個subMatrix
    int epoch = (int)(strlen(nDNA) / (size_t)6400);
    threadsPerBlock = 263;
    blocksPerGrid = 40;
    for(int i = 0; i < epoch; i++){
        printf("%d / %d \n", i+1, epoch);
        char *slice = substring(nDNA, 6400 * i, 6400);
        //copy nDNA slice to constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_slice_nDNA, slice, strlen(slice) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        // fill E vector with -∞
        blocksPerGrid = (6400 + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(E, INT_MIN - OPEN_GAP, strlen(slice));
        //start caculate submatrix
        int outer_diag = blocksPerGrid + (strlen(mtDNA) / threadsPerBlock);
        for(int i = 0; i < outer_diag; i++){
            // do first part in all blocks
            cal_first_phase<<<blocksPerGrid, threadsPerBlock>>>(i, threadsPerBlock, (int)(strlen(slice) / blocksPerGrid), strlen(slice), strlen(mtDNA), E, F, H, global_max_score, global_max_i, global_max_j);
            cudaDeviceSynchronize();
            // do second part in all blocks
            cal_second_phase<<<blocksPerGrid, threadsPerBlock>>>(i, threadsPerBlock, (int)(strlen(slice) / blocksPerGrid), strlen(slice), strlen(mtDNA), E, F, H, global_max_score, global_max_i, global_max_j);
            cudaDeviceSynchronize();
        }
        printf("max score : %d at (%d, %d)\n", *global_max_score, *global_max_i, *global_max_j);
    }


    //free memory space
    cudaFree(F);
    cudaFree(E);
    cudaFree(H);
    free(mtDNA); 
    free(nDNA);

    return 0;
}