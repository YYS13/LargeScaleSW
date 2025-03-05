#include "functions.cuh"


int main(){
    clock_t start, end;
    // 設定由哪個 gpu 執行
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    //各種參數設定
    int blocksPerGrid = prop.multiProcessorCount;
    printf("GPU%d 最大 SMs 數量 = %d\n", device_id, blocksPerGrid);
    int nDNA_slice_len;

    // 尋找最大的 nDNA_slice 長度且可以被 blocksPerGrid 整除
    nDNA_slice_len = find_max_slice_len(MTDNA_LEN, prop.totalGlobalMem, blocksPerGrid);
    printf("nDNA_slice_len = %d\n", nDNA_slice_len);
    int threadsPerBlock = (nDNA_slice_len / blocksPerGrid) / 2;
    printf("blocksPerGrid = %d\n", blocksPerGrid);
    printf("threadsPerBlocks = %d\n", threadsPerBlock);

    // 讀取數據
    char *nDNA = read_from_file("../data/nDNA.txt");
    char *mtDNA = read_from_file("../data/mtDNA.txt");
    mtDNA = substring(mtDNA, 0, 16806);
    printf("nDNA length = %zu\n", strlen(nDNA));
    printf("mtDNA length = %zu\n", strlen(mtDNA));

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

    //配置 gpu 空間
    // copyH
    int *copyH = (int*)calloc((MTDNA_LEN + 1) * (nDNA_slice_len + 1), sizeof(int));
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (MTDNA_LEN + 1) * (nDNA_slice_len + 1)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(H, copyH, sizeof(int) * (MTDNA_LEN + 1) * (nDNA_slice_len + 1), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // F
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * MTDNA_LEN), __FILE__, __LINE__);

    // E
    int *E;
    ErrorCheck(cudaMalloc((int**) &E, sizeof(int) * nDNA_slice_len), __FILE__, __LINE__);

    // 把 penalty score 複製到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(panalty, panalty_score, 4 * sizeof(int)), __FILE__, __LINE__);

    // 配置 max_score 位置
    int *global_max_score;
    ErrorCheck(cudaMalloc((int**)&global_max_score, sizeof(int)), __FILE__, __LINE__);
    ErrorCheck(cudaMemset(global_max_score, 0, sizeof(int)), __FILE__, __LINE__);
    int *global_max_i;
    ErrorCheck(cudaMalloc((int**)&global_max_i, sizeof(int)), __FILE__, __LINE__);
    int *global_max_j;
    ErrorCheck(cudaMalloc((long long**)&global_max_j, sizeof(long long)), __FILE__, __LINE__);

    // copy mtDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA, MTDNA_LEN + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //fill F vector with -∞
    int blocks = (MTDNA_LEN + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocks, threadsPerBlock>>>(F, INT_MIN - EXTEND_GAP + 1, MTDNA_LEN);
    cudaDeviceSynchronize();

    // 計算外部對角線次數以及一個 block 負責的矩陣大小 R x C
    int outer_diag = blocksPerGrid + (MTDNA_LEN / threadsPerBlock);
    int R = threadsPerBlock;
    int C = nDNA_slice_len / blocksPerGrid;
    printf("R = %d, C = %d \n", R, C);
    printf("外部對角線共 : %d\n", outer_diag);

    // 開始計算
    start = clock();
    for(int epoch = 0; epoch <= strlen(nDNA) - nDNA_slice_len; epoch += nDNA_slice_len){
        //if(epoch/nDNA_slice_len + 1 == 3) break;
        printf("Epoch %d/%d\n", epoch/nDNA_slice_len + 1, (int)(strlen(nDNA)/nDNA_slice_len));
        char* slice = substring(nDNA, epoch, nDNA_slice_len);
        // copy nDNA 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, slice, nDNA_slice_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        blocks = (nDNA_slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_value<<<blocks, threadsPerBlock>>>(E, INT_MIN - EXTEND_GAP + 1, nDNA_slice_len);
        cudaDeviceSynchronize();

        
        for(int i = 0; i < outer_diag; i++){
            //printf("%d/%d\n", i, outer_diag);
            //計算前半段
            cal_first_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len);
            cudaDeviceSynchronize();
        }

        if(MTDNA_LEN % threadsPerBlock != 0){
            int restThreadsPerBlock = MTDNA_LEN % threadsPerBlock;
            int start_row = (MTDNA_LEN / threadsPerBlock) * threadsPerBlock + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGrid; i++){
                int iter = i == blocksPerGrid ? restThreadsPerBlock - 1 : C;
                do_rest_row <<<1, restThreadsPerBlock, restThreadsPerBlock * sizeof(Result)>>>(i, restThreadsPerBlock, start_row, nDNA_slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len);
                cudaDeviceSynchronize();    
            }
        }
        

        //把H最後一行數值搬到第一行，提供下一個子矩陣計算
        blocks = ((MTDNA_LEN + threadsPerBlock - 1) / threadsPerBlock) + 1;
        move_data<<<blocks, threadsPerBlock>>>(H, MTDNA_LEN, nDNA_slice_len);
        cudaDeviceSynchronize();
    }

    //計算剩下部份
    int rest_nDNA_len = strlen(nDNA) % nDNA_slice_len;
    
    if(rest_nDNA_len > 0){
        printf("nDNA 剩下長度 = %d\n", rest_nDNA_len);
        char *rest_slice = substring(nDNA, strlen(nDNA) - rest_nDNA_len, rest_nDNA_len);
        // copy nDNA 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, rest_slice, rest_nDNA_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        blocksPerGrid = find_best_blocks(rest_nDNA_len, blocksPerGrid);
        threadsPerBlock = (rest_nDNA_len / blocksPerGrid) / 2;
        //fill E vector with -∞
        blocks = (nDNA_slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_value<<<blocks, threadsPerBlock>>>(E, INT_MIN - EXTEND_GAP + 1, nDNA_slice_len);
        cudaDeviceSynchronize();

        printf("新的 blocksPerGrid = %d\n", blocksPerGrid);
        printf("新的 threadsPerBlock = %d\n", threadsPerBlock);
        ////重新計算 outer_dig 、R 、C
        outer_diag = blocksPerGrid + (MTDNA_LEN / threadsPerBlock);
        R = threadsPerBlock;
        C = rest_nDNA_len / blocksPerGrid;
        printf("新的 R = %d, C = %d \n", R, C);
        printf("外部對角線共 : %d\n", outer_diag);

        long long start_col = strlen(nDNA) - rest_nDNA_len;
        // 開始計算
        for(int i = 0; i < outer_diag; i++){
            //printf("%d/%d\n", i, outer_diag);
            //計算前半段
            cal_first_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len);
            cudaDeviceSynchronize();
        }

        if(MTDNA_LEN % threadsPerBlock != 0){
            int restThreadsPerBlock = MTDNA_LEN % threadsPerBlock;
            int start_row = (MTDNA_LEN / threadsPerBlock) * threadsPerBlock + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGrid; i++){
                int iter = i == blocksPerGrid ? restThreadsPerBlock - 1 : C;
                do_rest_row <<<1, restThreadsPerBlock, restThreadsPerBlock * sizeof(Result)>>>(i, restThreadsPerBlock, start_row, nDNA_slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len);
                cudaDeviceSynchronize();    
            }
        }

        
    }


    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）

    //打印最大值
    int maxScore, maxI;
    long long maxJ;
    cudaMemcpy(&maxScore, global_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxI, global_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxJ, global_max_j, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("max score = %d at (%d, %lld)  \n", maxScore, maxI, maxJ);
    // 釋放 cpu 記憶體空間
    free(nDNA);
    free(mtDNA);
    free(copyH);
    free(panalty_score);

    // 釋放 gpu 記憶體空間
    cudaFree(H);
    cudaFree(E);
    cudaFree(F);
    cudaFree(global_max_score);
    cudaFree(global_max_i);
    cudaFree(global_max_j);

    // 打印執行時間
    printf("total time: %.6f seconds\n", elapsed_time);
    convert_time(elapsed_time);
    return 0;
}