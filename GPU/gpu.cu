#include "functions.cuh"


int main(int argc, char *argv[]){
    if(argc != 5){
        printf("Usage: %s <mtDNA Data Path> <nDNA Data Path> <threads Per Block> <expand nDNA option 0 = no 1 = yes>\n", argv[0]);
        return 1;
    }


    clock_t start, end;
    // 設定由哪個 gpu 執行
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    // 讀取數據
    char *nDNA = read_from_file(argv[2]);
    replaceN(nDNA);
    char *mtDNA = read_from_file(argv[1]);
    if(atoi(argv[4]) == 1){
        mtDNA = expand_mtDNA(mtDNA);
    }
    replaceN(mtDNA);
    int mtDNA_len = (int)strlen(mtDNA);
    //分配 nDNA 每個位置結果儲存陣列
    Cell *result_position = (Cell *)malloc(strlen(nDNA) * sizeof(Cell));
    
    printf("nDNA length = %zu\n", strlen(nDNA));
    printf("mtDNA length = %d\n", mtDNA_len);
    int nDNA_slice_len;

    //設定 threads 數量
    int threadsPerBlock = atoi(argv[3]);
    // 設定 C = 2R , 決定要使用的 blocks 數量
    // 根據 global memory 可配置最大空間 ，找尋最多可配置的 blocks 數(nDNA_slice_len 要整除 C)
    nDNA_slice_len = find_max_slice_len(mtDNA_len, prop.totalGlobalMem, threadsPerBlock);

    //分配 gpu 中每行最大值儲存空間
    Cell *col_max;
    ErrorCheck(cudaMalloc((Cell**) &col_max, sizeof(Cell) * nDNA_slice_len), __FILE__, __LINE__);

    int blocksPerGrid = nDNA_slice_len / (2 * threadsPerBlock);
    int writeBlocks = blocksPerGrid;

    printf("nDNA_slice_len = %d\n", nDNA_slice_len);
    printf("blocksPerGrid = %d\n", blocksPerGrid);
    printf("threadsPerBlocks = %d\n", threadsPerBlock);

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
    int *copyH = (int*)calloc((mtDNA_len + 1) * (nDNA_slice_len + 1), sizeof(int));
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (mtDNA_len + 1) * (nDNA_slice_len + 1)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(H, copyH, sizeof(int) * (mtDNA_len + 1) * (nDNA_slice_len + 1), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    free(copyH);
    // F
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * mtDNA_len), __FILE__, __LINE__);

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
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA, mtDNA_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //fill F vector with -∞
    int blocks = (mtDNA_len + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocks, threadsPerBlock>>>(F, 0, mtDNA_len);
    cudaDeviceSynchronize();

    // 計算外部對角線次數以及一個 block 負責的矩陣大小 R x C
    int outer_diag = blocksPerGrid + (mtDNA_len / threadsPerBlock);
    int R = threadsPerBlock;
    int C = nDNA_slice_len / blocksPerGrid;
    printf("R = %d, C = %d \n", R, C);
    printf("外部對角線共 : %d\n", outer_diag);

    // 開始計算
    start = clock();
    for(int epoch = 0; epoch <= strlen(nDNA) - nDNA_slice_len; epoch += nDNA_slice_len){
        printf("Epoch %d/%d\n", epoch/nDNA_slice_len + 1, (int)(strlen(nDNA)/nDNA_slice_len));
        //初始 col_max score 值為 0
        blocks = (nDNA_slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_cell_value<<<blocks, threadsPerBlock>>>(col_max, 0, nDNA_slice_len);
        cudaDeviceSynchronize();
        // copy nDNA slice 到 constant memory
        char* slice = substring(nDNA, epoch, nDNA_slice_len);
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, slice, nDNA_slice_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        fill_array_value<<<blocks, threadsPerBlock>>>(E, 0, nDNA_slice_len);
        cudaDeviceSynchronize();

        
        for(int i = 0; i < outer_diag; i++){
            //計算前半段
            cal_first_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, mtDNA_len, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len, col_max);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, mtDNA_len, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len, col_max);
            cudaDeviceSynchronize();
        }

        //計算剩餘的子矩陣
        if(mtDNA_len % threadsPerBlock != 0){
            int restThreadsPerBlock = mtDNA_len % threadsPerBlock;
            int start_row = (mtDNA_len / threadsPerBlock) * threadsPerBlock + 1; 
            for(int i = 0; i <= blocksPerGrid; i++){
                int iter = i == blocksPerGrid ? restThreadsPerBlock - 1 : C;
                do_rest_row <<<1, restThreadsPerBlock, restThreadsPerBlock * sizeof(Result)>>>(i, restThreadsPerBlock, start_row, nDNA_slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, (long long)epoch, nDNA_slice_len, col_max);
                cudaDeviceSynchronize();    
            }
        }
        

        //把 col_max 結果複製到result_position
        ErrorCheck(cudaMemcpy(result_position + epoch, col_max, nDNA_slice_len * sizeof(Cell), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

        //把H最後一行數值搬到第一行，提供下一個子矩陣計算
        blocks = ((mtDNA_len + threadsPerBlock - 1) / threadsPerBlock) + 1;
        move_data<<<blocks, threadsPerBlock>>>(H, mtDNA_len, nDNA_slice_len);
        cudaDeviceSynchronize();

        free(slice);
    }



    //計算剩下部份
    int rest_nDNA_len = strlen(nDNA) % nDNA_slice_len;
    
    if(rest_nDNA_len > 0){
        printf("nDNA 剩下長度 = %d\n", rest_nDNA_len);
        blocks = (nDNA_slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_cell_value<<<blocks, threadsPerBlock>>>(col_max, 0, nDNA_slice_len);
        cudaDeviceSynchronize();
        char *rest_slice = substring(nDNA, strlen(nDNA) - rest_nDNA_len, rest_nDNA_len);
        long long start_col = strlen(nDNA) - rest_nDNA_len;
        // copy nDNA slice 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, rest_slice, rest_nDNA_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        fill_array_value<<<blocks, threadsPerBlock>>>(E, 0, nDNA_slice_len);
        cudaDeviceSynchronize();

        // printf("新的 blocksPerGrid = %d\n", blocksPerGrid);
        // printf("新的 threadsPerBlock = %d\n", threadsPerBlock);
        // ////重新計算 outer_dig 、R 、C
        // outer_diag = blocksPerGrid + (mtDNA_len / threadsPerBlock);
        // R = threadsPerBlock;
        // printf("新的 R = %d, C = %d \n", R, C);
        // printf("外部對角線共 : %d\n", outer_diag);

        // 開始計算
        for(int i = 0; i < outer_diag; i++){
            //計算前半段
            cal_first_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, mtDNA_len, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len, col_max);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, R, C, nDNA_slice_len, mtDNA_len, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len, col_max);
            cudaDeviceSynchronize();
        }

        if(mtDNA_len % threadsPerBlock != 0){
            int restThreadsPerBlock = mtDNA_len % threadsPerBlock;
            int start_row = (mtDNA_len / threadsPerBlock) * threadsPerBlock + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGrid; i++){
                int iter = i == blocksPerGrid ? restThreadsPerBlock - 1 : C;
                do_rest_row <<<1, restThreadsPerBlock, restThreadsPerBlock * sizeof(Result)>>>(i, restThreadsPerBlock, start_row, nDNA_slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_nDNA_len, col_max);
                cudaDeviceSynchronize();    
            }
        }

        ErrorCheck(cudaMemcpy(result_position + strlen(nDNA) - rest_nDNA_len, col_max, rest_nDNA_len * sizeof(Cell), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        free(rest_slice);
    }

    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    double writeTime = elapsed_time;


    //打印最大值
    int maxScore = 0, maxI;
    long long maxJ;
    cudaMemcpy(&maxScore, global_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxI, global_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxJ, global_max_j, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("max score = %d at (%d, %lld)  \n", maxScore, maxI, maxJ);
    // 打印執行時間
    printf("total time: %.6f seconds\n", elapsed_time);
    convert_time(elapsed_time);
    save_experiment(argv[3], writeBlocks, writeTime, argv[4], maxScore, maxI, maxJ, argv[1], argv[2]);
    save_result_to_file(result_position, strlen(nDNA), argv[2], false, argv[4]);
    save_result_to_file(result_position, strlen(nDNA), argv[2], true, argv[4]);
    // 釋放 cpu 記憶體空間
    free(nDNA);
    free(mtDNA);
    free(panalty_score);
    free(result_position);

    // 釋放 gpu 記憶體空間
    cudaFree(col_max);
    cudaFree(H);
    cudaFree(E);
    cudaFree(F);
    cudaFree(global_max_score);
    cudaFree(global_max_i);
    cudaFree(global_max_j);


    //system("python3 ../cpu/draw.py");
    return 0;
}