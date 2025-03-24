#include "test.cuh"

int main(){
    clock_t start, end;
    // 設定由哪個 gpu 執行
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    // 讀取數據
    char *nDNA = read_from_file("../data/nDNA.txt");
    char *mtDNA = read_from_file("../data/mtDNA.txt");

    //test data
    char *nDNA_slice = substring(nDNA, 0, 78);
    char *mtDNA_slice = substring(mtDNA, 0, 43);

    printf("nDNA : %s\n", nDNA_slice);
    printf("mtDNA : %s\n", mtDNA_slice);
    int *result_position = (int *)calloc(strlen(nDNA_slice), sizeof(int));

    int slice_len = 24;

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

    // copyH
    int *copyH = (int*)calloc((strlen(mtDNA_slice) + 1) * (slice_len + 1), sizeof(int));

    
    // copyF
    int *copyF = (int*)malloc(sizeof(int) * strlen(mtDNA_slice));
    // fill_array(copyF, strlen(mtDNA_slice), INT_MIN - OPEN_GAP);
    // for(int i = 0; i < strlen(mtDNA_slice); i++){
    //     printf("%d  ", copyF[i]);
    // }
    // printf("\n");

    // copyE
    int *copyE = (int*)malloc(sizeof(int) * slice_len);
    // fill_array(copyE, slice_len, INT_MIN - OPEN_GAP);
    // for(int i = 0; i < slice_len; i++){
    //     printf("%d  ", copyE[i]);
    // }
    // printf("\n");

    //分配 GPU 空間
    ErrorCheck(cudaMemcpyToSymbol(panalty, panalty_score, 4 * sizeof(int)), __FILE__, __LINE__);
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * strlen(mtDNA_slice)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(F, copyF, sizeof(int) * strlen(mtDNA_slice), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *E;
    ErrorCheck(cudaMalloc((int**) &E, sizeof(int) * slice_len), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(E, copyE, sizeof(int) * slice_len, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (strlen(mtDNA_slice) + 1) * (slice_len + 1)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(H, copyH, sizeof(int) * (strlen(mtDNA_slice) + 1) * (slice_len + 1), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *global_max_score;
    ErrorCheck(cudaMalloc((int**)&global_max_score, sizeof(int)), __FILE__, __LINE__);
    ErrorCheck(cudaMemset(global_max_score, 0, sizeof(int)), __FILE__, __LINE__);
    int *global_max_i;
    ErrorCheck(cudaMalloc((int**)&global_max_i, sizeof(int)), __FILE__, __LINE__);
    int *global_max_j;
    ErrorCheck(cudaMalloc((int**)&global_max_j, sizeof(int)), __FILE__, __LINE__);
    // copy mtDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA_slice, strlen(mtDNA_slice) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //fill F vector with -∞
    int threadsPerBlock = 12;
    int blocksPerGrid = (strlen(mtDNA_slice) + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(F, INT_MIN - EXTEND_GAP + 1, strlen(mtDNA_slice));
    cudaDeviceSynchronize();


    //設定計算 blocks & threads 數
    int threadsPerBlockForSW = 4;
    int blocksPerGridForSW = 3;
    int outer_diag = blocksPerGridForSW + (strlen(mtDNA_slice) / threadsPerBlockForSW);
    int R = threadsPerBlockForSW;
    int C = slice_len / blocksPerGridForSW;
    printf("外部對角線共 : %d\n", outer_diag);

    //開始計算

    start = clock();
    for(int epoch = 0; epoch <= strlen(nDNA_slice) - slice_len; epoch += slice_len){
        printf("Epoch %d/%d\n", epoch/slice_len + 1, (int)(strlen(nDNA_slice)/slice_len));
        char* slice = substring(nDNA_slice, epoch, slice_len);
        // copy nDNA 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, slice, slice_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        threadsPerBlock = 12;
        blocksPerGrid = (slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(E, INT_MIN - EXTEND_GAP + 1, slice_len);
        cudaDeviceSynchronize();
        
        for(int i = 0; i < outer_diag; i++){
            printf("%d/%d\n", i, outer_diag-1);
            //計算前半段
            cal_first_phase<<<blocksPerGridForSW, threadsPerBlockForSW>>>(i, R, C, strlen(slice), strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j, (long long) epoch, strlen(slice));
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGridForSW, threadsPerBlockForSW>>>(i, R, C, strlen(slice), strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j, (long long) epoch, strlen(slice));
            cudaDeviceSynchronize();
        }

        if(strlen(mtDNA_slice) % threadsPerBlockForSW != 0){
            threadsPerBlock = strlen(mtDNA_slice) % threadsPerBlockForSW;
            int start_row = (strlen(mtDNA_slice) / threadsPerBlockForSW) * threadsPerBlockForSW + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGridForSW; i++){
                int iter = i == blocksPerGridForSW ? threadsPerBlock - 1 : C;
                do_rest_row <<<1, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, threadsPerBlock, start_row, slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, (long long) epoch, slice_len);
                cudaDeviceSynchronize();    
            }
        }
        
        check_matrix(copyH, H, mtDNA_slice, slice_len);
        // 把最後一行結果複製到 cpu 端存起來
        ErrorCheck(cudaMemcpy(result_position + epoch, H + (int)strlen(mtDNA_slice) * (slice_len + 1) + 1, slice_len * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

        //把H最後一行數值搬到第一行，提供下一個子矩陣計算
        threadsPerBlock = 13;
        blocksPerGrid = 4;
        move_data<<<blocksPerGrid, threadsPerBlock>>>(H, strlen(mtDNA_slice), slice_len);
        cudaDeviceSynchronize();

    }

    //計算剩餘的部份
    int rest_DNA_len = strlen(nDNA_slice) % slice_len;
    if(rest_DNA_len > 0){
        printf("nDNA 剩下的長度 = %d \n", rest_DNA_len);
        char * rest_slice = substring(nDNA_slice, strlen(nDNA_slice) - rest_DNA_len, rest_DNA_len);

        // copy nDNA 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, rest_slice, rest_DNA_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        threadsPerBlock = 12;
        blocksPerGrid = (slice_len + threadsPerBlock - 1) / threadsPerBlock;
        fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(E, INT_MIN - EXTEND_GAP + 1, slice_len);
        cudaDeviceSynchronize();

        //重新計算 outer_dig 、R 、C
        // blocksPerGridForSW = 4;
        // threadsPerBlockForSW = 2;
        // outer_diag = blocksPerGridForSW + (strlen(mtDNA_slice) / threadsPerBlockForSW);
        // R = threadsPerBlockForSW;
        // C = rest_DNA_len / blocksPerGridForSW;
        printf("外部對角線共 : %d\n", outer_diag);
        printf("新的 R = %d, C = %d\n", R, C);

        long long start_col = strlen(nDNA_slice) - rest_DNA_len;
        printf("start_col = %lld\n", start_col);
        //開始計算

        for(int i = 0; i < outer_diag; i++){
            printf("%d/%d\n", i, outer_diag-1);
            //計算前半段
            cal_first_phase<<<blocksPerGridForSW, threadsPerBlockForSW>>>(i, R, C, slice_len, strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_DNA_len);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGridForSW, threadsPerBlockForSW>>>(i, R, C, slice_len, strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_DNA_len);
            cudaDeviceSynchronize();
        }

        if(strlen(mtDNA_slice) % threadsPerBlockForSW != 0){
            threadsPerBlock = strlen(mtDNA_slice) % threadsPerBlockForSW;
            int start_row = (strlen(mtDNA_slice) / threadsPerBlockForSW) * threadsPerBlockForSW + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGridForSW; i++){
                int iter = i == blocksPerGridForSW ? threadsPerBlock - 1 : C;
                do_rest_row <<<1, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, threadsPerBlock, start_row, slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j, start_col, rest_DNA_len);
                cudaDeviceSynchronize();    
            }
        }

        check_matrix(copyH, H, mtDNA_slice, slice_len);

        ErrorCheck(cudaMemcpy(result_position + strlen(nDNA_slice) - rest_DNA_len, H + (int)strlen(mtDNA_slice) * (slice_len + 1) + 1, rest_DNA_len * sizeof(int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        
    }

    //printArray(result_position, strlen(nDNA_slice));
    
    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）

    //save_result_to_file(result_position, strlen(nDNA_slice), "../output/test.txt");

    int maxScore, maxI, maxJ;
    cudaMemcpy(&maxScore, global_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxI, global_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxJ, global_max_j, sizeof(int), cudaMemcpyDeviceToHost);
    printf("max score = %d at (%d, %d)  \n", maxScore, maxI, maxJ);

    free(nDNA);
    free(mtDNA);
    free(nDNA_slice);
    free(mtDNA_slice);
    free(copyH);
    free(copyE);
    free(copyF);

    cudaFree(H);
    cudaFree(E);
    cudaFree(F);

    // 打印執行時間
    printf("total time: %.6f seconds\n", elapsed_time);
    convert_time(elapsed_time);

    //system("python3 ../cpu/draw.py");

}