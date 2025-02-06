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
    char *nDNA_slice = substring(nDNA, 0, 24);
    char *mtDNA_slice = substring(mtDNA, 0, 39);

    printf("nDNA : %s\n", nDNA_slice);
    printf("mtDNA : %s\n", mtDNA_slice);

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
    int *copyH = (int*)malloc((strlen(mtDNA_slice) + 1) * (strlen(nDNA_slice) + 1) * sizeof(int));
    fill_array(copyH, (strlen(mtDNA_slice) + 1) * (strlen(nDNA_slice) + 1), 0);

    
    // copyF
    int *copyF = (int*)malloc(sizeof(int) * strlen(mtDNA_slice));
    // fill_array(copyF, strlen(mtDNA_slice), INT_MIN - OPEN_GAP);
    // for(int i = 0; i < strlen(mtDNA_slice); i++){
    //     printf("%d  ", copyF[i]);
    // }
    // printf("\n");

    // copyE
    int *copyE = (int*)malloc(sizeof(int) * strlen(nDNA_slice));
    // fill_array(copyE, strlen(nDNA_slice), INT_MIN - OPEN_GAP);
    // for(int i = 0; i < strlen(nDNA_slice); i++){
    //     printf("%d  ", copyE[i]);
    // }
    // printf("\n");

    //分配 GPU 空間
    ErrorCheck(cudaMemcpyToSymbol(panalty, panalty_score, 4 * sizeof(int)), __FILE__, __LINE__);
    int *F;
    ErrorCheck(cudaMalloc((int**) &F, sizeof(int) * strlen(mtDNA_slice)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(F, copyF, sizeof(int) * strlen(mtDNA_slice), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *E;
    ErrorCheck(cudaMalloc((int**) &E, sizeof(int) * strlen(nDNA_slice)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(E, copyE, sizeof(int) * strlen(nDNA_slice), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *H;
    ErrorCheck(cudaMalloc((int**)&H, sizeof(int) * (strlen(mtDNA_slice) + 1) * (strlen(nDNA_slice) + 1)), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(H, copyH, sizeof(int) * (strlen(mtDNA_slice) + 1) * (strlen(nDNA_slice) + 1), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    int *global_max_score;
    ErrorCheck(cudaMalloc((int**)&global_max_score, sizeof(int)), __FILE__, __LINE__);
    ErrorCheck(cudaMemset(global_max_score, 0, sizeof(int)), __FILE__, __LINE__);
    int *global_max_i;
    ErrorCheck(cudaMalloc((int**)&global_max_i, sizeof(int)), __FILE__, __LINE__);
    int *global_max_j;
    ErrorCheck(cudaMalloc((int**)&global_max_j, sizeof(int)), __FILE__, __LINE__);
    // copy mtDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_mtDNA, mtDNA_slice, strlen(mtDNA_slice) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    // copy nDNA 到 constant memory
    ErrorCheck(cudaMemcpyToSymbol(device_nDNA, nDNA_slice, strlen(nDNA_slice) + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    //fill F vector with -∞
    int threadsPerBlock = 12;
    int blocksPerGrid = (strlen(mtDNA_slice) + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(F, INT_MIN - OPEN_GAP, strlen(mtDNA_slice));
    cudaDeviceSynchronize();


    //fill E vector with -∞
    threadsPerBlock = 12;
    blocksPerGrid = (strlen(nDNA_slice) + threadsPerBlock - 1) / threadsPerBlock;
    fill_array_value<<<blocksPerGrid, threadsPerBlock>>>(E, INT_MIN - OPEN_GAP, strlen(nDNA_slice));
    cudaDeviceSynchronize();

    //計算外部對角線次數
    threadsPerBlock = 3;
    blocksPerGrid = 4;
    int outer_diag = blocksPerGrid + (strlen(mtDNA_slice) / threadsPerBlock);
    int R = threadsPerBlock;
    int C = strlen(nDNA_slice) / blocksPerGrid;
    printf("外部對角線共 : %d\n", outer_diag);

    //開始計算
    start = clock();
    for(int i = 0; i < outer_diag; i++){
        printf("%d/%d\n", i, outer_diag);
        //計算前半段
        cal_first_phase<<<blocksPerGrid, threadsPerBlock>>>(i, R, C, strlen(nDNA_slice), strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j);
        cudaDeviceSynchronize();
        //計算後半段
        cal_second_phase<<<blocksPerGrid, threadsPerBlock>>>(i, R, C, strlen(nDNA_slice), strlen(mtDNA_slice), E, F, H, global_max_score, global_max_i, global_max_j);
        cudaDeviceSynchronize();
    }
    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）

    int maxScore, maxI, maxJ;
    cudaMemcpy(&maxScore, global_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxI, global_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxJ, global_max_j, sizeof(int), cudaMemcpyDeviceToHost);
    printf("max score = %d at (%d, %d)  \n", maxScore, maxI, maxJ);


    // 查看結果矩陣
    // ErrorCheck(cudaMemcpy(copyH, H, sizeof(int) * (strlen(mtDNA_slice) + 1) * (strlen(nDNA_slice) + 1), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    // for(int i = 0; i < (strlen(mtDNA_slice) + 1); i++){
    //     for(int j = 0; j < (strlen(nDNA_slice) + 1); j++){
    //         printf("%d  ", copyH[i * (strlen(nDNA_slice) + 1) + j]);
    //     }
    //     printf("\n");
    // }

    // 打印執行時間
    printf("total time: %.6f seconds\n", elapsed_time);
    convert_time(elapsed_time);

}