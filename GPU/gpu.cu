#include "functions.cuh"


int main(){
    clock_t start, end;
    // 設定由哪個 gpu 執行
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp prop;
    ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

    //各種參數設定
    int blocksPerGrid = 40;
    int nDNA_slice_len;

    // 尋找最大的 nDNA_slice 長度且可以被 blocksPerGrid 整除
    nDNA_slice_len = find_max_slice_len(MTDNA_LEN, prop.totalGlobalMem, blocksPerGrid);
    printf("nDNA_slice_len = %d\n", nDNA_slice_len);

    // 讀取數據
    char *chromosome2 = read_from_file("../data/GRCh38.chromosome2.txt");
    char *nc1 = read_from_file("../data/NC_012920.1.txt");
    printf("chromosome2 length = %zu\n", strlen(chromosome2));
    free(chromosome2);
    printf("NC_012920.1 length = %zu \n", strlen(nc1));
    free(nc1);
    char *nDNA = read_from_file("../data/nDNA.txt");
    char *mtDNA = read_from_file("../data/mtDNA.txt");
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
    int blocks = (MTDNA_LEN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_array_value<<<blocks, THREADS_PER_BLOCK>>>(F, INT_MIN - EXTEND_GAP + 1, MTDNA_LEN);
    cudaDeviceSynchronize();

    // 計算外部對角線次數以及一個 block 負責的矩陣大小 R x C
    int outer_diag = blocksPerGrid + (MTDNA_LEN / THREADS_PER_BLOCK);
    int R = THREADS_PER_BLOCK;
    int C = nDNA_slice_len / blocksPerGrid;
    printf("R = %d, C = %d \n", R, C);
    printf("外部對角線共 : %d\n", outer_diag);

    // 開始計算
    start = clock();
    for(int epoch = 0; epoch <= strlen(nDNA) - nDNA_slice_len; epoch += nDNA_slice_len){
        //if(epoch/nDNA_slice_len + 1 == 4) break;
        printf("Epoch %d/%d\n", epoch/nDNA_slice_len + 1, (int)(strlen(nDNA)/nDNA_slice_len));
        char* slice = substring(nDNA, epoch, nDNA_slice_len);
        // copy nDNA 到 constant memory
        ErrorCheck(cudaMemcpyToSymbol(device_nDNA, slice, nDNA_slice_len + 1, 0, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        //fill E vector with -∞
        blocks = (nDNA_slice_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        fill_array_value<<<blocks, THREADS_PER_BLOCK>>>(E, INT_MIN - EXTEND_GAP + 1, nDNA_slice_len);
        cudaDeviceSynchronize();

        
        for(int i = 0; i < outer_diag; i++){
            //printf("%d/%d\n", i, outer_diag);
            //計算前半段
            cal_first_phase<<<blocksPerGrid, THREADS_PER_BLOCK>>>(i, R, C, strlen(slice), MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j);
            cudaDeviceSynchronize();
            //計算後半段
            cal_second_phase<<<blocksPerGrid, THREADS_PER_BLOCK>>>(i, R, C, strlen(slice), MTDNA_LEN, E, F, H, global_max_score, global_max_i, global_max_j);
            cudaDeviceSynchronize();
        }

        if(MTDNA_LEN % THREADS_PER_BLOCK != 0){
            int threadsPerBlock = MTDNA_LEN % THREADS_PER_BLOCK;
            int start_row = (MTDNA_LEN / THREADS_PER_BLOCK) * THREADS_PER_BLOCK + 1; 
            //計算剩餘的子矩陣
            for(int i = 0; i <= blocksPerGrid; i++){
                int iter = i == blocksPerGrid ? threadsPerBlock - 1 : C;
                do_rest_row <<<1, threadsPerBlock, threadsPerBlock * sizeof(Result)>>>(i, threadsPerBlock, start_row, nDNA_slice_len, iter, C, E, F, H, global_max_score, global_max_i, global_max_j);
                cudaDeviceSynchronize();    
            }
        }
        

        //把H最後一行數值搬到第一行，提供下一個子矩陣計算
        blocks = ((MTDNA_LEN + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) + 1;
        move_data<<<blocks, THREADS_PER_BLOCK>>>(H, MTDNA_LEN, nDNA_slice_len);
        cudaDeviceSynchronize();
    }
    end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;  // 计算耗时（秒）

    //打印最大值
    int maxScore, maxI;
    long long maxJ;
    cudaMemcpy(&maxScore, global_max_score, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxI, global_max_i, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxJ, global_max_j, sizeof(long), cudaMemcpyDeviceToHost);
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