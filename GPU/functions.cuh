#pragma once
#include <stdio.h>
#include <stdlib.h>

#define MATCH 3
#define MISMATCH -1
#define OPEN_GAP -5
#define EXTEND_GAP -1


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) >= (b) ? (b) : (a))


__constant__ char device_mtDNA[33139];
__constant__ char device_slice_nDNA[6401];
__constant__ int panalty[4];

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
                error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

// host (只在cpu執行)

// 初始化序列
__host__ void initialize_sequence(char **seq, long long len){
    //鹼基集
    const char baseset[] = "ACGT";

    //分配空間給sequence
    *seq = (char*)malloc((len + 1) * sizeof(char));

    //隨機生成 seq
    for(long long i = 0; i <= len; i++){
        (*seq)[i] = baseset[rand() % 4];
    }

    //最後一個字設中止符號
    (*seq)[len] = '\0';
}

//將長度為 n 的 mtDNA 複製一份
__host__ char* expand_mtDNA(char *mtDNA){
    int len = strlen(mtDNA);
    char *result = (char *)malloc(sizeof(char) * (2 * len + 1));

    // 複製前半段
    memcpy(result + 0 * len, mtDNA, len);
    // 複製後半段
    memcpy(result + 1 * len, mtDNA, len);

    result[2 * len] = '\0';

    return result;
}

// 保存字串到文件
__host__ void save_to_file(const char *filename, const char *data) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%s", data);
    fclose(file);
    printf("Data saved to %s\n", filename);
}

// 讀字串
__host__ char* read_from_file(const char *filename) {
    FILE *file = fopen(filename, "r");  
    if (file == NULL) {
        perror("Failed to open file for reading");
        exit(EXIT_FAILURE);
    }

    
    fseek(file, 0, SEEK_END); //file 指標一到最後 
    size_t file_size = ftell(file);  // 目前位置
    rewind(file);  // file 指標回到開頭

    char *data = (char *)malloc((file_size + 1) * sizeof(char));
    if (data == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // 读取文件内容到字符串
    fread(data, sizeof(char), file_size, file);
    data[file_size] = '\0';  // 添加字符串终止符

    fclose(file);  // 关闭文件
    return data;
}

// 取 substring
char* substring(const char* str, size_t start, size_t length) {
    size_t str_len = strlen(str);

    // 防止亂丟參數
    if (start >= str_len) {
        return strdup("");  // 返回空字符串
    }

    // start + length 不能超過最大 idx
    if (start + length > str_len) {
        length = str_len - start;
    }

    // 分配空間（包含 '\0'）
    char* result = (char*)malloc((length + 1) * sizeof(char));
    if (result == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    // 複製字串
    strncpy(result, str + start, length);
    result[length] = '\0';  // 添加终止符

    return result;
}

// device

__device__ int max2(int a, int b) {
    return (a>b)?a:b;
}

/**
 * Returns the maximum of four numbers.
 */
__device__ int max4(int a, int b, int c, int d) {
    return max2(max2(a,b), max2(c,d));
}


// global
//幫 vector 填值
__global__ void fill_array_value(int *array, int value, size_t n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        array[idx] = value;
    }
}

// 打印 vector 內容(用來檢查 fill_array_value 有無成功)
__global__ void proint_arrInfo(int *array, int idx){
    printf("H[%d] : %d\n", idx,  array[idx]);
}

// 初始化 H (第一行第一列皆為 0);
__global__ void initializeH(int *H) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < 6401){
        H[tid] = 0;
        H[tid * 6401] = 0;
    }
    if(tid < 33139){
        H[tid * 6401] = 0;
    }
}

// submatrix 算完把最後一行資料搬到第一行
__global__ void move_data(int *H, int mtDNA_len, int nDNA_slice_len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > 0 && tid < mtDNA_len + 1){
        H[tid * (nDNA_slice_len + 1)] = H[tid * (nDNA_slice_len + 1) + nDNA_slice_len];
    }
}
/*
    outer_dig : 第幾輪外部對角線
    C : 一個 submatrix 大小為 R x C

 */


__global__ void cal_first_phase(int outer_dig, int R,  int C, int slice_len, int mtDNA_len, int *E, int *F, int *H, int *global_max_score, int *global_max_i, int *global_max_j){
    //放比對分數
    __shared__ int shared_panalty[4];
    //紀錄 block 內最大值
    __shared__ int block_max_score[263]; // Block 內各tid最大值
    __shared__ int block_max_i[263];     // Block 內各tid最大值對應的 i
    __shared__ int block_max_j[263];     // Block 內各tid最大值對應的 j
    if(threadIdx.x == 0){
        for(int i = 0; i < 4; i++){
            shared_panalty[i] = panalty[i];
        }
    }

    // 計算 block 座標
    int block_x = outer_dig - blockIdx.x;
    int block_y = blockIdx.x;
    if(threadIdx.x == 55){
        printf("Block Position = (%d, %d) \n", block_x, block_y);
    }

    // 計算 thread 處理的位置
    int row = block_x * blockDim.x + 1;
    int col = block_y * C - threadIdx.x + 1;
    
    if(threadIdx.x == 55){
        printf("Cell Position = (%d, %d) \n", row, col);
    }
    //最左邊的 block 需計算 cell delegation 的內容
    if(col <= 0){
        row = row - blockDim.x * R;
        col = slice_len + col;
    }

    //開始計算
    //for(int i = 0; i < (C - R); i++){
    //     if(row > 0 || row <= mtDNA_len){
             //計算
             //E
    //         E[col-1] = max2(E[col-1] + shared_panalty[2], H[(row - 1) * (slice_len + 1) + col] + shared_panalty[3]);
             //F
             //F[row-1] = max2(F[row-1] + shared_panalty[2], H[row * (slice_len + 1) + (col - 1)] + shared_panalty[3]);
    //         //H[row][col];
    //         int match = device_mtDNA[row - 1] == device_slice_nDNA[col - 1] ? shared_panalty[0] : shared_panalty[1];

    //         int curV = max4(E[col], F[row], H[(row - 1) * (slice_len + 1) + (col - 1)] + match, 0);
    //         H[row * (slice_len + 1) + col] = curV;

    //         //store cell value & position in shared memory
    //         block_max_score[threadIdx.x] = curV;
    //         block_max_i[threadIdx.x] = row;
    //         block_max_j[threadIdx.x] = col; 


    //     }

    //     col++;
    //     if(col == slice_len){
    //         col = 0;
    //         row = row + blockDim.x * R;
    //     }

    //     __syncthreads();
    //     //check max value in block 
    //     if(threadIdx.x == 0){
    //         int max_score = 0, max_i = -1, max_j = -1;
    //     for (int k = 0; k < R; k++) {
    //         if (block_max_score[k] > max_score) {
    //             max_score = block_max_score[k];
    //             max_i = block_max_i[k];
    //             max_j = block_max_j[k];
    //         }
    //     }

    //     // 使用原子操作（atomicMax）更新全局最大得分
    //     atomicMax(global_max_score, max_score);
    //         if (*global_max_score == max_score) {
    //             *global_max_i = max_i;
    //             *global_max_j = max_j;
    //         }
    //     }

    //     __syncthreads();
    //}



}

__global__ void cal_second_phase(int outer_dig, int R,  int C, int slice_len, int mtDNA_len, int *E, int *F, int *H, int *global_max_score, int *global_max_i, int *global_max_j){
    //放比對分數
    __shared__ int shared_panalty[4];
    //紀錄 block 內最大值
    __shared__ int block_max_score[263]; // Block 內各tid最大值
    __shared__ int block_max_i[263];     // Block 內各tid最大值對應的 i
    __shared__ int block_max_j[263];     // Block 內各tid最大值對應的 j
    
    if(threadIdx.x == 0){
        for(int i = 0; i < 4; i++){
            shared_panalty[i] = panalty[i];
        }
    }
    
    // 計算 block 座標
    int block_x = outer_dig - blockIdx.x;
    int block_y = blockIdx.x;

    // 計算 thread 處理的位置
    int row = block_x * blockDim.x + 1;
    int col = block_y * C - threadIdx.x + 1 + (C - R);


    //開始計算
    for(int i = 0; i < R; i++){
        if(row > 0 || row <= mtDNA_len){
            //計算
            //E
            E[col] = max2(E[col] + shared_panalty[2], H[(row - 1) * (slice_len + 1) + col] + shared_panalty[3]);
            //F
            F[row] = max2(F[row] + shared_panalty[2], H[row * (slice_len + 1) + (col - 1)] + shared_panalty[3]);
            //H[row][col];
            int match = device_mtDNA[row - 1] == device_slice_nDNA[col - 1] ? shared_panalty[0] : shared_panalty[1];

            int curV = max4(E[col], F[row], H[(row - 1) * (slice_len + 1) + (col - 1)] + match, 0);
            H[row * (slice_len + 1) + col] = curV;

            //store cell value & position in shared memory
            block_max_score[threadIdx.x] = curV;
            block_max_i[threadIdx.x] = row;
            block_max_j[threadIdx.x] = col; 

        }

        col++;

        __syncthreads();

        //check max value in block 
        if(threadIdx.x == 0){
            int max_score = 0, max_i = -1, max_j = -1;
        for (int k = 0; k < R; k++) {
            if (block_max_score[k] > max_score) {
                max_score = block_max_score[k];
                max_i = block_max_i[k];
                max_j = block_max_j[k];
            }
        }

        // 使用原子操作（atomicMax）更新全局最大得分
        atomicMax(global_max_score, max_score);
            if (*global_max_score == max_score) {
                *global_max_i = max_i;
                *global_max_j = max_j;
            }
        }

        __syncthreads();
    }
}
