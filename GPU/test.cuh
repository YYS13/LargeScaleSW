#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MATCH 2
#define MISMATCH -3
#define OPEN_GAP -5
#define EXTEND_GAP -2
#define SCORE_SIZE 4



#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) >= (b) ? (b) : (a))

struct Result{
    int score;
    int i;
    int j;
};


__constant__ char device_mtDNA[53];
__constant__ char device_nDNA[26];
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

void fill_array(int *array, long long size ,int val){
    for(int i = 0; i < size; i++){
        array[i] = val;
    }
}

void convert_time(double total_seconds) {
    int days = (int)(total_seconds / (24 * 3600));       
    total_seconds = fmod(total_seconds, 24 * 3600);      

    int hours = (int)(total_seconds / 3600);             
    total_seconds = fmod(total_seconds, 3600);           

    int minutes = (int)(total_seconds / 60);             
    int seconds = (int)fmod(total_seconds, 60);         

    printf("Elapsed time: %d days, %d hours, %d minutes, %d seconds\n", days, hours, minutes, seconds);
}

// 查看結果矩陣
void check_matrix(int *copyH, int *H, char *mtDNA_slice, int slice_len){
    ErrorCheck(cudaMemcpy(copyH, H, sizeof(int) * (strlen(mtDNA_slice) + 1) * (slice_len + 1), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    for(int i = 0; i < (strlen(mtDNA_slice) + 1); i++){
        for(int j = 0; j < (slice_len + 1); j++){
            printf("%d  ", copyH[i * (slice_len + 1) + j]);
        }
        printf("\n");
    }
}

void printArray(int *array, size_t size){
    printf("[");
    for(long long i = 0; i < (long long)size; i++){
        printf("%d, ", array[i]);
    }
    printf("]\n");
}

int save_result_to_file(int* array, size_t size, const char *filename){
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("fopen");
        return -1;  // 開檔失敗
    }

    for(long long i = 0; i < (long long) size; i++){
        double log;
        if(array[i] != 0){
            log = log2((double)array[i]);
        }else{
            log = 0.0;
        }
        fprintf(fp, "%.1f\n", log);
    }

    fclose(fp);

    return 0;
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

/**
 * Dk : 第幾輪外部對角線
 * blockId : 第幾個 block
 * R : 一個 block 計算子矩陣的高
 * C : 一個 block 計算子矩陣的寬
 * i : 該 block 負責的子矩陣第一格列位置 
 * j : 該 block 負責的子矩陣第一格行位置
 */
__device__ inline void getBlockCoordinate(int Dk, int blockId, int threadId, int R, int C, int *i, int *j){
    
    // 取得該 block 座標
    int block_i = Dk - blockId;
    int block_j = blockId;

    *i = (block_i * R) + threadId + 1;
    *j = block_j * C - threadId + 1;

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

__global__ void move_data(int *H, int mtDNA_len, int slice_len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= 1 && tid <= mtDNA_len){
        H[tid * (slice_len + 1)] = H[tid * (slice_len + 1) + slice_len];
    }
}
/*
    outer_dig : 第幾輪外部對角線
    C : 一個 submatrix 大小為 R x C

 */


__global__ void cal_first_phase(int outer_dig, int R,  int C, int slice_len, int mtDNA_len, int *E, int *F, int *H, int *global_max_score, int *global_max_i, int *global_max_j, long long start_col, int rest_len, int *col_max){
    //放比對分數
    __shared__ int shared_panalty[SCORE_SIZE];
    //紀錄 block 內最大值
    __shared__ int block_max_score[4]; // Block 內各tid最大值
    __shared__ int block_max_i[4];     // Block 內各tid最大值對應的 i
    __shared__ int block_max_j[4];     // Block 內各tid最大值對應的 j

    //把 penalty score 搬到 shared memory 中
    if(threadIdx.x == 0){
        for(int i = 0; i < 4; i++){
            shared_panalty[i] = panalty[i];
            block_max_score[i] = 0;
            block_max_i[i] = 0;
            block_max_j[i] = 0;
        }
    }
    __syncthreads();

    int i, j; //計算各 thread 負責的起始位置
    getBlockCoordinate(outer_dig, blockIdx.x, threadIdx.x, R, C, &i, &j);

    if(j < 1){
        i = i - gridDim.x * R;
        j = rest_len + j;
    }



    //開始計算
    for(int times = 0; times < R; times++){
        // if(blockIdx.x == 0){
        //     printf("tid = %d, round = %d, cell = (%d, %d)\n", threadIdx.x, times+1, i, j);
        // }
        if(i >= 1 && i <= (mtDNA_len / blockDim.x) * blockDim.x && j <= rest_len){
            
            E[j-1] = max2(E[j-1] + shared_panalty[2], H[(i-1) * (slice_len + 1) + j] + shared_panalty[3]);

            F[i-1] = max2(F[i-1] + shared_panalty[2], H[i * (slice_len + 1) + (j-1)] + shared_panalty[3]);
            
            int match = device_mtDNA[i - 1] == device_nDNA[j - 1] ? shared_panalty[0] : shared_panalty[1];


            int curV = max4(E[j-1], F[i-1], H[(i - 1) * (slice_len + 1) + (j - 1)] + match, 0);

            if(curV > col_max[j-1]){
                col_max[j-1] = curV;
            }

            H[i * (slice_len + 1) + j] = curV;


            //儲存 H[i][j] 和其位置到 shared memory 中
            block_max_score[threadIdx.x] = curV;
            block_max_i[threadIdx.x] = i;
            block_max_j[threadIdx.x] = j;
        }

        j++;
        // 是否結束 cell delegation 回去原本的位置繼續做
        if(j == slice_len + 1){
            j = 1;
            i = i + gridDim.x * R; 
        }

        __syncthreads(); //等待大家都把該輪內部對角線都寫入 shared memory 中

        //由 tid = 0 的 thread 檢查 shared memory 中最大值與其位置
        if(threadIdx.x == 0){
            int max_score = 0, max_i = -1, max_j = -1;
            
            for (int k = 0; k < R; k++) {
                if (block_max_score[k] > max_score) {
                    max_score = block_max_score[k];
                    max_i = block_max_i[k];
                    max_j = block_max_j[k];
                }
            }
            

            //使用原子操作（atomicMax）更新全局最大得分
            atomicMax(global_max_score, max_score);
            if (*global_max_score == max_score) {
                *global_max_i = max_i;
                *global_max_j = start_col + max_j;
            }
        
        }

        __syncthreads(); //等待 tid0 檢查最大值

    }


}

__global__ void cal_second_phase(int outer_dig, int R,  int C, int slice_len, int mtDNA_len, int *E, int *F, int *H, int *global_max_score, int *global_max_i, int *global_max_j, long long start_col, int rest_len, int *col_max){
    //放比對分數
    __shared__ int shared_panalty[4];
    //紀錄 block 內最大值
    __shared__ int block_max_score[4]; // Block 內各tid最大值
    __shared__ int block_max_i[4];     // Block 內各tid最大值對應的 i
    __shared__ int block_max_j[4];     // Block 內各tid最大值對應的 j
    
    //把 penalty score 搬到 shared memory 中
    if(threadIdx.x == 0){
        for(int i = 0; i < 4; i++){
            shared_panalty[i] = panalty[i];
            block_max_score[i] = 0;
            block_max_i[i] = 0;
            block_max_j[i] = 0;
        }
    }

    __syncthreads();


    int i, j; //計算各 thread 負責的起始位置
    getBlockCoordinate(outer_dig, blockIdx.x, threadIdx.x, R, C, &i, &j);

    j = j + R;

    //開始計算
    for(int times = 0; times < (C - R); times++){
        if(i >= 1 && i <= (mtDNA_len / blockDim.x) * blockDim.x && j <= rest_len){
            E[j-1] = max2(E[j-1] + shared_panalty[2], H[(i-1) * (slice_len + 1) + j] + shared_panalty[3]);

            F[i-1] = max2(F[i-1] + shared_panalty[2], H[i * (slice_len + 1) + (j-1)] + shared_panalty[3]);

            int match = device_mtDNA[i - 1] == device_nDNA[j - 1] ? shared_panalty[0] : shared_panalty[1];

            int curV = max4(E[j-1], F[i-1], H[(i - 1) * (slice_len + 1) + (j - 1)] + match, 0);

            if(curV > col_max[j-1]){
                col_max[j-1] = curV;
            }
            //printf("H[i-1][j-1] + match = %d", H[(i - 1) * (slice_len + 1) + (j - 1)]);
            H[i * (slice_len + 1) + j] = curV;
            //if(i == 2 && j== 23) printf("H[2][23] = %d\n", H[i * (slice_len + 1) + j]);

            //儲存 H[i][j] 和其位置到 shared memory 中
            block_max_score[threadIdx.x] = curV;
            block_max_i[threadIdx.x] = i;
            block_max_j[threadIdx.x] = j;
        }
        j++;

        __syncthreads(); //等待大家都把該輪內部對角線都寫入 shared memory 中

        //由 tid = 0 的 thread 檢查 shared memory 中最大值與其位置
        if(threadIdx.x == 0){
            int max_score = 0, max_i = -1, max_j = -1;
            
            for (int k = 0; k < R; k++) {
                if (block_max_score[k] > max_score) {
                    max_score = block_max_score[k];
                    max_i = block_max_i[k];
                    max_j = block_max_j[k];
                }
            }
            

            //使用原子操作（atomicMax）更新全局最大得分
            atomicMax(global_max_score, max_score);
            if (*global_max_score == max_score) {
                *global_max_i = max_i;
                *global_max_j = start_col + max_j;
            }
        
        }

        __syncthreads(); //等待 tid0 檢查最大值

    }
}

__global__ void do_rest_row(int round, int restRows, int start_row, int slice_len, int iter, int C, int *E, int *F, int *H, int *global_max_score, int *global_max_i, int *global_max_j, long long start_col, int rest_len, int *col_max){
    //放比對分數
    __shared__ int shared_panalty[4];
    //紀錄 block 內最大值
    extern __shared__ Result cellValues[]; 
    
    //把 penalty score 搬到 shared memory 中
    if(threadIdx.x == 0){
        for(int i = 0; i < 4; i++){
            shared_panalty[i] = panalty[i];
        }
    }
    cellValues[threadIdx.x].score = 0;
    cellValues[threadIdx.x].i = 0;
    cellValues[threadIdx.x].j = 0;

    __syncthreads();

    int i = start_row + threadIdx.x;
    int j = round * C - threadIdx.x + 1;
 

    //計算
    for(int times = 0; times < iter; times++){
        //if(round == 3) printf("shift = %d, tid = %d, deal(%d, %d)\n", times, threadIdx.x, i, j);
        if(j >= 1 && j <= rest_len){
            E[j-1] = max2(E[j-1] + shared_panalty[2], H[(i-1) * (slice_len + 1) + j] + shared_panalty[3]);

            F[i-1] = max2(F[i-1] + shared_panalty[2], H[i * (slice_len + 1) + (j-1)] + shared_panalty[3]);
            
            int match = device_mtDNA[i - 1] == device_nDNA[j - 1] ? shared_panalty[0] : shared_panalty[1];


            int curV = max4(E[j-1], F[i-1], H[(i - 1) * (slice_len + 1) + (j - 1)] + match, 0);
            if(curV > col_max[j-1]){
                col_max[j-1] = curV;
            }

            H[i * (slice_len + 1) + j] = curV;


            //儲存 H[i][j] 和其位置到 shared memory 中
            cellValues[threadIdx.x].score = curV;
            cellValues[threadIdx.x].i = i;
            cellValues[threadIdx.x].j = j;
        }

        j++;

        __syncthreads();

        //由 tid = 0 的 thread 檢查 shared memory 中最大值與其位置
        if(threadIdx.x == 0){
            int max_score = 0, max_i = -1, max_j = -1;
            
            for (int k = 0; k < restRows; k++) {
                if (cellValues[k].score > max_score) {
                    max_score = cellValues[k].score;
                    max_i = cellValues[k].i;
                    max_j = cellValues[k].j;
                }
            }
            

            //使用原子操作（atomicMax）更新全局最大得分
            atomicMax(global_max_score, max_score);
            if (*global_max_score == max_score) {
                *global_max_i = max_i;
                *global_max_j = start_col + max_j;
            }
        
        }

        __syncthreads(); //等待 tid0 檢查最大值
    }
}
