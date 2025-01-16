#pragma once
#include <stdio.h>
#include <stdlib.h>


#define MATCH 3
#define MISMATCH -1
#define OPEN_GAP -5
#define EXTEND_GAP -1

__constant__ char device_mtDNA[33139];
__constant__ char device_slice_nDNA[6401];

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

// global
__global__ void fill_array_value(int *array, int value, size_t n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        array[idx] = value;
    }
}

__global__ void proint_arrInfo(int *array, int idx){
    printf("F[%d] : %d\n", idx,  array[idx]);
}

// device