#include "LargeScaleFunctions.h"
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// 初始化序列
void initialize_sequence(char **seq, long long len){
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
char* expand_mtDNA(char *mtDNA){
    int len = strlen(mtDNA);
    char *result = malloc(sizeof(char) * (2 * len + 1));

    // 複製前半段
    memcpy(result + 0 * len, mtDNA, len);
    // 複製後半段
    memcpy(result + 1 * len, mtDNA, len);

    result[2 * len] = '\0';

    return result;
}


void get_memory_info() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp == NULL) {
        perror("Failed to open /proc/meminfo");
        return;
    }

    char line[256];
    long total_memory = 0, available_memory = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line, "MemTotal: %ld kB", &total_memory);
        } else if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line, "MemAvailable: %ld kB", &available_memory);
        }
    }

    fclose(fp);

    printf("Total Memory: %.2f GB\n", total_memory / 1024.0 / 1024.0);
    printf("Available Memory: %.2f GB\n", available_memory / 1024.0 / 1024.0);
}


int* initialize_vector(long long len, int val){
    int *vector = malloc(sizeof(int) * len);
    for(long long i = 0; i < len; i++){
        vector[i] = val;
    }

    return vector;
}

//caculate local alignment
void local_alignment(int *H, int *E, char *reference, char *query, Result *result){
    for(int i = 1; i <= strlen(query); i++){
        //printf("%d / %ld\n", i, strlen(query));
        int dig_H = 0;
        int cur_H = 0;
        int F = INT_MIN - EXTEND_GAP;
        for(long long j = 1; j <= strlen(reference); j++){
            E[j] = MAX(E[j] + EXTEND_GAP , H[j] + OPEN_GAP);
            
            F = MAX(F + EXTEND_GAP , cur_H + OPEN_GAP);
            
            int match = (query[i-1] == reference[j-1])? MATCH : MISMATCH;

            cur_H = MAX(0, MAX((dig_H + match), MAX(E[j], F)));

            if(result->maxScore < cur_H){
                result->maxScore = cur_H;
                result->row = i;
                result->col = j;
            };

            dig_H = H[j];

            H[j] = cur_H;

            //printf(" %d ", cur_H);
        }
        //printf("\n");
    }
}

//秒數轉換
void convert_time(double total_seconds) {
    int days = (int)(total_seconds / (24 * 3600));       
    total_seconds = fmod(total_seconds, 24 * 3600);      

    int hours = (int)(total_seconds / 3600);             
    total_seconds = fmod(total_seconds, 3600);           

    int minutes = (int)(total_seconds / 60);             
    int seconds = (int)fmod(total_seconds, 60);         

    printf("Elapsed time: %d days, %d hours, %d minutes, %d seconds\n", days, hours, minutes, seconds);
}

// 讀字串
char* read_from_file(const char *filename) {
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
