#include "functions.h"
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int max2(int a, int b) {
    return (a>b)?a:b;
}

/**
 * Returns the maximum of four numbers.
 */
int max4(int a, int b, int c, int d) {
    return max2(max2(a,b), max2(c,d));
}

void replaceN(char *sequence){
    if(sequence == NULL) return;

    char bases[] = "ACGT";
    srand(time(NULL));  

    for (int i = 0; sequence[i] != '\0'; i++) {
        if (sequence[i] == 'N') {
            sequence[i] = bases[rand() % 4];  
        }
    }


}

void initialize_sequence(char **seq, int len){
    //鹼基集
    const char baseset[] = "ACGT";

    //分配空間給sequence
    *seq = (char*)malloc((len + 1) * sizeof(char));

    //隨機生成 seq
    for(int i = 0; i <= len; i++){
        (*seq)[i] = baseset[rand() % 4];
    }

    //最後一個字設中止符號
    (*seq)[len] = '\0';
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


int** initialize_matrix(int m, int n){
    // 分配空間
    int **matrix = (int**)malloc((m + 1) * sizeof(int*));
    for(int i = 0; i <= m; i++){
        matrix[i] = (int*)malloc((n + 1) * sizeof(int));
    }
    matrix[0][0] = 0;
    // initialize first row & first col to zero
    for(int i = 1; i <= m; i++){
        matrix[i][0] = 0;
    }

    for(int j = 1; j <= n; j++){
        matrix[0][j] = 0;
    }
   

    return matrix;
}

void fill_vector(int *vector, int value, int length){
    for(int i = 0; i < length; i++){
        vector[i] = value;
    }
}


void reverseString(char *str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = temp;
    }
}

void local_alignment(int **H, int *E, int *F, char *reference, char *query, Result *result, long long start_col){
    int m = strlen(query);
    int n = strlen(reference);
    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
            //計算來自三個方向的值
            E[j] = (E[j] + EXTEND_GAP > H[i-1][j] + OPEN_GAP) ? E[j] + EXTEND_GAP: H[i-1][j] + OPEN_GAP;
            F[i] = (F[i] + EXTEND_GAP > H[i][j-1] + OPEN_GAP) ? F[i] + EXTEND_GAP : H[i][j-1] + OPEN_GAP;
            int match = (query[i-1] == reference[j-1])? MATCH : MISMATCH;
            H[i][j] = H[i-1][j-1] + match;

            /*
                H[i][j] = max{up, left, dig, 0}
            */
            H[i][j] = max4(E[j], F[i], H[i][j], 0);
            
            // check global max
            if(H[i][j] >= result -> maxScore){
                result -> maxScore = H[i][j];
                result -> row = i;
                result -> col = start_col + j;
            }

        }
    }

    
}


void printMatrix(int **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {  
        for (int j = 0; j < cols; j++) {  
            printf("%d  ", matrix[i][j]);  
        }
        printf("\n");
    }
}

//讀字串
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

void move_data(int **H, int mtDNA_len, int slice_len){
    for(int i = 1; i <= mtDNA_len; i++){
        H[i][0] = H[i][slice_len];
    }
}

void copy_data(int **dst, int **src, int mtDNA_len){
    for(int i = 1; i <= mtDNA_len; i++){
        dst[i][0] = src[i][0];
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