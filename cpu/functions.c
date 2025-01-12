#include "functions.h"
#include <limits.h>
#include <stddef.h>

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


Cell** initialize_matrix(char *reference, char *query){
    int m = strlen(query);
    int n = strlen(reference);
    // 分配空間
    Cell **matrix = (Cell**)malloc((m + 1) * sizeof(Cell*));
    for(int i = 0; i <= m; i++){
        matrix[i] = (Cell*)malloc((n + 1) * sizeof(Cell));
    }
    matrix[0][0].score = 0;
    // initialize first row & first col to zero
    for(int i = 1; i <= m; i++){
        matrix[i][0].score = 0;
    }

    for(int j = 1; j <= n; j++){
        matrix[0][j].score = 0;
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

void local_alignment(Cell **H, int *E, int *F, char *reference, char *query, Result *result){
    int m = strlen(query);
    int n = strlen(reference);
    for(int i = 1; i <= m; i++){
        for(int j = 1; j <= n; j++){
            //計算來自三個方向的值
            E[j] = (E[j] + EXTEND_GAP > H[i-1][j].score + OPEN_GAP) ? E[j] + EXTEND_GAP: H[i-1][j].score + OPEN_GAP;
            F[i] = (F[i] + EXTEND_GAP > H[i][j-1].score + OPEN_GAP) ? F[i] + EXTEND_GAP : H[i][j-1].score + OPEN_GAP;
            int match = (query[i-1] == reference[j-1])? MATCH : MISMATCH;
            H[i][j].score = H[i-1][j-1].score + match;
            /*
                H[i][j] = max{up, left, dig, 0}
                direction : none = 0 up = 1 , dig = 2 , left = 3
            */
            H[i][j].direction = 2;
            if (E[j] > H[i][j].score){
                H[i][j].score = E[j];
                H[i][j].direction = 1;
            }
            if (F[i] > H[i][j].score){
                H[i][j].score = F[i];
                H[i][j].direction = 3;
            } 

            if (H[i][j].score < 0){
                H[i][j].score = 0;
                H[i][j].direction = 0;
            }
            
            // check global max
            if(H[i][j].score > result -> maxScore){
                result -> maxScore = H[i][j].score;
                result -> row = i;
                result -> col = j;
            }

        }
    }

    printf("max value = %d, maxI = %d, maxJ = %d\n", result -> maxScore, result->row, result->col);
}


void traceback(Cell **H, Result *result, char *reference, char *query){
    
    char * ref = malloc(sizeof(char) * (strlen(query) + strlen(reference)));
    char * q = malloc(sizeof(char) * (strlen(query) + strlen(reference)));
    
    int strIdx = 0;
    int i = result -> row;
    int j = result -> col;


    while(H[i][j].direction != 0){
        switch (H[i][j].direction)
        {
            case 1:
                ref[strIdx] = '-';
                q[strIdx] = query[i-1];
                i--;
                break;
            case 2:
                ref[strIdx] = reference[j-1];
                q[strIdx] = query[i-1];
                i--;
                j--;
                break;
            case 3:
                ref[strIdx] = reference[j-1];
                q[strIdx] = '-';
                j--;
                break;
            default:
                break;
        }

        strIdx++;        
    }

    reverseString(ref);
    reverseString(q);

    printf("%s\n", ref);
    printf("%s\n", q);
}

void printMatrix(Cell **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {  
        for (int j = 0; j < cols; j++) {  
            printf("%d  ", matrix[i][j].score);  
        }
        printf("\n");
    }
}